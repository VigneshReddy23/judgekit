from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from typer.testing import CliRunner

from judgekit import cli, tracing
from judgekit.judges import LLMJudge
from judgekit.models import EvalCase
from judgekit.providers import FakeProvider
from judgekit.runner import SuiteConfig, run_suite

PASS = '{"reason": "ok", "verdict": "pass"}'


class ModelFakeProvider(FakeProvider):
    model_id = "fake-judge-model"


def by_name(spans: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in spans.get_finished_spans() if s.name == name]


def run_two_cases(workers: int = 2) -> None:
    suite = SuiteConfig.model_validate(
        {
            "name": "traced",
            "cases_file": "unused.jsonl",
            "checks": [{"name": "pii_leak"}],
            "judges": ["toxicity"],
            "max_workers": workers,
        }
    )
    cases = [
        EvalCase(id="a", input="q", output="fine"),
        EvalCase(id="b", input="q", output="mail a@b.com"),
    ]
    run_suite(suite, cases, [LLMJudge("toxicity", ModelFakeProvider([PASS]))])


def test_span_tree_is_run_then_case_then_judge(spans: InMemorySpanExporter) -> None:
    run_two_cases()
    [run] = by_name(spans, "eval.run")
    cases = by_name(spans, "eval.case")
    judges = by_name(spans, "judge toxicity")
    assert len(cases) == 2 and len(judges) == 2

    assert run.context is not None
    # Every case is a child of the run, even though cases ran in worker threads.
    for case in cases:
        assert case.parent is not None and case.parent.span_id == run.context.span_id
        assert case.context is not None and case.context.trace_id == run.context.trace_id
    # Every judge span is a child of its own case span.
    case_ids = {c.context.span_id for c in cases if c.context is not None}
    assert {j.parent.span_id for j in judges if j.parent is not None} == case_ids


def test_span_attributes(spans: InMemorySpanExporter) -> None:
    run_two_cases()
    [run] = by_name(spans, "eval.run")
    assert run.attributes is not None
    assert run.attributes["judgekit.suite.name"] == "traced"
    assert run.attributes["judgekit.case.count"] == 2
    assert run.attributes["judgekit.run.passed"] is True  # no thresholds configured

    case_passed = {
        s.attributes["judgekit.case.id"]: s.attributes["judgekit.case.passed"]
        for s in by_name(spans, "eval.case")
        if s.attributes is not None
    }
    assert case_passed == {"a": True, "b": False}  # b leaks an email

    judge = by_name(spans, "judge toxicity")[0]
    assert judge.attributes is not None
    assert judge.attributes["judgekit.judge.verdict"] == "pass"
    assert judge.attributes["gen_ai.request.model"] == "fake-judge-model"
    assert isinstance(judge.attributes["judgekit.judge.latency_ms"], float)
    assert judge.resource.attributes["service.name"] == "judgekit"


def test_malformed_judge_output_marks_span_as_error(spans: InMemorySpanExporter) -> None:
    LLMJudge("toxicity", FakeProvider(["not json"])).judge(EvalCase(id="x", input="q"), "o")
    [span] = by_name(spans, "judge toxicity")
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes["gen_ai.request.model"] == "unknown"  # fake has no model_id


class ExplodingTarget:
    model_id = "target-model"

    def complete(self, prompt: str, system: str = "") -> str:
        raise ConnectionError("down")


def test_target_span_and_target_error(spans: InMemorySpanExporter) -> None:
    suite = SuiteConfig.model_validate(
        {"name": "t", "cases_file": "x", "checks": [{"name": "pii_leak"}]}
    )
    run_suite(suite, [EvalCase(id="a", input="q")], [], ExplodingTarget())
    [target] = by_name(spans, "target.generate")
    assert target.attributes is not None
    assert target.attributes["gen_ai.request.model"] == "target-model"
    assert target.status.status_code == StatusCode.ERROR  # exception recorded by OTel
    [case] = by_name(spans, "eval.case")
    assert case.status.status_code == StatusCode.ERROR


def test_cli_trace_flag_sets_up_and_flushes(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class StubProvider:
        def shutdown(self) -> None:
            calls.append("shutdown")

    def fake_setup() -> StubProvider:
        calls.append("setup")
        return StubProvider()

    monkeypatch.setattr(cli, "setup_tracing", fake_setup)
    cases = tmp_path / "c.jsonl"
    cases.write_text('{"id": "a", "input": "q", "output": "hi"}\n')
    suite = tmp_path / "s.yaml"
    suite.write_text(f"name: t\ncases_file: {cases}\nchecks: [{{name: pii_leak}}]\n")

    result = CliRunner().invoke(cli.app, ["run", str(suite), "--trace"])
    assert result.exit_code == 0, result.output
    assert calls == ["setup", "shutdown"]


def test_setup_tracing_installs_otlp_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    installed: list[Any] = []
    monkeypatch.setattr(trace, "set_tracer_provider", installed.append)
    provider = tracing.setup_tracing()
    assert installed == [provider]
    assert provider.resource.attributes["service.name"] == "judgekit"
    provider.shutdown()
