import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from judgekit.judges import LLMJudge
from judgekit.models import EvalCase
from judgekit.providers import FakeProvider
from judgekit.runner import SuiteConfig, load_cases, load_suite, run_suite, summarize

REPO_ROOT = Path(__file__).parent.parent
PASS = '{"reason": "ok", "verdict": "pass"}'
FAIL = '{"reason": "bad", "verdict": "fail"}'


def make_suite(**overrides: Any) -> SuiteConfig:
    data: dict[str, Any] = {"name": "t", "cases_file": "unused.jsonl"}
    data.update(overrides)
    return SuiteConfig.model_validate(data)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


# --- Suite validation --------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["example.yaml", "example_offline.yaml", "example_anthropic.yaml", "example_ollama.yaml"],
)
def test_repo_example_suites_are_valid(name: str) -> None:
    suite = load_suite(REPO_ROOT / "suites" / name)
    assert load_cases(REPO_ROOT / suite.cases_file)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"checks": [{"name": "nope"}]}, "unknown check"),
        ({"judges": ["nope"]}, "unknown judge"),
        ({"checks": [{"name": "max_length"}]}, "bad params"),  # missing max_chars
        ({"checks": [{"name": "pii_leak", "params": {"x": 1}}]}, "bad params"),
        ({"checks": [{"name": "pii_leak"}, {"name": "pii_leak"}]}, "only once"),
        ({"thresholds": {"toxicity": 0.9}}, "not a configured"),
        ({"judges": ["toxicity"], "thresholds": {"toxicity": 1.5}}, "less than or equal"),
        ({"max_workers": 0}, "greater than or equal"),
        ({"treshold": {}}, "Extra inputs"),  # typo in a top-level key
    ],
)
def test_invalid_suites_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_suite(**overrides)


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: [unclosed")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_suite(path)


# --- Case loading ------------------------------------------------------------


def test_load_cases_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text('{"id": "a", "input": "q"}\n\n{"id": "b", "input": "q"}\n')
    assert [c.id for c in load_cases(path)] == ["a", "b"]


def test_load_cases_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text('{"id": "a", "input": "q"}\n{"id": "b"}\n')
    with pytest.raises(ValueError, match=r"c\.jsonl:2"):
        load_cases(path)


def test_load_cases_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no cases"):
        load_cases(path)


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "c.jsonl", [{"id": "a", "input": "q"}] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


# --- Running -----------------------------------------------------------------


def test_run_suite_combines_checks_and_judges() -> None:
    suite = make_suite(
        checks=[{"name": "pii_leak"}],
        judges=["toxicity"],
        thresholds={"pii_leak": 1.0, "toxicity": 0.5},
    )
    cases = [
        EvalCase(id="clean", input="q", output="Hello there."),
        EvalCase(id="leak", input="q", output="Mail me at a@b.com"),
    ]
    judge = LLMJudge("toxicity", FakeProvider([PASS]))
    result = run_suite(suite, cases, [judge])

    assert [cr.case.id for cr in result.cases] == ["clean", "leak"]  # input order kept
    pii, tox = result.summaries
    assert (pii.name, pii.passed, pii.total, pii.meets_threshold) == ("pii_leak", 1, 2, False)
    assert (tox.name, tox.pass_rate, tox.meets_threshold) == ("toxicity", 1.0, True)
    assert not result.passed
    assert result.total_latency_ms >= 0


def test_scorer_without_threshold_never_fails_gate() -> None:
    suite = make_suite(judges=["toxicity"])
    result = run_suite(
        suite,
        [EvalCase(id="a", input="q", output="x")],
        [LLMJudge("toxicity", FakeProvider([FAIL]))],
    )
    assert result.summaries[0].pass_rate == 0.0
    assert result.summaries[0].threshold is None
    assert result.passed


def test_threshold_equal_to_pass_rate_passes() -> None:
    # 3 of 10 pass = 0.3; float division must not make 0.3 < 0.3.
    suite = make_suite(judges=["toxicity"], thresholds={"toxicity": 0.3})
    cases = [EvalCase(id=str(i), input="q", output="x") for i in range(10)]
    judge = LLMJudge("toxicity", FakeProvider([PASS, PASS, PASS] + [FAIL] * 7))
    result = run_suite(suite, cases, [judge])
    assert result.summaries[0].passed == 3
    assert result.passed


def test_summarize_with_no_results() -> None:
    [summary] = summarize([], ["pii_leak"], {"pii_leak": 0.5})
    assert summary.total == 0
    assert not summary.meets_threshold


def test_target_generates_missing_outputs_with_context() -> None:
    suite = make_suite(checks=[{"name": "pii_leak"}])
    target = FakeProvider(["generated answer"])
    case = EvalCase(id="a", input="question?", context="the docs")
    result = run_suite(suite, [case], [], target)
    assert result.cases[0].output == "generated answer"
    assert target.calls == [("question?", "Answer using only this context:\nthe docs")]


def test_recorded_output_skips_target() -> None:
    target = FakeProvider(["should not be used"])
    result = run_suite(make_suite(), [EvalCase(id="a", input="q", output="recorded")], [], target)
    assert result.cases[0].output == "recorded"
    assert target.calls == []


def test_missing_output_without_target_fails_every_scorer() -> None:
    suite = make_suite(checks=[{"name": "pii_leak"}], judges=["toxicity"])
    judge = LLMJudge("toxicity", FakeProvider([PASS]))
    result = run_suite(suite, [EvalCase(id="a", input="q")], [judge])
    reasons = [r.reason for r in result.cases[0].results]
    assert len(reasons) == 2
    assert all("has no output and the suite has no target" in r for r in reasons)


class ExplodingProvider:
    def complete(self, prompt: str, system: str = "") -> str:
        raise ConnectionError("target down")


def test_target_error_fails_case_without_crashing() -> None:
    suite = make_suite(checks=[{"name": "pii_leak"}])
    cases = [EvalCase(id="a", input="q"), EvalCase(id="b", input="q", output="fine")]
    result = run_suite(suite, cases, [], ExplodingProvider())
    assert result.cases[0].results[0].reason == "target error: ConnectionError: target down"
    assert result.cases[1].results[0].passed


def test_suite_yaml_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({"name": "t", "cases_file": "c.jsonl", "judges": ["toxicity"]}))
    suite = load_suite(path)
    assert suite.judges == ["toxicity"]
    assert suite.judge.provider == "bedrock"
    assert suite.judge.model.startswith("us.anthropic.claude-haiku-4-5")
