from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from judgekit import cli
from judgekit.judges import LLMJudge
from judgekit.models import EvalCase
from judgekit.providers import BedrockProvider, FakeProvider
from judgekit.report import ModelUsage, collect_usage, render_report, total_cost
from judgekit.runner import ModelPricing, SuiteConfig, run_suite

FAIL = '{"reason": "demeaning tone", "verdict": "fail"}'


def make_result(outputs: dict[str, str], **suite_overrides: Any) -> tuple[Any, SuiteConfig]:
    data: dict[str, Any] = {
        "name": "report-test",
        "cases_file": "unused.jsonl",
        "checks": [{"name": "pii_leak"}],
        "judges": ["toxicity"],
        "thresholds": {"pii_leak": 1.0},
    }
    data.update(suite_overrides)
    suite = SuiteConfig.model_validate(data)
    cases = [EvalCase(id=case_id, input="q", output=out) for case_id, out in outputs.items()]
    judges = [LLMJudge(name, FakeProvider([FAIL])) for name in suite.judges]
    result = run_suite(suite, cases, judges)
    return result, suite


def test_report_shows_scorers_status_and_failures() -> None:
    result, suite = make_result({"clean": "hello", "leak": "mail a@b.com"})
    html = render_report(result, suite)
    assert "<title>judgekit · report-test</title>" in html
    assert "Failed: 1 scorer below threshold" in html
    assert 'pii_leak<div class="kind">check</div>' in html
    assert 'toxicity<div class="kind">LLM judge</div>' in html
    assert "below threshold" in html and "reported only" in html  # toxicity has no threshold
    assert "50.0% (1/2)" in html
    assert "<strong>pii_leak</strong>: PII detected: email" in html
    assert "<strong>toxicity</strong>: demeaning tone" in html
    assert 'class="marker" style="left: 100.00%"' in html


def test_report_escapes_untrusted_model_output() -> None:
    result, suite = make_result({"xss": "<script>alert('pwned')</script>"})
    html = render_report(result, suite)
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(&#39;pwned&#39;)&lt;/script&gt;" in html


def test_passed_report_has_no_failure_section() -> None:
    result, suite = make_result({"clean": "hello"}, judges=[])
    html = render_report(result, suite)
    assert "Passed: every scorer met its threshold" in html
    assert "No failures." in html
    assert "Model usage" not in html  # no model calls, no usage table


def test_cost_not_configured_without_pricing() -> None:
    result, suite = make_result({"a": "hi"})
    usage = [
        ModelUsage(role="judge", model_id="m", input_tokens=10, output_tokens=5, cost_usd=None)
    ]
    html = render_report(result, suite, usage)
    assert "not configured" in html
    assert "no pricing" in html
    assert "10 in · 5 out" in html


class FakeBedrockClient:
    def converse(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "output": {"message": {"content": [{"text": "x"}]}},
            "usage": {"inputTokens": 2_000_000, "outputTokens": 100_000},
        }


def test_collect_usage_applies_per_model_pricing() -> None:
    # Arbitrary test prices, NOT real Bedrock prices.
    priced = BedrockProvider("priced-model", client=FakeBedrockClient())
    unpriced = BedrockProvider("unpriced-model", client=FakeBedrockClient())
    priced.complete("p")
    unpriced.complete("p")
    pricing = {"priced-model": ModelPricing(input_per_million_usd=3.0, output_per_million_usd=10.0)}

    judge, target = collect_usage([("judge", priced), ("target", unpriced)], pricing)
    assert judge.cost_usd == pytest.approx(2 * 3.0 + 0.1 * 10.0)  # = 7.0
    assert (target.role, target.cost_usd) == ("target", None)
    assert total_cost([judge]) == pytest.approx(7.0)
    assert total_cost([judge, target]) is None  # a partial sum would understate cost
    assert total_cost([]) == 0.0  # no model calls at all


def test_configured_cost_is_rendered() -> None:
    result, suite = make_result({"a": "hi"})
    usage = [ModelUsage(role="judge", model_id="m", input_tokens=1, output_tokens=1, cost_usd=0.5)]
    assert "$0.5000" in render_report(result, suite, usage)


def test_negative_pricing_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SuiteConfig.model_validate(
            {
                "name": "t",
                "cases_file": "x",
                "pricing": {"m": {"input_per_million_usd": -1, "output_per_million_usd": 1}},
            }
        )


def test_cli_writes_report(tmp_path: Path) -> None:
    cases = tmp_path / "c.jsonl"
    cases.write_text('{"id": "a", "input": "q", "output": "hi"}\n')
    suite = tmp_path / "s.yaml"
    suite.write_text(f"name: cli-report\ncases_file: {cases}\nchecks: [{{name: pii_leak}}]\n")
    report = tmp_path / "report.html"

    result = CliRunner().invoke(cli.app, ["run", str(suite), "--report", str(report)])
    assert result.exit_code == 0, result.output
    assert f"Report written to {report}" in result.output
    assert "cli-report" in report.read_text()


def test_cli_report_includes_judge_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_bedrock(model_id: str, **kwargs: Any) -> BedrockProvider:
        return BedrockProvider(model_id, client=FakeBedrockClient())

    monkeypatch.setattr(cli, "BedrockProvider", fake_bedrock)
    cases = tmp_path / "c.jsonl"
    cases.write_text('{"id": "a", "input": "q"}\n')
    suite = tmp_path / "s.yaml"
    suite.write_text(
        f"name: t\ncases_file: {cases}\njudges: [toxicity]\njudge_model_id: jm\n"
        "target: {model_id: tm}\n"
    )
    report = tmp_path / "r.html"
    CliRunner().invoke(cli.app, ["run", str(suite), "--report", str(report)])
    html = report.read_text()
    assert "<code>jm</code>" in html and "<code>tm</code>" in html
