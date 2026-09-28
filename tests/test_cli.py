import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from judgekit import cli
from judgekit.providers import FakeProvider, ModelConfig

runner = CliRunner()
REPO_ROOT = Path(__file__).parent.parent


def write_suite(tmp_path: Path, outputs: list[str], **overrides: Any) -> Path:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join(
            json.dumps({"id": f"c{i}", "input": "q", "output": out})
            for i, out in enumerate(outputs)
        )
    )
    suite: dict[str, Any] = {
        "name": "cli-test",
        "cases_file": str(cases),
        "checks": [{"name": "pii_leak"}],
        "thresholds": {"pii_leak": 1.0},
    }
    suite.update(overrides)
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    return path


def test_exit_0_when_all_thresholds_met(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["run", str(write_suite(tmp_path, ["hello", "hi"]))])
    assert result.exit_code == 0, result.output
    assert "RESULT: PASSED" in result.output
    assert "100.0% (2/2)" in result.output


def test_exit_1_when_below_threshold(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["run", str(write_suite(tmp_path, ["hi", "a@b.com"]))])
    assert result.exit_code == 1
    assert "BELOW THRESHOLD" in result.output
    assert "[pii_leak] c1: PII detected: email" in result.output
    assert "RESULT: FAILED (1 below threshold)" in result.output


def test_long_failure_list_is_truncated(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["run", str(write_suite(tmp_path, ["a@b.com"] * 12))])
    assert result.output.count("[pii_leak]") == cli.MAX_FAILURES_SHOWN
    assert "... and 2 more" in result.output


def test_exit_2_on_invalid_suite(tmp_path: Path) -> None:
    path = write_suite(tmp_path, ["hi"], judges=["not_a_judge"])
    result = runner.invoke(cli.app, ["run", str(path)])
    assert result.exit_code == 2
    assert "unknown judge" in result.output


def test_exit_2_on_missing_cases_file(tmp_path: Path) -> None:
    path = write_suite(tmp_path, ["hi"], cases_file=str(tmp_path / "missing.jsonl"))
    assert runner.invoke(cli.app, ["run", str(path)]).exit_code == 2


def test_missing_suite_file_is_a_usage_error(tmp_path: Path) -> None:
    assert runner.invoke(cli.app, ["run", str(tmp_path / "nope.yaml")]).exit_code == 2


def test_judges_use_bedrock_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    def fake_make_provider(config: ModelConfig) -> FakeProvider:
        created.append(config.model)
        return FakeProvider(['{"reason": "rude", "verdict": "fail"}'])

    monkeypatch.setattr(cli, "make_provider", fake_make_provider)
    path = write_suite(
        tmp_path, ["hi"], judges=["toxicity"], judge={"model": "judge-model"}, thresholds={}
    )
    result = runner.invoke(cli.app, ["run", str(path)])
    assert created == ["judge-model"]
    assert "[toxicity] c0: rude" in result.output
    assert result.exit_code == 0  # no threshold on toxicity, so the gate passes


def test_target_is_built_from_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[ModelConfig] = []

    def fake_make_provider(config: ModelConfig) -> FakeProvider:
        created.append(config)
        return FakeProvider(["generated"])

    monkeypatch.setattr(cli, "make_provider", fake_make_provider)
    path = write_suite(tmp_path, [], target={"model": "target-model", "max_tokens": 50})
    (tmp_path / "cases.jsonl").write_text('{"id": "c0", "input": "q"}\n')  # no output: target runs
    result = runner.invoke(cli.app, ["run", str(path)])
    assert result.exit_code == 0, result.output
    assert [(c.provider, c.model, c.max_tokens) for c in created] == [
        ("bedrock", "target-model", 50)
    ]


def test_repo_offline_example_fails_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    result = runner.invoke(cli.app, ["run", "suites/example_offline.yaml"])
    assert result.exit_code == 1
    assert "[pii_leak] support-contact-leak" in result.output
