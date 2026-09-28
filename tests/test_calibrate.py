"""Calibration tests. All data here is a SYNTHETIC fixture, not real results."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from judgekit import cli
from judgekit.calibrate import calibrate, format_calibration, kappa_label, load_labeled
from judgekit.judges import LLMJudge
from judgekit.models import LabeledCase, Verdict
from judgekit.providers import ModelConfig

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_toxicity_labels.jsonl"
PASS = '{"reason": "fine", "verdict": "pass"}'
FAIL = '{"reason": "toxic", "verdict": "fail"}'

# Scripted judge verdicts for the fixture, chosen to give a known confusion matrix:
#   3 true fails, 1 missed fail (sarcasm), 1 false fail (slur discussion), 3 true passes.
FIXTURE_JUDGE_FAILS = [
    "You're an idiot",
    "People like you",
    "Shut up",
    "That word is a slur",
]


class KeywordProvider:
    """Fails any output containing a keyword. Deterministic under threads,
    unlike FakeProvider's call-order responses."""

    def __init__(self, fail_keywords: list[str]) -> None:
        self.fail_keywords = fail_keywords

    def complete(self, prompt: str, system: str = "") -> str:
        output = prompt.split("<output>")[1]
        return FAIL if any(k in output for k in self.fail_keywords) else PASS


def labeled(label: Verdict, output: str = "x", i: int = 0) -> LabeledCase:
    return LabeledCase(id=f"c{i}", input="q", output=output, human_label=label)


def test_fixture_metrics_match_hand_calculation() -> None:
    judge = LLMJudge("toxicity", KeywordProvider(FIXTURE_JUDGE_FAILS))
    result = calibrate(judge, load_labeled(FIXTURE))

    assert (result.true_fail, result.missed_fail, result.false_fail, result.true_pass) == (
        3,
        1,
        1,
        3,
    )
    assert result.n == 8
    assert result.human_fail_count == 4
    # precision = 3 / (3 + 1); recall = 3 / (3 + 1); agreement = 6 / 8
    assert result.precision_fail == pytest.approx(0.75)
    assert result.recall_fail == pytest.approx(0.75)
    assert result.agreement == pytest.approx(0.75)
    # kappa = (p_o - p_e) / (1 - p_e) = (0.75 - 0.5) / (1 - 0.5) = 0.5
    assert result.kappa == pytest.approx(0.5)
    assert result.judge_errors == 0
    assert {d.case_id for d in result.disagreements} == {"fx-fail-4", "fx-pass-4"}


def test_perfect_judge() -> None:
    cases = [labeled("fail", "bad", 0), labeled("pass", "good", 1)]
    result = calibrate(LLMJudge("toxicity", KeywordProvider(["bad"])), cases)
    assert result.kappa == pytest.approx(1.0)
    assert result.disagreements == []


def test_judge_that_always_passes_has_undefined_precision_and_zero_recall() -> None:
    cases = [labeled("fail", i=0), labeled("pass", i=1)]
    result = calibrate(LLMJudge("toxicity", KeywordProvider([])), cases)
    assert result.precision_fail is None  # judge never said fail: 0/0
    assert result.recall_fail == 0.0
    assert result.kappa == pytest.approx(0.0)


def test_single_class_labels_give_undefined_kappa() -> None:
    cases = [labeled("pass", i=0), labeled("pass", i=1)]
    result = calibrate(LLMJudge("toxicity", KeywordProvider([])), cases)
    assert result.kappa is None
    assert result.recall_fail is None
    assert "undefined" in format_calibration(result)


class GarbageProvider:
    def complete(self, prompt: str, system: str = "") -> str:
        return "not json"


def test_malformed_judge_output_is_counted() -> None:
    cases = [labeled("fail", i=0), labeled("pass", i=1)]
    result = calibrate(LLMJudge("toxicity", GarbageProvider()), cases)
    assert result.judge_errors == 2


def test_load_labeled_requires_output(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    path.write_text('{"id": "a", "input": "q", "human_label": "fail"}\n')
    with pytest.raises(ValueError, match="need an 'output'"):
        load_labeled(path)


def test_load_labeled_rejects_bad_label(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    path.write_text('{"id": "a", "input": "q", "output": "x", "human_label": "Fail"}\n')
    with pytest.raises(ValueError, match=r"l\.jsonl:1"):
        load_labeled(path)


@pytest.mark.parametrize(
    ("kappa", "label"),
    [
        (None, "undefined"),
        (-0.1, "worse than chance"),
        (0.1, "slight"),
        (0.3, "fair"),
        (0.5, "moderate"),
        (0.7, "substantial"),
        (0.9, "almost perfect"),
    ],
)
def test_kappa_label(kappa: float | None, label: str) -> None:
    assert kappa_label(kappa) == label


def test_format_shows_table_and_disagreements() -> None:
    judge = LLMJudge("toxicity", KeywordProvider(FIXTURE_JUDGE_FAILS))
    text = format_calibration(calibrate(judge, load_labeled(FIXTURE)))
    assert "Cohen kappa                 0.500 (moderate)" in text
    assert "human fail               3           1" in text
    assert "human pass               1           3" in text
    assert "fx-fail-4: human=fail judge=pass" in text
    assert "your note:    SYNTHETIC TEST FIXTURE: sarcasm" in text


# --- CLI -------------------------------------------------------------------

runner = CliRunner()


def test_cli_calibrate(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[tuple[str, Any]] = []

    def fake_make_provider(config: ModelConfig) -> KeywordProvider:
        created.append((config.provider, config.model))
        return KeywordProvider(FIXTURE_JUDGE_FAILS)

    monkeypatch.setattr(cli, "make_provider", fake_make_provider)
    result = runner.invoke(
        cli.app,
        [
            "calibrate",
            "--judge",
            "toxicity",
            "--data",
            str(FIXTURE),
            "--model",
            "m",
            "--workers",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert created == [("bedrock", "m")]
    assert "Calibration: toxicity vs human labels (n=8; human fail=4, pass=4)" in result.output


def test_cli_calibrate_unknown_judge() -> None:
    result = runner.invoke(cli.app, ["calibrate", "--judge", "nope", "--data", str(FIXTURE)])
    assert result.exit_code == 2
    assert "unknown judge" in result.output


def test_cli_calibrate_bad_data(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    path.write_text("not json\n")
    result = runner.invoke(cli.app, ["calibrate", "--judge", "toxicity", "--data", str(path)])
    assert result.exit_code == 2
    assert "config error" in result.output
