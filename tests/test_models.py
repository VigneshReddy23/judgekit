import pytest
from pydantic import ValidationError

from judgekit.models import EvalCase, JudgeResult, LabeledCase
from judgekit.providers import FakeProvider, Provider


def test_eval_case_optional_fields_default_to_none() -> None:
    case = EvalCase(id="c1", input="hi")
    assert case.context is None
    assert case.output is None


def test_eval_case_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate({"id": "c1", "input": "hi", "ouput": "typo"})


def test_labeled_case_only_accepts_pass_or_fail() -> None:
    LabeledCase(id="c1", input="hi", human_label="fail")
    with pytest.raises(ValidationError):
        LabeledCase.model_validate({"id": "c1", "input": "hi", "human_label": "maybe"})


def test_judge_result_round_trips_through_json() -> None:
    result = JudgeResult(
        case_id="c1", judge="toxicity", passed=True, score=1.0, reason="ok", latency_ms=1.5
    )
    assert JudgeResult.model_validate_json(result.model_dump_json()) == result


def test_fake_provider_cycles_and_records_calls() -> None:
    fake = FakeProvider(["a", "b"])
    provider: Provider = fake  # type-checks against the Protocol
    outputs = [provider.complete("p1"), provider.complete("p2", "s"), provider.complete("p3")]
    assert outputs == ["a", "b", "a"]
    assert fake.calls[1] == ("p2", "s")


def test_fake_provider_requires_responses() -> None:
    with pytest.raises(ValueError):
        FakeProvider([])
