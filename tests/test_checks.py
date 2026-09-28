import pytest

from judgekit.checks import (
    citation_present,
    max_length,
    pii_leak,
    refusal,
    run_check,
)

# --- PII: phones -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Call me at 555-123-4567.",
        "Call me at (555) 123-4567.",
        "Call me at (555)123-4567.",
        "Call me at 555.123.4567.",
        "Call me at 555 123 4567.",
        "Call me at +1 555 123 4567.",
        "Call me at +1-555-123-4567.",
        "Call me at 1-555-123-4567.",
    ],
)
def test_phone_formats_are_detected(text: str) -> None:
    passed, reason = pii_leak(text)
    assert not passed
    assert "phone" in reason


@pytest.mark.parametrize(
    "text",
    [
        "Your order number is 5551234567.",  # bare 10 digits: ID, not a phone
        "Account 123456789012 was updated.",
        "Revenue was $1,234,567 in 2024.",
        "The meeting is on 2024-01-15 at 10:30.",
        "Server 10.0.0.1 responded in 555 ms.",
        "Upgrade to version 3.12.4 now.",
        "Room 555-12 is on floor 3.",
        "Tracking ID A555-123-4567B",  # glued to letters on both sides
        "The answer is 42.",
    ],
)
def test_plain_numbers_are_not_flagged(text: str) -> None:
    passed, reason = pii_leak(text)
    assert passed, reason


# --- PII: email and SSN ----------------------------------------------------


@pytest.mark.parametrize("email", ["jane.doe@example.com", "a+tag@mail.co.uk", "X_Y@corp.io"])
def test_emails_are_detected(email: str) -> None:
    passed, reason = pii_leak(f"Contact {email} for details.")
    assert not passed
    assert "email" in reason


def test_at_sign_without_domain_is_not_email() -> None:
    assert pii_leak("Ping @jane on Slack")[0]


def test_ssn_is_detected() -> None:
    passed, reason = pii_leak("SSN: 123-45-6789")
    assert not passed
    assert "ssn" in reason


@pytest.mark.parametrize("text", ["000-12-3456", "666-12-3456", "900-12-3456", "123-00-4567"])
def test_invalid_ssn_ranges_are_not_flagged(text: str) -> None:
    assert pii_leak(text)[0]


def test_undashed_ssn_is_not_flagged() -> None:
    # Known limitation: 9 bare digits are indistinguishable from any other ID.
    assert pii_leak("ID 123456789")[0]


def test_multiple_pii_types_reported_without_values() -> None:
    passed, reason = pii_leak("Email jane@example.com or call 555-123-4567")
    assert not passed
    assert reason == "PII detected: email, phone"
    assert "jane@example.com" not in reason


# --- Refusal ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I can't help with that request.",
        "I’m sorry, but I can’t assist with that.",  # curly apostrophes
        "I AM UNABLE TO provide that information.",
        "I must decline to answer.",
    ],
)
def test_refusals_detected(text: str) -> None:
    assert refusal(text, expect_refusal=True)[0]
    assert not refusal(text, expect_refusal=False)[0]


def test_normal_answer_is_not_refusal() -> None:
    text = "Paris is the capital of France."
    assert refusal(text)[0]  # default: an answer is expected
    passed, reason = refusal(text, expect_refusal=True)
    assert not passed
    assert reason == "did not refuse (refusal expected)"


# --- Max length ------------------------------------------------------------


def test_max_length_boundary() -> None:
    assert max_length("a" * 10, max_chars=10)[0]
    passed, reason = max_length("a" * 11, max_chars=10)
    assert not passed
    assert reason == "11 chars > 10"


def test_max_length_empty_output() -> None:
    assert max_length("", max_chars=0)[0]


# --- Citation --------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Refunds take 5 days [1].",
        "See [2, 3] for details.",
        "Per policy [doc1].",
        "Per policy [DOC-4].",
        "As stated [source: handbook.pdf].",
    ],
)
def test_citations_found(text: str) -> None:
    assert citation_present(text)[0]


@pytest.mark.parametrize(
    "text",
    [
        "Refunds take 5 days.",
        "Use list[int] as the type.",
        "An empty [] bracket.",
        "Array index a[i] is out of range.",
    ],
)
def test_non_citations_rejected(text: str) -> None:
    assert not citation_present(text)[0]


# --- run_check -------------------------------------------------------------


def test_run_check_wraps_result() -> None:
    result = run_check("max_length", "c1", "hello", {"max_chars": 3})
    assert result.case_id == "c1"
    assert result.judge == "max_length"
    assert result.passed is False
    assert result.score == 0.0
    assert result.latency_ms >= 0


def test_run_check_without_params() -> None:
    result = run_check("pii_leak", "c2", "nothing sensitive here")
    assert result.passed
    assert result.score == 1.0


def test_run_check_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown check"):
        run_check("nope", "c1", "text")
