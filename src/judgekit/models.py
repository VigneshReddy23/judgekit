"""Pydantic data models shared across judgekit.

Pydantic validates every JSONL line at load time, so bad data fails loudly
at the boundary instead of deep inside a judge.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

Verdict = Literal["pass", "fail"]


class EvalCase(BaseModel):
    """One test case: the prompt sent to the target, plus optional RAG context.

    `output` is optional: if present we score it directly (offline eval);
    if absent the runner calls the target model to produce it.
    """

    model_config = ConfigDict(extra="forbid")  # typos in JSONL keys become errors

    id: str
    input: str
    context: str | None = None
    output: str | None = None


class LabeledCase(EvalCase):
    """An EvalCase with a human verdict, used only for judge calibration."""

    human_label: Verdict
    note: str = ""


class JudgeResult(BaseModel):
    """The outcome of one check or judge on one case."""

    case_id: str
    judge: str
    passed: bool
    score: float  # 1.0 / 0.0 today; kept as float so graded scorers fit later
    reason: str
    latency_ms: float
