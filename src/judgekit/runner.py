"""Load a suite, run every case through its checks and judges, apply thresholds.

Flow: load_suite -> load_cases -> run_suite (parallel over cases) -> RunResult.
All validation happens at load time so a typo fails in milliseconds, before
any money is spent on model calls.
"""

import inspect
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from judgekit.checks import CHECKS, run_check
from judgekit.judges import LLMJudge, available_judges
from judgekit.models import EvalCase, JudgeResult
from judgekit.providers import DEFAULT_JUDGE_MODEL_ID, Provider

# --- Suite config (mirrors suites/*.yaml) ------------------------------------


class CheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class TargetConfig(BaseModel):
    """The model being evaluated. Only needed when cases have no recorded output."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["bedrock"] = "bedrock"
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 1024


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    cases_file: Path  # relative paths resolve from the directory you run judgekit in
    target: TargetConfig | None = None
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID
    region: str | None = None  # None -> boto3 uses AWS_REGION / your AWS config
    checks: list[CheckConfig] = Field(default_factory=list)
    judges: list[str] = Field(default_factory=list)
    thresholds: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(default_factory=dict)
    max_workers: int = Field(default=4, ge=1)

    def scorer_names(self) -> list[str]:
        return [check.name for check in self.checks] + self.judges

    @model_validator(mode="after")
    def _validate_names(self) -> "SuiteConfig":
        for check in self.checks:
            if check.name not in CHECKS:
                raise ValueError(f"unknown check {check.name!r}; available: {sorted(CHECKS)}")
            try:  # catch missing/extra params now, not in a worker thread mid-run
                inspect.signature(CHECKS[check.name]).bind("", **check.params)
            except TypeError as exc:
                raise ValueError(f"bad params for check {check.name!r}: {exc}") from exc
        for judge in self.judges:
            if judge not in available_judges():
                raise ValueError(f"unknown judge {judge!r}; available: {available_judges()}")
        names = self.scorer_names()
        if len(names) != len(set(names)):
            raise ValueError(f"each check/judge may appear only once: {names}")
        for name in self.thresholds:
            if name not in names:
                raise ValueError(f"threshold for {name!r}, which is not a configured check/judge")
        return self


def load_suite(path: Path) -> SuiteConfig:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))  # safe_load: no code execution
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    return SuiteConfig.model_validate(data)


CaseT = TypeVar("CaseT", bound=EvalCase)


def load_jsonl(path: Path, model: type[CaseT]) -> list[CaseT]:
    """Read a JSONL file (one JSON object per line) into validated models."""
    cases: list[CaseT] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                cases.append(model.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not cases:
        raise ValueError(f"{path}: no cases found")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate case ids")
    return cases


def load_cases(path: Path) -> list[EvalCase]:
    return load_jsonl(path, EvalCase)


# --- Results ---------------------------------------------------------------


class CaseResult(BaseModel):
    case: EvalCase
    output: str
    results: list[JudgeResult]


class ScorerSummary(BaseModel):
    name: str
    total: int
    passed: int
    pass_rate: float
    threshold: float | None
    meets_threshold: bool


class RunResult(BaseModel):
    suite_name: str
    cases: list[CaseResult]
    summaries: list[ScorerSummary]
    total_latency_ms: float  # wall-clock time for the whole run

    @property
    def passed(self) -> bool:
        return all(summary.meets_threshold for summary in self.summaries)


# --- Running ---------------------------------------------------------------


def get_output(case: EvalCase, target: Provider | None) -> str:
    """Use the recorded output if present, otherwise ask the target model."""
    if case.output is not None:
        return case.output
    if target is None:
        raise ValueError(f"case {case.id!r} has no output and the suite has no target")
    system = f"Answer using only this context:\n{case.context}" if case.context else ""
    return target.complete(case.input, system=system)


def evaluate_case(
    case: EvalCase, checks: list[CheckConfig], judges: list[LLMJudge], target: Provider | None
) -> CaseResult:
    try:
        output = get_output(case, target)
    except Exception as exc:
        # The target failed, so nothing can be scored: fail every scorer with the reason.
        reason = f"target error: {type(exc).__name__}: {exc}"
        names = [check.name for check in checks] + [judge.name for judge in judges]
        failed = [
            JudgeResult(
                case_id=case.id, judge=name, passed=False, score=0.0, reason=reason, latency_ms=0.0
            )
            for name in names
        ]
        return CaseResult(case=case, output="", results=failed)

    results = [run_check(check.name, case.id, output, check.params) for check in checks]
    results += [judge.judge(case, output) for judge in judges]
    return CaseResult(case=case, output=output, results=results)


def summarize(
    case_results: list[CaseResult], names: list[str], thresholds: dict[str, float]
) -> list[ScorerSummary]:
    summaries = []
    for name in names:
        results = [r for cr in case_results for r in cr.results if r.judge == name]
        passed = sum(r.passed for r in results)
        rate = passed / len(results) if results else 0.0
        threshold = thresholds.get(name)
        # isclose guards against float rounding, e.g. 0.1 * 3 != 0.3
        meets = threshold is None or rate >= threshold or math.isclose(rate, threshold)
        summaries.append(
            ScorerSummary(
                name=name,
                total=len(results),
                passed=passed,
                pass_rate=rate,
                threshold=threshold,
                meets_threshold=meets,
            )
        )
    return summaries


def run_suite(
    suite: SuiteConfig,
    cases: list[EvalCase],
    judges: list[LLMJudge],
    target: Provider | None = None,
) -> RunResult:
    """Evaluate all cases in parallel threads and summarise per scorer."""
    start = time.perf_counter()
    # Threads suit this workload: each case mostly waits on network I/O, and
    # pool.map returns results in input order, so output is deterministic.
    with ThreadPoolExecutor(max_workers=suite.max_workers) as pool:
        case_results = list(
            pool.map(lambda case: evaluate_case(case, suite.checks, judges, target), cases)
        )
    total_latency_ms = (time.perf_counter() - start) * 1000
    return RunResult(
        suite_name=suite.name,
        cases=case_results,
        summaries=summarize(case_results, suite.scorer_names(), suite.thresholds),
        total_latency_ms=total_latency_ms,
    )
