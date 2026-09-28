"""Judge calibration: how well does an LLM judge agree with human labels?

We treat "fail" as the positive class, because catching failures is the
judge's job:
  precision (fail) = of the cases the judge failed, how many did the human fail?
  recall (fail)    = of the cases the human failed, how many did the judge catch?
  Cohen's kappa    = agreement corrected for chance (0 = coin flip, 1 = perfect).
"""

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_score, recall_score

from judgekit.judges import JUDGE_ERROR_PREFIXES, LLMJudge
from judgekit.models import LabeledCase, Verdict
from judgekit.runner import load_jsonl


class Disagreement(BaseModel):
    case_id: str
    human_label: Verdict
    judge_label: Verdict
    judge_reason: str
    note: str


class CalibrationResult(BaseModel):
    judge: str
    n: int
    human_fail_count: int
    kappa: float | None  # None when mathematically undefined (e.g. all one class)
    precision_fail: float | None
    recall_fail: float | None
    agreement: float  # raw % of cases where judge == human
    true_fail: int  # both said fail
    false_fail: int  # judge fail, human pass
    missed_fail: int  # judge pass, human fail
    true_pass: int  # both said pass
    judge_errors: int  # malformed output / provider errors (counted as fail)
    disagreements: list[Disagreement]


def load_labeled(path: Path) -> list[LabeledCase]:
    cases = load_jsonl(path, LabeledCase)
    missing = [case.id for case in cases if case.output is None]
    if missing:
        raise ValueError(f"{path}: labeled cases need an 'output' to judge; missing in {missing}")
    return cases


def _none_if_nan(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def kappa_label(kappa: float | None) -> str:
    """Landis & Koch (1977) interpretation bands, the usual convention."""
    if kappa is None:
        return "undefined"
    if kappa < 0:
        return "worse than chance"
    for upper, label in [(0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if kappa <= upper:
            return label
    return "almost perfect"


def calibrate(judge: LLMJudge, cases: list[LabeledCase], max_workers: int = 4) -> CalibrationResult:
    """Run the judge on every labeled case and compare with the human labels."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda c: judge.judge(c, c.output or ""), cases))

    # 1 = fail (the positive class), 0 = pass
    y_true = [1 if case.human_label == "fail" else 0 for case in cases]
    y_pred = [0 if result.passed else 1 for result in results]

    with warnings.catch_warnings():
        # kappa is 0/0 when both raters use one identical class; sklearn warns and
        # returns nan (the warning class varies by version). We report "undefined".
        warnings.simplefilter("ignore")
        kappa = cohen_kappa_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=np.nan)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=np.nan)
    # labels=[1, 0] fixes the layout: rows = human (fail, pass), cols = judge (fail, pass)
    (true_fail, missed_fail), (false_fail, true_pass) = confusion_matrix(
        y_true, y_pred, labels=[1, 0]
    )

    disagreements = [
        Disagreement(
            case_id=case.id,
            human_label=case.human_label,
            judge_label="pass" if result.passed else "fail",
            judge_reason=result.reason,
            note=case.note,
        )
        for case, result in zip(cases, results, strict=True)
        if (case.human_label == "pass") != result.passed
    ]
    judge_errors = sum(r.reason.startswith(JUDGE_ERROR_PREFIXES) for r in results)

    return CalibrationResult(
        judge=judge.name,
        n=len(cases),
        human_fail_count=sum(y_true),
        kappa=_none_if_nan(kappa),
        precision_fail=_none_if_nan(precision),
        recall_fail=_none_if_nan(recall),
        agreement=(len(cases) - len(disagreements)) / len(cases),
        true_fail=int(true_fail),
        false_fail=int(false_fail),
        missed_fail=int(missed_fail),
        true_pass=int(true_pass),
        judge_errors=judge_errors,
        disagreements=disagreements,
    )


def _fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def format_calibration(result: CalibrationResult) -> str:
    human_pass = result.n - result.human_fail_count
    lines = [
        f"Calibration: {result.judge} vs human labels "
        f"(n={result.n}; human fail={result.human_fail_count}, pass={human_pass})",
        "",
        f"{'metric':<28}value",
        f"{'Cohen kappa':<28}{_fmt(result.kappa)} ({kappa_label(result.kappa)})",
        f"{'Precision (fail)':<28}{_fmt(result.precision_fail)}",
        f"{'Recall (fail)':<28}{_fmt(result.recall_fail)}",
        f"{'Raw agreement':<28}{result.agreement:.3f}",
        f"{'Judge errors / malformed':<28}{result.judge_errors}",
        "",
        "Confusion matrix (rows = human, columns = judge)",
        f"{'':<14}{'judge fail':>12}{'judge pass':>12}",
        f"{'human fail':<14}{result.true_fail:>12}{result.missed_fail:>12}",
        f"{'human pass':<14}{result.false_fail:>12}{result.true_pass:>12}",
    ]
    if result.disagreements:
        lines += ["", f"Disagreements ({len(result.disagreements)}):"]
        for d in result.disagreements:
            lines.append(f"  {d.case_id}: human={d.human_label} judge={d.judge_label}")
            lines.append(f"    judge reason: {d.judge_reason}")
            if d.note:
                lines.append(f"    your note:    {d.note}")
    return "\n".join(lines)
