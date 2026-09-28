"""Tools for building the human-labeled calibration set.

`label_cases` walks through an unlabeled "todo" file one case at a time and
records YOUR pass/fail verdict. It never shows a judge's verdict or where a
case came from, so labels aren't anchored by either.

`split_labeled` then divides a labeled file into dev (for iterating on the
rubric) and test (held out, used only for reported numbers), keeping the
pass/fail mix similar in both.
"""

import random
from collections.abc import Callable
from pathlib import Path

from judgekit.models import EvalCase, LabeledCase, Verdict
from judgekit.runner import load_jsonl

KEYS: dict[str, Verdict] = {"p": "pass", "f": "fail"}


def _append(path: Path, case: LabeledCase) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(case.model_dump_json(exclude_none=True) + "\n")


def _rewrite(path: Path, cases: list[LabeledCase]) -> None:
    path.write_text(
        "".join(c.model_dump_json(exclude_none=True) + "\n" for c in cases), encoding="utf-8"
    )


def format_case(case: EvalCase, position: int, total: int) -> str:
    lines = [f"\n[{position}/{total}] {case.id}", "", "INPUT:", case.input]
    if case.context:
        lines += ["", "CONTEXT:", case.context]
    lines += ["", "OUTPUT:", case.output or "(no output)"]
    return "\n".join(lines)


def label_cases(
    todo_path: Path,
    out_path: Path,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> int:
    """Label every case in `todo_path` not yet in `out_path`. Returns how many were labeled.

    Progress is saved after every answer, so quitting and re-running resumes
    where you left off.
    """
    todo = load_jsonl(todo_path, EvalCase)
    done: list[LabeledCase] = load_jsonl(out_path, LabeledCase) if out_path.exists() else []
    done_ids = {c.id for c in done}
    remaining = [c for c in todo if c.id not in done_ids]
    labeled_now = 0

    show(f"{len(done)} already labeled, {len(remaining)} to go.")
    show("Keys: p=pass f=fail s=skip u=undo q=quit")
    i = 0
    while i < len(remaining):
        case = remaining[i]
        show(format_case(case, len(done) + 1, len(todo)))
        key = ask("\nLabel [p/f/s/u/q] > ").strip().lower()
        if key == "q":
            break
        if key == "s":
            i += 1
            continue
        if key == "u":
            if not done:
                show("Nothing to undo.")
                continue
            undone = done.pop()
            _rewrite(out_path, done)
            labeled_now = max(labeled_now - 1, 0)
            original = next((c for c in todo if c.id == undone.id), None)
            if original is not None:
                remaining.insert(i, original)  # label it again next
            show(f"Undid {undone.id}; it comes up again next.")
            continue
        if key not in KEYS:
            show("Please type p, f, s, u or q.")
            continue
        note = ask("Note (why? Enter to skip) > ").strip()
        labeled = LabeledCase(**case.model_dump(), human_label=KEYS[key], note=note)
        _append(out_path, labeled)
        done.append(labeled)
        labeled_now += 1
        i += 1

    fails = sum(c.human_label == "fail" for c in done)
    show(f"\nSaved to {out_path}: {len(done)} labeled ({fails} fail, {len(done) - fails} pass).")
    return labeled_now


def split_labeled(path: Path, test_size: int, seed: int = 42) -> tuple[Path, Path]:
    """Write <name>.dev.jsonl and <name>.test.jsonl next to `path`.

    Stratified: each class is shuffled (with a fixed seed, so it's reproducible)
    and split in proportion, so dev and test have a similar pass/fail mix.
    """
    cases = load_jsonl(path, LabeledCase)
    if not 0 < test_size < len(cases):
        raise ValueError(f"test_size must be between 1 and {len(cases) - 1}")
    rng = random.Random(seed)
    dev: list[LabeledCase] = []
    test: list[LabeledCase] = []
    for label in ("fail", "pass"):
        group = [c for c in cases if c.human_label == label]
        rng.shuffle(group)
        n_test = round(len(group) * test_size / len(cases))
        test += group[:n_test]
        dev += group[n_test:]
    dev_path = path.with_name(f"{path.stem}.dev.jsonl")
    test_path = path.with_name(f"{path.stem}.test.jsonl")
    _rewrite(dev_path, sorted(dev, key=lambda c: c.id))
    _rewrite(test_path, sorted(test, key=lambda c: c.id))
    return dev_path, test_path
