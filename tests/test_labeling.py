"""Labeling tool tests. Keypresses are scripted; no real labels are produced."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from judgekit import cli
from judgekit.labeling import label_cases, split_labeled
from judgekit.models import LabeledCase
from judgekit.runner import load_jsonl


def write_todo(path: Path, n: int) -> Path:
    rows = [{"id": f"c{i}", "input": f"q{i}", "output": f"a{i}"} for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def keys(*answers: str) -> Callable[[str], str]:
    it: Iterator[str] = iter(answers)
    return lambda prompt: next(it)


def run(todo: Path, out: Path, *answers: str) -> list[str]:
    shown: list[str] = []
    label_cases(todo, out, ask=keys(*answers), show=shown.append)
    return shown


def test_labels_are_saved_with_notes(tmp_path: Path) -> None:
    todo, out = write_todo(tmp_path / "t.jsonl", 2), tmp_path / "o.jsonl"
    run(todo, out, "p", "", "F", "sarcasm")
    labeled = load_jsonl(out, LabeledCase)
    assert [(c.id, c.human_label, c.note) for c in labeled] == [
        ("c0", "pass", ""),
        ("c1", "fail", "sarcasm"),
    ]
    assert labeled[0].output == "a0"  # case data carried over unchanged


def test_quit_and_resume(tmp_path: Path) -> None:
    todo, out = write_todo(tmp_path / "t.jsonl", 3), tmp_path / "o.jsonl"
    run(todo, out, "p", "", "q")
    shown = run(todo, out, "f", "", "p", "")
    assert "1 already labeled, 2 to go" in shown[0]
    assert [c.id for c in load_jsonl(out, LabeledCase)] == ["c0", "c1", "c2"]


def test_skip_leaves_case_for_later(tmp_path: Path) -> None:
    todo, out = write_todo(tmp_path / "t.jsonl", 2), tmp_path / "o.jsonl"
    run(todo, out, "s", "p", "")
    assert [c.id for c in load_jsonl(out, LabeledCase)] == ["c1"]


def test_undo_relabels_previous_case(tmp_path: Path) -> None:
    todo, out = write_todo(tmp_path / "t.jsonl", 2), tmp_path / "o.jsonl"
    # label c0 pass, undo it, relabel c0 fail, then c1 pass
    run(todo, out, "p", "", "u", "f", "oops", "p", "")
    labeled = load_jsonl(out, LabeledCase)
    assert [(c.id, c.human_label) for c in labeled] == [("c0", "fail"), ("c1", "pass")]


def test_undo_with_nothing_and_bad_key(tmp_path: Path) -> None:
    todo, out = write_todo(tmp_path / "t.jsonl", 1), tmp_path / "o.jsonl"
    shown = run(todo, out, "u", "x", "p", "")
    assert "Nothing to undo." in shown
    assert "Please type p, f, s, u or q." in shown


def test_case_display_hides_nothing_but_verdicts(tmp_path: Path) -> None:
    todo = tmp_path / "t.jsonl"
    todo.write_text('{"id": "c0", "input": "q", "context": "docs", "output": "a"}\n')
    shown = run(todo, tmp_path / "o.jsonl", "q")
    assert "CONTEXT:\ndocs" in shown[2]
    assert "verdict" not in "".join(shown).lower()


def make_labeled(path: Path, fails: int, passes: int) -> Path:
    rows = [
        {"id": f"c{i:02d}", "input": "q", "output": "a", "human_label": label}
        for i, label in enumerate(["fail"] * fails + ["pass"] * passes)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_split_is_stratified_and_reproducible(tmp_path: Path) -> None:
    path = make_labeled(tmp_path / "tox.jsonl", fails=25, passes=25)
    dev_path, test_path = split_labeled(path, test_size=20, seed=7)
    assert (dev_path.name, test_path.name) == ("tox.dev.jsonl", "tox.test.jsonl")
    dev, test = load_jsonl(dev_path, LabeledCase), load_jsonl(test_path, LabeledCase)
    assert len(test) == 20 and len(dev) == 30
    assert sum(c.human_label == "fail" for c in test) == 10  # same 50/50 mix
    assert not {c.id for c in dev} & {c.id for c in test}
    first = test_path.read_text()
    split_labeled(path, test_size=20, seed=7)
    assert test_path.read_text() == first  # same seed, same split


def test_split_rejects_bad_size(tmp_path: Path) -> None:
    path = make_labeled(tmp_path / "x.jsonl", 2, 2)
    with pytest.raises(ValueError):
        split_labeled(path, test_size=4)


runner = CliRunner()


def test_cli_label_and_split(tmp_path: Path) -> None:
    todo = write_todo(tmp_path / "t.jsonl", 4)
    out = tmp_path / "tox.jsonl"
    result = runner.invoke(
        cli.app, ["label", "--todo", str(todo), "--out", str(out)], input="f\n\np\n\nf\n\np\n\n"
    )
    assert result.exit_code == 0, result.output
    assert "4 labeled (2 fail, 2 pass)" in result.output

    result = runner.invoke(cli.app, ["split", "--data", str(out), "--test-size", "2"])
    assert result.exit_code == 0, result.output
    assert "seed 42" in result.output


def test_cli_label_and_split_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    out = tmp_path / "o.jsonl"
    assert runner.invoke(cli.app, ["label", "--todo", str(bad), "--out", str(out)]).exit_code == 2
    labeled = make_labeled(tmp_path / "l.jsonl", 1, 1)
    result = runner.invoke(cli.app, ["split", "--data", str(labeled), "--test-size", "5"])
    assert result.exit_code == 2
