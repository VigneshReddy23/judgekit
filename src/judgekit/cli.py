"""Command-line interface.

Exit codes are the CI contract:
  0 = every scorer met its threshold
  1 = at least one scorer is below its threshold (quality regression)
  2 = the suite, cases or labeled file is invalid (configuration problem)
"""

from pathlib import Path
from typing import Annotated

import typer

from judgekit.calibrate import calibrate as run_calibration
from judgekit.calibrate import format_calibration, load_labeled
from judgekit.judges import LLMJudge, available_judges
from judgekit.providers import DEFAULT_JUDGE_MODEL_ID, BedrockProvider
from judgekit.runner import RunResult, load_cases, load_suite, run_suite
from judgekit.tracing import setup_tracing

app = typer.Typer(
    help="judgekit: evaluate LLM outputs with checks and calibrated judges.",
    no_args_is_help=True,
)

MAX_FAILURES_SHOWN = 10


@app.callback()
def main() -> None:
    """judgekit command-line interface."""


def format_summary(result: RunResult) -> str:
    lines = [f"Suite: {result.suite_name} ({len(result.cases)} cases)", ""]
    lines.append(f"{'scorer':<24}{'pass rate':<18}{'threshold':<12}status")
    for s in result.summaries:
        rate = f"{s.pass_rate:.1%} ({s.passed}/{s.total})"
        threshold = f"{s.threshold:.1%}" if s.threshold is not None else "-"
        status = "ok" if s.meets_threshold else "BELOW THRESHOLD"
        lines.append(f"{s.name:<24}{rate:<18}{threshold:<12}{status}")

    failures = [r for cr in result.cases for r in cr.results if not r.passed]
    if failures:
        lines += ["", f"Failures ({len(failures)}):"]
        for r in failures[:MAX_FAILURES_SHOWN]:
            lines.append(f"  [{r.judge}] {r.case_id}: {r.reason}")
        if len(failures) > MAX_FAILURES_SHOWN:
            lines.append(f"  ... and {len(failures) - MAX_FAILURES_SHOWN} more")

    below = sum(not s.meets_threshold for s in result.summaries)
    lines += ["", f"Total time: {result.total_latency_ms / 1000:.1f}s"]
    lines.append("RESULT: PASSED" if result.passed else f"RESULT: FAILED ({below} below threshold)")
    return "\n".join(lines)


@app.command()
def run(
    suite_path: Annotated[
        Path, typer.Argument(help="Path to a suite YAML file.", exists=True, dir_okay=False)
    ],
    trace: Annotated[
        bool,
        typer.Option(
            "--trace",
            help="Export OpenTelemetry spans via OTLP (OTEL_EXPORTER_OTLP_ENDPOINT, "
            "default http://localhost:4318).",
        ),
    ] = False,
) -> None:
    """Run an eval suite. Exits 1 if any scorer is below its threshold."""
    try:
        suite = load_suite(suite_path)
        cases = load_cases(suite.cases_file)
    except (ValueError, OSError) as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    judges: list[LLMJudge] = []
    if suite.judges:
        judge_provider = BedrockProvider(suite.judge_model_id, region=suite.region)
        judges = [LLMJudge(name, judge_provider) for name in suite.judges]

    target = None
    if suite.target is not None:
        target = BedrockProvider(
            suite.target.model_id,
            region=suite.region,
            temperature=suite.target.temperature,
            max_tokens=suite.target.max_tokens,
        )

    provider = setup_tracing() if trace else None
    try:
        result = run_suite(suite, cases, judges, target)
    finally:
        if provider is not None:
            provider.shutdown()  # flush buffered spans before the process exits
    typer.echo(format_summary(result))
    raise typer.Exit(code=0 if result.passed else 1)


@app.command()
def calibrate(
    judge: Annotated[str, typer.Option("--judge", help="Judge name, e.g. toxicity.")],
    data: Annotated[
        Path,
        typer.Option("--data", help="Labeled JSONL file.", exists=True, dir_okay=False),
    ],
    model_id: Annotated[
        str, typer.Option("--model-id", help="Bedrock judge model ID.")
    ] = DEFAULT_JUDGE_MODEL_ID,
    region: Annotated[str | None, typer.Option("--region", help="AWS region.")] = None,
    workers: Annotated[int, typer.Option("--workers", min=1, help="Parallel calls.")] = 4,
) -> None:
    """Compare a judge with your human labels (Cohen's kappa, precision, recall)."""
    if judge not in available_judges():
        typer.echo(
            f"config error: unknown judge {judge!r}; available: {available_judges()}", err=True
        )
        raise typer.Exit(code=2)
    try:
        cases = load_labeled(data)
    except (ValueError, OSError) as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    llm_judge = LLMJudge(judge, BedrockProvider(model_id, region=region))
    result = run_calibration(llm_judge, cases, max_workers=workers)
    typer.echo(format_calibration(result))
