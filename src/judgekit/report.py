"""Render a RunResult as a single self-contained HTML file (Jinja2).

The file has no JavaScript and no external assets, so it can be opened from a
CI artifact download, emailed, or archived, and it renders the same offline.
"""

from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import BaseModel

from judgekit import __version__
from judgekit.providers import BedrockProvider
from judgekit.runner import ModelPricing, RunResult, SuiteConfig


class ModelUsage(BaseModel):
    role: str  # "judge" or "target"
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when the suite has no pricing for this model


def collect_usage(
    providers: list[tuple[str, BedrockProvider]], pricing: dict[str, ModelPricing]
) -> list[ModelUsage]:
    """Turn each provider's token counters into a usage row with an estimated cost."""
    usages = []
    for role, provider in providers:
        price = pricing.get(provider.model_id)
        cost = None
        if price is not None:
            cost = (
                provider.input_tokens * price.input_per_million_usd
                + provider.output_tokens * price.output_per_million_usd
            ) / 1_000_000
        usages.append(
            ModelUsage(
                role=role,
                model_id=provider.model_id,
                input_tokens=provider.input_tokens,
                output_tokens=provider.output_tokens,
                cost_usd=cost,
            )
        )
    return usages


def total_cost(usages: list[ModelUsage]) -> float | None:
    """Sum of costs; None if any model is unpriced (a partial sum would mislead)."""
    if any(u.cost_usd is None for u in usages):
        return None
    return sum(u.cost_usd or 0.0 for u in usages)


# autoescape=True: model outputs are untrusted text. Without escaping, an output
# containing <script> would run in the browser of whoever opens the report.
# (select_autoescape wouldn't match our ".j2" extension, so we set it explicitly.)
# StrictUndefined: a typo in the template raises instead of rendering blank.
_ENV = Environment(
    loader=PackageLoader("judgekit", "templates"),
    autoescape=True,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_report(
    result: RunResult, suite: SuiteConfig, usages: list[ModelUsage] | None = None
) -> str:
    usages = usages or []
    failed_cases = [cr for cr in result.cases if any(not r.passed for r in cr.results)]
    return _ENV.get_template("report.html.j2").render(
        result=result,
        judge_names=set(suite.judges),
        failed_cases=failed_cases,
        usages=usages,
        total_cost=total_cost(usages),
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        version=__version__,
    )


def write_report(
    path: Path, result: RunResult, suite: SuiteConfig, usages: list[ModelUsage] | None = None
) -> None:
    path.write_text(render_report(result, suite, usages), encoding="utf-8")
