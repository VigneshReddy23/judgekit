"""Shared test setup.

Installs one global tracer provider that records spans in memory, so tracing
tests can inspect spans without Jaeger or a network. OpenTelemetry allows the
global provider to be set only once per process, hence doing it here.
"""

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from judgekit.tracing import build_tracer_provider

_EXPORTER = InMemorySpanExporter()
trace.set_tracer_provider(build_tracer_provider(_EXPORTER, batch=False))


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    _EXPORTER.clear()
    yield _EXPORTER
    _EXPORTER.clear()
