"""OpenTelemetry tracing.

Span tree for one run:
  eval.run                      (suite name, case count, overall result)
  └── eval.case                 (one per case, in parallel threads)
      ├── target.generate       (only when the target model produces the output)
      └── judge <name>          (verdict, latency, model)

Tracing is off unless `judgekit run --trace` calls `setup_tracing()`. Until
then the OpenTelemetry API hands out no-op spans, so the instrumentation in
runner.py and judges.py costs almost nothing.
"""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

from judgekit import __version__

# Instrumented modules import this. It is a proxy: spans go to whichever
# provider is installed globally at the time they are created.
tracer = trace.get_tracer("judgekit")


def build_tracer_provider(exporter: SpanExporter, batch: bool = True) -> TracerProvider:
    """A provider that tags every span with our service name and sends it to `exporter`.

    Batch export sends spans in the background in groups (production default);
    simple export sends each span immediately (used in tests).
    """
    resource = Resource.create({"service.name": "judgekit", "service.version": __version__})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    return provider


def setup_tracing() -> TracerProvider:
    """Install a global provider that exports spans over OTLP/HTTP.

    The endpoint comes from the standard OTEL_EXPORTER_OTLP_ENDPOINT variable
    and defaults to http://localhost:4318 (Jaeger from docker-compose.yml).
    Call `.shutdown()` on the result before exiting so buffered spans are sent.
    """
    provider = build_tracer_provider(OTLPSpanExporter())
    trace.set_tracer_provider(provider)
    return provider
