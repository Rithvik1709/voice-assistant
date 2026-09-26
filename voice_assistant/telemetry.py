import logging
import os

from opentelemetry import trace


class TelemetryLogFilter(logging.Filter):
    """Injects OTel trace_id and span_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            record.trace_id = trace.format_trace_id(ctx.trace_id)
            record.span_id = trace.format_span_id(ctx.span_id)
        else:
            record.trace_id = trace.format_trace_id(0)
            record.span_id = trace.format_span_id(0)
        return True


def tracing_requested() -> bool:
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")) or os.environ.get("VAANI_TRACE_CONSOLE") == "1"


def init_telemetry(service_name: str = "voice-assistant") -> bool:
    """Initialize OpenTelemetry tracing when it has somewhere to send spans.

    Tracing is opt-in: set OTEL_EXPORTER_OTLP_ENDPOINT to export to a
    collector, or VAANI_TRACE_CONSOLE=1 to print spans for debugging.
    Otherwise the default no-op tracer is kept, which costs nothing.
    Returns True when tracing was enabled.
    """
    if not tracing_requested():
        return False

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except ImportError:
        logging.warning("opentelemetry-sdk is not installed; tracing disabled. Install `voice-assistant[otel]`.")
        return False

    # Check if a TracerProvider is already registered
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return True

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        except ImportError:
            logging.warning("opentelemetry-exporter-otlp is not installed; tracing disabled.")
            return False
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
    else:
        processor = SimpleSpanProcessor(ConsoleSpanExporter())

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # Instrument logging to include trace_id and span_id
    telemetry_filter = TelemetryLogFilter()

    # We get the root logger and inject our filter
    # and update the formatter to include trace_id and span_id
    root_logger = logging.getLogger()

    # Only update handlers that don't already have the filter
    for handler in root_logger.handlers:
        if telemetry_filter not in handler.filters:
            handler.addFilter(telemetry_filter)

            # If there's a formatter, augment it, or create a new one
            if handler.formatter:
                formatter = handler.formatter
                # Accessing _fmt is discouraged but necessary for augmentation.
                # We use getattr to be safer and copy other attributes to preserve config.
                fmt = getattr(formatter, "_fmt", None)
                if fmt and "trace_id" not in fmt:
                    # Determine style (%, {, or $) to preserve it and use correct placeholders
                    style = "%"
                    if hasattr(formatter, "_style"):
                        from logging import PercentStyle, StrFormatStyle, StringTemplateStyle
                        if isinstance(formatter._style, PercentStyle):
                            style = "%"
                        elif isinstance(formatter._style, StrFormatStyle):
                            style = "{"
                        elif isinstance(formatter._style, StringTemplateStyle):
                            style = "$"

                    if style == "{":
                        new_fmt = fmt.replace("{message}", "[trace_id={trace_id} span_id={span_id}] {message}")
                    elif style == "$":
                        new_fmt = fmt.replace("${message}", "[trace_id=$trace_id span_id=$span_id] ${message}").replace("$message", "[trace_id=$trace_id span_id=$span_id] $message")
                    else:
                        new_fmt = fmt.replace("%(message)s", "[trace_id=%(trace_id)s span_id=%(span_id)s] %(message)s")

                    handler.setFormatter(logging.Formatter(
                        fmt=new_fmt,
                        datefmt=formatter.datefmt,
                        style=style
                    ))
            else:
                handler.setFormatter(logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s span_id=%(span_id)s] %(message)s"
                ))

    # Instrument gRPC
    try:
        from opentelemetry.instrumentation.grpc import (
            GrpcAioInstrumentorClient,
            GrpcAioInstrumentorServer,
        )

        GrpcAioInstrumentorClient().instrument()
        GrpcAioInstrumentorServer().instrument()
    except ImportError:
        logging.warning("opentelemetry-instrumentation-grpc not installed, gRPC won't be traced.")

    return True
