import os
from dataclasses import dataclass
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import NoOpMeterProvider, Observation, CallbackOptions
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
OTEL_GATEWAY_ENDPOINT = OTEL_EXPORTER_OTLP_ENDPOINT  # for backwards compatibility

@dataclass
class MetricsMetadata:
    job_id: str
    node: str
    cluster: str
    gpu: int
    global_rank: int
    local_rank: int
    username: str
    task_id: int

OTEL_METRICS = {
    "train_batch_time_s": 0,
    "train_data_loading_time_ms": 0,
    "train_forward_pass_time_ms": 0,
    "train_backward_pass_time_ms": 0,
    "train_loss": 0,
}


def train_batch_time_gauge_callback(_: CallbackOptions):
    yield Observation(OTEL_METRICS["train_batch_time_s"], {})


def train_data_loading_time_gauge_callback(_: CallbackOptions):
    yield Observation(OTEL_METRICS["train_data_loading_time_ms"], {})


def train_forward_pass_time_gauge_callback(_: CallbackOptions):
    yield Observation(OTEL_METRICS["train_forward_pass_time_ms"], {})


def train_backward_pass_time_gauge_callback(_: CallbackOptions):
    yield Observation(OTEL_METRICS["train_backward_pass_time_ms"], {})


def train_loss_gauge_callback(_: CallbackOptions):
    yield Observation(OTEL_METRICS["train_loss"], {})


def initialize_otel(metrics_metadata: MetricsMetadata, log_otel: bool):
    if not log_otel:
        metrics.set_meter_provider(NoOpMeterProvider())
        return

    # you can send metrics to ODS3 ("fb.metric.schema") and/or Scuba ("fb.scuba.table")
    # see https://www.internalfb.com/code/fbsource/fbcode/monitoring/otel_gateway/README.md for more details
    # on available params
    resource_attributes = {
        # if you want to ship custom metrics that are not covered on
        # fair_model you'll need to create a new ODS3 schema
        "fb.metric.schema": "fair_model",
        "fb.metric.case_insensitive": True,
        # scuba table where metrics will be expanded to
        "fb.scuba.table": "otelpt_test",
        # attributes that should be expanded as its own scuba column
        "fb.scuba.columns": [
            "gpu",
            "job_id",
            "node",
            "cluster",
            "global_rank",
            "local_rank",
            "username",
            "task_id"
        ],
        "fb.scuba.include_attributes": False,
        "job_id": metrics_metadata.job_id,
        "node": metrics_metadata.node,
        "cluster": metrics_metadata.cluster,
        "gpu": metrics_metadata.gpu,
        "global_rank": metrics_metadata.global_rank,
        "local_rank": metrics_metadata.local_rank,
        "username": metrics_metadata.username,
        "task_id": metrics_metadata.task_id,
    }

    resource = Resource(attributes=resource_attributes)
    exporter = OTLPMetricExporter(
        endpoint=OTEL_GATEWAY_ENDPOINT + "/v1/metrics",
        timeout=5,
    )
    reader = PeriodicExportingMetricReader(
        exporter, export_interval_millis=60000, export_timeout_millis=5000
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader], shutdown_on_exit=True)
    metrics.set_meter_provider(meter_provider)
