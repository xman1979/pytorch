# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Configuration for training metrics collection."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricsConfig:
    """Configuration for training metrics collection.

    Attributes:
        enabled: Whether metrics collection is enabled.
        otel_endpoint: OpenTelemetry exporter endpoint URL.
        schema_name: ODS3 schema name (default: "fair_model").
        export_interval_ms: How often to export metrics in milliseconds.
        log_every_n_batches: Only log metrics every N batches to reduce overhead.
        track_forward_time: Whether to track forward pass time.
        track_backward_time: Whether to track backward pass time.
        track_optimizer_time: Whether to track optimizer step time.
        track_data_loading_time: Whether to track data loading time.
        job_id: Training job ID (defaults to SLURM_JOB_ID).
        node: Node name (defaults to SLURMD_NODENAME).
        cluster: Cluster name (defaults to SCENV).
        global_rank: Global rank in distributed training.
        local_rank: Local rank on this node.
        gpu: GPU index.
        username: Username running the training.
        task_id: SLURM array task ID.
    """

    enabled: bool = True
    otel_endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    )
    schema_name: str = "fair_model"
    export_interval_ms: int = 5000
    export_timeout_ms: int = 5000
    log_every_n_batches: int = 1

    # Feature flags
    track_forward_time: bool = True
    track_backward_time: bool = True
    track_optimizer_time: bool = True
    track_data_loading_time: bool = True

    # Job metadata (dimensions in ODS3)
    job_id: str = field(
        default_factory=lambda: os.getenv("SLURM_JOB_ID", "0")
    )
    node: str = field(
        default_factory=lambda: os.getenv("SLURMD_NODENAME", "localhost")
    )
    cluster: str = field(
        default_factory=lambda: os.getenv("SCENV", "fair_cluster")
    )
    global_rank: int = field(
        default_factory=lambda: int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
    )
    local_rank: int = field(
        default_factory=lambda: int(os.getenv("SLURM_LOCALID", os.getenv("LOCAL_RANK", "0")))
    )
    gpu: int = field(
        default_factory=lambda: int(os.getenv("LOCAL_RANK", "0"))
    )
    username: str = field(
        default_factory=lambda: os.getenv("USER", "unknown")
    )
    task_id: int = field(
        default_factory=lambda: int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    )

    @classmethod
    def from_environment(cls) -> "MetricsConfig":
        """Create a MetricsConfig with all defaults from environment variables."""
        return cls()

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.enabled and not self.otel_endpoint:
            import warnings
            warnings.warn(
                "OTEL_EXPORTER_OTLP_ENDPOINT not set. Metrics will not be exported. "
                "Set this environment variable to enable metric export.",
                UserWarning,
            )

        if self.log_every_n_batches < 1:
            raise ValueError("log_every_n_batches must be >= 1")

        if self.export_interval_ms < 100:
            raise ValueError("export_interval_ms must be >= 100")


# Global configuration instance
_config: Optional[MetricsConfig] = None


def get_config() -> MetricsConfig:
    """Get the current metrics configuration."""
    global _config
    if _config is None:
        _config = MetricsConfig.from_environment()
    return _config


def set_config(config: MetricsConfig) -> None:
    """Set the metrics configuration."""
    global _config
    config.validate()
    _config = config
