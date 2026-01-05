# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
PyTorch Training Metrics Module.

This module provides AUTOMATIC instrumentation of PyTorch training loops to record
metrics to ODS3 via OpenTelemetry. NO APPLICATION CHANGES REQUIRED.

When OTEL_EXPORTER_OTLP_ENDPOINT is set, metrics are automatically collected for:
- epoch: Current training epoch
- train_batch_time: Total time for one training batch (ms)
- train_data_loading_time: Time spent loading data (ms)
- train_forward_pass_time: Time for forward pass (ms)
- train_backward_pass_time: Time for backward pass (ms)
- train_loss: Training loss value (auto-captured from loss modules)
- training_step: Current training step/batch index

Auto-activation:
    Set OTEL_EXPORTER_OTLP_ENDPOINT environment variable to enable automatic metrics.
    Set PYTORCH_TRAINING_METRICS_ENABLED=1 to force enable even without endpoint.

The instrumentation hooks into PyTorch's core components:
- nn.Module: Forward pass timing via global hooks
- autograd.backward: Backward pass timing
- Optimizer.step: Optimizer step timing via global hooks
- Loss modules: Automatic loss value capture

Example (NO CODE CHANGES NEEDED):
    # Just set the environment variable and run your training script:
    # export OTEL_EXPORTER_OTLP_ENDPOINT="http://<otel-gateway-endpoint>"
    # python train.py

    # Your existing training code works unchanged:
    for epoch in range(num_epochs):
        for data, target in train_loader:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # Metrics are automatically collected and exported!

Manual control (optional):
    from torch.training_metrics import enable_training_metrics, disable_training_metrics

    enable_training_metrics()  # Explicitly enable
    # ... training code ...
    disable_training_metrics()  # Explicitly disable
"""

from .collector import (
    AutoTrainingMetricsCollector,
    TrainingMetricsCollector,
    get_collector,
    enable_training_metrics,
    disable_training_metrics,
    is_enabled,
    _auto_initialize,
)
from .context import TrainingContext
from .config import MetricsConfig

__all__ = [
    "AutoTrainingMetricsCollector",
    "TrainingMetricsCollector",
    "TrainingContext",
    "MetricsConfig",
    "enable_training_metrics",
    "disable_training_metrics",
    "is_enabled",
    "get_collector",
]

# Auto-initialize if OTEL endpoint is configured
_auto_initialize()
