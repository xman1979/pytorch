# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Fully automatic training metrics collector with PyTorch instrumentation.

This module automatically collects training metrics without any application
code changes. It hooks into PyTorch's core components to capture:
- Forward pass timing (via nn.Module hooks)
- Backward pass timing (via autograd.backward instrumentation)
- Optimizer step timing (via Optimizer hooks)
- Loss values (via hooks on loss modules)
- Data loading time (via DataLoader hooks)
- Epoch and step tracking (via pattern detection)

Metrics are exported to ODS3 via OpenTelemetry using the fair_model schema.

Auto-activation:
    Set OTEL_EXPORTER_OTLP_ENDPOINT environment variable to enable.
    Set PYTORCH_TRAINING_METRICS_ENABLED=1 to force enable even without endpoint.
"""

import os
import threading
import time
import weakref
from typing import Any, Callable, Optional, Tuple, Set

import torch
from torch.utils.hooks import RemovableHandle

from .config import MetricsConfig, get_config

# Try to import OpenTelemetry, gracefully degrade if not available
try:
    from opentelemetry import metrics
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    metrics = None
    MeterProvider = None


class AutoTrainingMetricsCollector:
    """
    Fully automatic training metrics collector.

    This collector hooks into PyTorch's internals to automatically capture
    training metrics without requiring any application code changes.

    Key features:
    - Auto-detects training batches by monitoring forward -> backward -> step pattern
    - Auto-captures loss from loss modules (CrossEntropyLoss, MSELoss, etc.)
    - Auto-tracks epochs by monitoring DataLoader iteration patterns
    - Thread-safe with minimal overhead when disabled
    """

    # Loss module classes to hook into
    LOSS_MODULES = (
        # Standard PyTorch loss modules
        'CrossEntropyLoss', 'NLLLoss', 'MSELoss', 'L1Loss', 'SmoothL1Loss',
        'BCELoss', 'BCEWithLogitsLoss', 'CTCLoss', 'KLDivLoss', 'HuberLoss',
        'MultiMarginLoss', 'MultiLabelMarginLoss', 'MultiLabelSoftMarginLoss',
        'SoftMarginLoss', 'TripletMarginLoss', 'TripletMarginWithDistanceLoss',
        'CosineEmbeddingLoss', 'HingeEmbeddingLoss', 'MarginRankingLoss',
        'PoissonNLLLoss', 'GaussianNLLLoss',
        # Custom/fused loss modules commonly used in training
        'FusedCrossEntropyLoss', 'ChunkedCrossEntropyLoss',
        'LabelSmoothingCrossEntropyLoss', 'FocalLoss',
    )

    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or get_config()
        self._lock = threading.RLock()

        # State tracking
        self._enabled = False
        self._epoch = 0  # 0-indexed to match Python convention (for epoch in range(n))
        self._training_step = 0
        self._batch_count = 0
        self._last_loss: Optional[float] = None

        # Training pattern detection
        self._in_training_batch = False
        self._had_forward = False
        self._had_backward = False
        self._had_optimizer_step = False

        # Timing state
        self._batch_start_time: Optional[float] = None
        self._data_loading_start: Optional[float] = None
        self._forward_start: Optional[float] = None
        self._backward_start: Optional[float] = None
        self._optimizer_start: Optional[float] = None

        # Accumulated times for current batch
        self._data_loading_time_ms: float = 0.0
        self._forward_pass_time_ms: float = 0.0
        self._backward_pass_time_ms: float = 0.0
        self._optimizer_step_time_ms: float = 0.0

        # Hook handles for cleanup
        self._hook_handles: list[RemovableHandle] = []

        # Track forward depth to only time the outermost call
        self._forward_depth = 0

        # Track hooked loss modules to avoid double-hooking
        self._hooked_loss_modules: Set[int] = set()

        # DataLoader tracking for epoch detection
        self._dataloader_iteration_count = 0
        self._last_dataloader_len: Optional[int] = None

        # OpenTelemetry components
        self._meter_provider: Optional[Any] = None
        self._meter: Optional[Any] = None
        self._gauges: dict = {}

        # Prefetch queue size tracking (for observable gauge)
        self._prefetch_queue_size: int = 0
        self._active_dataloader_iterators: weakref.WeakSet = weakref.WeakSet()

        # Data preprocessing (collate_fn) time tracking
        self._data_preprocessing_time_ms: float = 0.0

    def _initialize_otel(self) -> None:
        """Initialize OpenTelemetry with fair_model schema for ODS3."""
        if not OTEL_AVAILABLE:
            return

        if not self.config.otel_endpoint:
            return

        resource_attributes = {
            "fb.metric.schema": self.config.schema_name,
            "fb.metric.case_insensitive": True,
            "job_id": self.config.job_id,
            "node": self.config.node,
            "cluster": self.config.cluster,
            "gpu": self.config.gpu,
            "global_rank": self.config.global_rank,
            "local_rank": self.config.local_rank,
            "username": self.config.username,
            "task_id": self.config.task_id,
        }
        resource = Resource(attributes=resource_attributes)
        exporter = OTLPMetricExporter(
            endpoint=f"{self.config.otel_endpoint}/v1/metrics",
            timeout=self.config.export_timeout_ms / 1000,
        )
        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=self.config.export_interval_ms,
            export_timeout_millis=self.config.export_timeout_ms,
        )
        self._meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(self._meter_provider)
        self._meter = metrics.get_meter_provider().get_meter("pytorch_training")

        # Create gauges for training metrics
        self._gauges["epoch"] = self._meter.create_gauge(
            name="epoch",
            description="Current training epoch",
            unit="1"
        )
        self._gauges["train_batch_time"] = self._meter.create_gauge(
            name="train_batch_time",
            description="Total time for one training batch",
            unit="ms"
        )
        self._gauges["train_data_loading_time"] = self._meter.create_gauge(
            name="train_data_loading_time",
            description="Time spent loading data for a batch",
            unit="ms"
        )
        self._gauges["train_forward_pass_time"] = self._meter.create_gauge(
            name="train_forward_pass_time",
            description="Time for forward pass",
            unit="ms"
        )
        self._gauges["train_backward_pass_time"] = self._meter.create_gauge(
            name="train_backward_pass_time",
            description="Time for backward pass",
            unit="ms"
        )
        self._gauges["train_loss"] = self._meter.create_gauge(
            name="train_loss",
            description="Training loss value",
            unit="1"
        )
        self._gauges["training_step"] = self._meter.create_gauge(
            name="training_step",
            description="Current training step/batch index",
            unit="1"
        )

        # Data preprocessing (collate_fn) time gauge
        self._gauges["data_preprocessing_time"] = self._meter.create_gauge(
            name="data_preprocessing_time",
            description="Time spent in collate_fn preprocessing data",
            unit="ms"
        )

        # Async observable gauge for prefetch queue size (ready batches in queue)
        self._gauges["data_prefetch_queue_size"] = self._meter.create_observable_gauge(
            name="data_prefetch_queue_size",
            description="Gauge the data loader prefetch queue size (ready batches)",
            callbacks=[self._prefetch_queue_size_callback],
            unit="Batches"
        )

        # Async observable gauge for tasks outstanding (total in-flight work)
        self._gauges["data_read_tasks_outstanding"] = self._meter.create_observable_gauge(
            name="data_read_tasks_outstanding",
            description="Total tasks dispatched to workers but not yet consumed",
            callbacks=[self._tasks_outstanding_callback],
            unit="Batches"
        )

    def _forward_pre_hook(
        self, module: torch.nn.Module, args: Tuple[Any, ...]
    ) -> None:
        """Global forward pre-hook to start timing forward pass."""
        if not self._enabled:
            return

        # Skip loss modules - they should not reset forward timing
        if self._is_loss_module(module):
            return

        with self._lock:
            self._forward_depth += 1
            if self._forward_depth == 1:
                # Start of a new training batch if we've completed a previous one
                if self._had_optimizer_step:
                    self._complete_batch()

                # Start new batch timing
                if not self._in_training_batch:
                    self._start_batch()

                self._forward_start = time.perf_counter()
                self._had_forward = True

                # End data loading timing when forward starts
                if self._data_loading_start is not None:
                    self._data_loading_time_ms = (
                        time.perf_counter() - self._data_loading_start
                    ) * 1000
                    self._data_loading_start = None

    def _forward_hook(
        self,
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        output: Any,
    ) -> None:
        """Global forward hook to end timing forward pass."""
        if not self._enabled:
            return

        # Skip loss modules - they should not affect forward timing
        if self._is_loss_module(module):
            return

        with self._lock:
            self._forward_depth -= 1
            if self._forward_depth == 0:
                if self._forward_start is not None:
                    # Accumulate forward time (for gradient accumulation scenarios)
                    self._forward_pass_time_ms += (
                        time.perf_counter() - self._forward_start
                    ) * 1000
                    self._forward_start = None

    def _loss_forward_hook(
        self,
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        output: Any,
    ) -> None:
        """Hook on loss modules to capture loss values."""
        if not self._enabled:
            return

        with self._lock:
            # Capture scalar loss value
            if isinstance(output, torch.Tensor):
                try:
                    if output.numel() == 1:
                        self._last_loss = output.detach().item()
                except Exception:
                    pass

    def _is_loss_module(self, module: torch.nn.Module) -> bool:
        """Check if a module is a loss module."""
        class_name = module.__class__.__name__
        # Check explicit list first, then check if name ends with "Loss"
        return class_name in self.LOSS_MODULES or class_name.endswith('Loss')

    def _global_forward_hook_for_loss(
        self,
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        output: Any,
    ) -> None:
        """
        Global forward hook that captures loss from ANY loss module.

        This is a fallback to catch loss modules that weren't hooked during creation.
        """
        if not self._enabled:
            return

        # Only process loss modules
        if not self._is_loss_module(module):
            return

        with self._lock:
            # Capture scalar loss value
            if isinstance(output, torch.Tensor):
                try:
                    if output.numel() == 1:
                        self._last_loss = output.detach().item()
                except Exception:
                    pass

    def _optimizer_pre_hook(
        self, optimizer: torch.optim.Optimizer, args: Tuple[Any, ...], kwargs: dict
    ) -> Optional[Tuple[Tuple[Any, ...], dict]]:
        """Global optimizer pre-hook to start timing optimizer step."""
        if not self._enabled:
            return None

        with self._lock:
            self._optimizer_start = time.perf_counter()
        return None

    def _optimizer_post_hook(
        self, optimizer: torch.optim.Optimizer, args: Tuple[Any, ...], kwargs: dict
    ) -> None:
        """Global optimizer post-hook to end timing and complete batch."""
        if not self._enabled:
            return

        with self._lock:
            if self._optimizer_start is not None:
                self._optimizer_step_time_ms = (
                    time.perf_counter() - self._optimizer_start
                ) * 1000
                self._optimizer_start = None

            self._had_optimizer_step = True

    def _start_batch(self) -> None:
        """Start timing a new training batch."""
        self._batch_start_time = time.perf_counter()
        self._in_training_batch = True
        self._data_loading_time_ms = 0.0
        self._forward_pass_time_ms = 0.0
        self._backward_pass_time_ms = 0.0
        self._optimizer_step_time_ms = 0.0
        self._had_forward = False
        self._had_backward = False
        self._had_optimizer_step = False
        self._last_loss = None

        # Reset timing start markers to ensure clean state
        self._forward_start = None
        self._backward_start = None
        self._optimizer_start = None

        # Start data loading timer (will be stopped when forward starts)
        self._data_loading_start = time.perf_counter()

    def _complete_batch(self) -> None:
        """Complete the current training batch and record metrics."""
        if not self._in_training_batch:
            return

        self._batch_count += 1
        self._training_step += 1

        # Skip if not at logging interval
        if self._batch_count % self.config.log_every_n_batches != 0:
            self._reset_batch_state()
            return

        # Compute total batch time
        total_batch_time = 0.0
        if self._batch_start_time is not None:
            total_batch_time = (time.perf_counter() - self._batch_start_time) * 1000

        # Record metrics if OTel is available
        if self._meter is not None:
            self._gauges["epoch"].set(self._epoch)
            self._gauges["training_step"].set(self._training_step)
            self._gauges["train_batch_time"].set(total_batch_time)
            self._gauges["train_data_loading_time"].set(self._data_loading_time_ms)
            self._gauges["train_forward_pass_time"].set(self._forward_pass_time_ms)
            self._gauges["train_backward_pass_time"].set(self._backward_pass_time_ms)

            if self._last_loss is not None:
                self._gauges["train_loss"].set(self._last_loss)

        self._reset_batch_state()

    def _reset_batch_state(self) -> None:
        """Reset batch state for next iteration."""
        self._in_training_batch = False
        self._had_forward = False
        self._had_backward = False
        self._had_optimizer_step = False
        self._batch_start_time = None

    def start_backward(self) -> None:
        """Called by autograd.backward() to start timing backward pass."""
        if not self._enabled:
            return

        with self._lock:
            self._backward_start = time.perf_counter()
            self._had_backward = True

    def end_backward(self) -> None:
        """Called by autograd.backward() to end timing backward pass."""
        if not self._enabled:
            return

        with self._lock:
            if self._backward_start is not None:
                # Accumulate backward time (for gradient accumulation scenarios)
                self._backward_pass_time_ms += (
                    time.perf_counter() - self._backward_start
                ) * 1000
                self._backward_start = None

    def record_data_loading_time(self, time_seconds: float) -> None:
        """
        Record data loading time for the current batch.

        Called automatically by DataLoader.__next__() when a batch is loaded.

        Args:
            time_seconds: Time in seconds spent loading the data batch.
        """
        if not self._enabled:
            return

        with self._lock:
            self._data_loading_time_ms = time_seconds * 1000

    def on_epoch_end(self) -> None:
        """
        Called when a DataLoader epoch ends (StopIteration raised).

        This is called automatically by DataLoader when iteration is exhausted.
        Increments the epoch counter for the next epoch.
        """
        if not self._enabled:
            return

        with self._lock:
            # Complete any pending batch before epoch ends
            if self._in_training_batch:
                self._complete_batch()

            self._epoch += 1

    def increment_epoch(self) -> None:
        """Increment the epoch counter. Called when DataLoader restarts."""
        with self._lock:
            self._epoch += 1

    def set_epoch(self, epoch: int) -> None:
        """Manually set the epoch number."""
        with self._lock:
            self._epoch = epoch

    def _prefetch_queue_size_callback(self, options: Any) -> list:
        """
        Callback for the observable gauge to report prefetch queue size.

        This is called periodically by the OpenTelemetry SDK to get the current
        prefetch queue size from active DataLoader iterators.

        Reports _data_queue.qsize() which represents the number of batches
        that are ready to be consumed (already processed by workers).

        Note: May return 0 on platforms where qsize() is not implemented (e.g., macOS).
        """
        if not self._enabled:
            return []

        try:
            from opentelemetry.metrics import Observation

            with self._lock:
                total_queue_size = 0
                # Sum up _data_queue.qsize() from all active multi-process iterators
                for iterator in list(self._active_dataloader_iterators):
                    try:
                        if hasattr(iterator, '_data_queue') and iterator._data_queue is not None:
                            total_queue_size += iterator._data_queue.qsize()
                    except (ReferenceError, AttributeError, NotImplementedError):
                        # qsize() may raise NotImplementedError on some platforms (e.g., macOS)
                        pass

                return [Observation(total_queue_size)]
        except Exception:
            return []

    def _tasks_outstanding_callback(self, options: Any) -> list:
        """
        Callback for the observable gauge to report tasks outstanding.

        This is called periodically by the OpenTelemetry SDK to get the current
        number of tasks dispatched to workers but not yet consumed.

        Reports _tasks_outstanding which represents the total number of batches
        "in flight" - including those being processed by workers AND those
        waiting in the result queue.

        Use this with data_prefetch_queue_size to diagnose stuck workers:
        - High outstanding + High queue size = healthy prefetch
        - High outstanding + Low queue size = workers stuck/slow
        """
        if not self._enabled:
            return []

        try:
            from opentelemetry.metrics import Observation

            with self._lock:
                total_outstanding = 0
                # Sum up _tasks_outstanding from all active multi-process iterators
                for iterator in list(self._active_dataloader_iterators):
                    try:
                        if hasattr(iterator, '_tasks_outstanding'):
                            total_outstanding += iterator._tasks_outstanding
                    except (ReferenceError, AttributeError):
                        pass

                return [Observation(total_outstanding)]
        except Exception:
            return []

    def register_dataloader_iterator(self, iterator: Any) -> None:
        """
        Register a DataLoader iterator to track its prefetch queue size.

        Called automatically by _MultiProcessingDataLoaderIter when created.
        """
        if not self._enabled:
            return

        with self._lock:
            self._active_dataloader_iterators.add(iterator)

    def unregister_dataloader_iterator(self, iterator: Any) -> None:
        """
        Unregister a DataLoader iterator when it's destroyed.
        """
        with self._lock:
            self._active_dataloader_iterators.discard(iterator)

    def record_data_preprocessing_time(self, time_seconds: float) -> None:
        """
        Record the time spent in collate_fn preprocessing data.

        Called automatically by DataLoader when collate_fn completes.

        Args:
            time_seconds: Time in seconds spent in collate_fn.
        """
        if not self._enabled:
            return

        with self._lock:
            self._data_preprocessing_time_ms = time_seconds * 1000

            # Record the metric if OTel is available
            if self._meter is not None and "data_preprocessing_time" in self._gauges:
                self._gauges["data_preprocessing_time"].set(self._data_preprocessing_time_ms)

    def _hook_loss_module(self, module: torch.nn.Module) -> None:
        """Hook a loss module to capture its output."""
        module_id = id(module)
        if module_id in self._hooked_loss_modules:
            return

        self._hooked_loss_modules.add(module_id)
        handle = module.register_forward_hook(self._loss_forward_hook)
        self._hook_handles.append(handle)

    def _module_registration_hook(
        self,
        module: torch.nn.Module,
        name: str,
        submodule: torch.nn.Module,
    ) -> None:
        """Hook called when a submodule is registered. Used to hook loss modules."""
        class_name = submodule.__class__.__name__
        if class_name in self.LOSS_MODULES:
            self._hook_loss_module(submodule)

    def _register_hooks(self) -> None:
        """Register global hooks for automatic instrumentation."""
        # Forward hooks on nn.Module
        if self.config.track_forward_time:
            handle = torch.nn.modules.module.register_module_forward_pre_hook(
                self._forward_pre_hook
            )
            self._hook_handles.append(handle)

            handle = torch.nn.modules.module.register_module_forward_hook(
                self._forward_hook
            )
            self._hook_handles.append(handle)

        # Global forward hook for loss capture - catches ALL loss modules
        # regardless of when they were created
        handle = torch.nn.modules.module.register_module_forward_hook(
            self._global_forward_hook_for_loss
        )
        self._hook_handles.append(handle)

        # Optimizer hooks
        if self.config.track_optimizer_time:
            # Lazy import to avoid circular import during torch initialization
            import torch.optim.optimizer as optim_module
            handle = optim_module.register_optimizer_step_pre_hook(
                self._optimizer_pre_hook
            )
            self._hook_handles.append(handle)

            handle = optim_module.register_optimizer_step_post_hook(
                self._optimizer_post_hook
            )
            self._hook_handles.append(handle)

        # Hook for new module registration to catch loss modules
        handle = torch.nn.modules.module.register_module_module_registration_hook(
            self._module_registration_hook
        )
        self._hook_handles.append(handle)

        # Hook existing loss modules in commonly used loss functions
        for name in self.LOSS_MODULES:
            try:
                loss_class = getattr(torch.nn, name, None)
                if loss_class is not None:
                    # Monkey-patch __init__ to hook instances on creation
                    original_init = loss_class.__init__
                    collector = self

                    def make_hooked_init(orig_init):
                        def hooked_init(self, *args, **kwargs):
                            orig_init(self, *args, **kwargs)
                            collector._hook_loss_module(self)
                        return hooked_init

                    loss_class.__init__ = make_hooked_init(original_init)
            except Exception:
                pass

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles.clear()
        self._hooked_loss_modules.clear()

    def enable(self) -> None:
        """Enable metrics collection and register hooks."""
        if self._enabled:
            return

        self._initialize_otel()
        self._register_hooks()
        self._enabled = True

    def disable(self) -> None:
        """Disable metrics collection and remove hooks."""
        if not self._enabled:
            return

        self._remove_hooks()
        self._enabled = False

    def shutdown(self) -> None:
        """Shutdown the metrics collector and flush pending metrics."""
        self.disable()
        if self._meter_provider is not None:
            self._meter_provider.shutdown()
            self._meter_provider = None

    @property
    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._enabled


# Global collector instance
_collector: Optional[AutoTrainingMetricsCollector] = None
_collector_lock = threading.Lock()


def get_collector() -> AutoTrainingMetricsCollector:
    """Get the global training metrics collector instance."""
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = AutoTrainingMetricsCollector()
        return _collector


def enable_training_metrics(config: Optional[MetricsConfig] = None) -> AutoTrainingMetricsCollector:
    """
    Enable automatic training metrics collection.

    This is called automatically when PyTorch starts if OTEL_EXPORTER_OTLP_ENDPOINT
    is set. You can also call it manually to enable metrics.
    """
    global _collector
    with _collector_lock:
        if _collector is not None:
            _collector.disable()

        _collector = AutoTrainingMetricsCollector(config)
        _collector.enable()
        return _collector


def disable_training_metrics() -> None:
    """Disable automatic training metrics collection and remove hooks."""
    global _collector
    with _collector_lock:
        if _collector is not None:
            _collector.shutdown()
            _collector = None


def is_enabled() -> bool:
    """Check if training metrics collection is currently enabled."""
    global _collector
    with _collector_lock:
        return _collector is not None and _collector._enabled


def _auto_initialize() -> None:
    """
    Auto-initialize training metrics if environment is configured.

    This is called automatically when PyTorch is imported.
    """
    # Check if auto-initialization should happen
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    force_enabled = os.getenv("PYTORCH_TRAINING_METRICS_ENABLED", "0") == "1"

    if otel_endpoint or force_enabled:
        try:
            enable_training_metrics()
        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to auto-initialize training metrics: {e}",
                UserWarning,
            )


# For backward compatibility
TrainingMetricsCollector = AutoTrainingMetricsCollector
