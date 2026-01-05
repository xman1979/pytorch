# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Training context manager for simplified metrics collection."""

from typing import Optional

from .collector import get_collector, TrainingMetricsCollector


class TrainingContext:
    """
    Context manager for tracking training metrics with minimal code changes.

    This provides a clean API for instrumenting training loops without needing
    to manually manage the metrics collector lifecycle.

    Example:
        >>> from torch.training_metrics import enable_training_metrics, TrainingContext
        >>>
        >>> enable_training_metrics()
        >>>
        >>> with TrainingContext() as ctx:
        ...     for epoch in range(num_epochs):
        ...         ctx.set_epoch(epoch)
        ...         for batch_idx, (data, target) in enumerate(train_loader):
        ...             ctx.start_batch()
        ...
        ...             # Forward pass (automatically timed via hooks)
        ...             output = model(data)
        ...             loss = criterion(output, target)
        ...
        ...             # Backward pass
        ...             ctx.start_backward()
        ...             loss.backward()
        ...             ctx.end_backward()
        ...
        ...             # Optimizer step (automatically timed via hooks)
        ...             optimizer.step()
        ...             optimizer.zero_grad()
        ...
        ...             # Record loss and complete batch
        ...             ctx.end_batch(loss=loss.item())

    For even simpler usage with a decorator-style approach:
        >>> with TrainingContext() as ctx:
        ...     for epoch in range(num_epochs):
        ...         ctx.set_epoch(epoch)
        ...         for batch_idx, (data, target) in enumerate(train_loader):
        ...             with ctx.batch() as batch:
        ...                 output = model(data)
        ...                 loss = criterion(output, target)
        ...                 with batch.backward():
        ...                     loss.backward()
        ...                 optimizer.step()
        ...                 batch.set_loss(loss.item())
    """

    def __init__(self, collector: Optional[TrainingMetricsCollector] = None):
        """
        Initialize the training context.

        Args:
            collector: Optional metrics collector. If None, uses the global collector.
        """
        self._collector = collector
        self._owns_collector = False

    def __enter__(self) -> "TrainingContext":
        """Enter the training context."""
        if self._collector is None:
            self._collector = get_collector()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the training context."""
        pass

    @property
    def collector(self) -> TrainingMetricsCollector:
        """Get the underlying metrics collector."""
        if self._collector is None:
            self._collector = get_collector()
        return self._collector

    def set_epoch(self, epoch: int) -> None:
        """Set the current training epoch."""
        self.collector.set_epoch(epoch)

    def set_training_step(self, step: int) -> None:
        """Set the current training step."""
        self.collector.set_training_step(step)

    def start_batch(self) -> None:
        """
        Start timing a new training batch.

        Call this at the beginning of each batch iteration, before any
        data processing. This starts the data loading timer.
        """
        self.collector.start_batch()

    def start_backward(self) -> None:
        """
        Start timing the backward pass.

        Call this immediately before loss.backward().
        """
        self.collector.start_backward()

    def end_backward(self) -> None:
        """
        End timing the backward pass.

        Call this immediately after loss.backward() completes.
        """
        self.collector.end_backward()

    def end_batch(
        self,
        loss: Optional[float] = None,
        epoch: Optional[int] = None,
        training_step: Optional[int] = None,
    ) -> None:
        """
        End the current batch and record metrics.

        Args:
            loss: The training loss value for this batch.
            epoch: Override the epoch number.
            training_step: Override the training step.
        """
        self.collector.end_batch(loss=loss, epoch=epoch, training_step=training_step)

    def batch(self) -> "BatchContext":
        """
        Create a context manager for a single batch.

        Returns:
            A BatchContext that can be used with 'with' statement.

        Example:
            >>> with ctx.batch() as batch:
            ...     output = model(data)
            ...     loss = criterion(output, target)
            ...     with batch.backward():
            ...         loss.backward()
            ...     batch.set_loss(loss.item())
        """
        return BatchContext(self.collector)


class BatchContext:
    """Context manager for a single training batch."""

    def __init__(self, collector: TrainingMetricsCollector):
        self._collector = collector
        self._loss: Optional[float] = None

    def __enter__(self) -> "BatchContext":
        """Start the batch."""
        self._collector.start_batch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the batch and record metrics."""
        self._collector.end_batch(loss=self._loss)

    def set_loss(self, loss: float) -> None:
        """Set the loss value for this batch."""
        self._loss = loss

    def backward(self) -> "BackwardContext":
        """
        Create a context manager for the backward pass.

        Returns:
            A BackwardContext that can be used with 'with' statement.

        Example:
            >>> with batch.backward():
            ...     loss.backward()
        """
        return BackwardContext(self._collector)


class BackwardContext:
    """Context manager for timing the backward pass."""

    def __init__(self, collector: TrainingMetricsCollector):
        self._collector = collector

    def __enter__(self) -> "BackwardContext":
        """Start timing the backward pass."""
        self._collector.start_backward()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End timing the backward pass."""
        self._collector.end_backward()
