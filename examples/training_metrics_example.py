#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Example script demonstrating PyTorch training metrics instrumentation.

This script shows different usage patterns for the torch.training_metrics module,
from fully automatic (zero code changes) to various manual control options.

The metrics are exported to ODS3 via OpenTelemetry using the fair_model schema.

ZERO-CODE-CHANGE USAGE (Recommended):
=====================================
Simply set these environment variables before running your existing training script:

    export OTEL_EXPORTER_OTLP_ENDPOINT="http://<otel-gateway-endpoint>"
    export PYTORCH_TRAINING_METRICS_ENABLED=1  # Optional, enables even without endpoint

That's it! No code changes required. The following metrics are collected automatically:
- epoch: Detected from DataLoader exhaustion (StopIteration)
- train_batch_time: Total time for each training batch
- train_data_loading_time: Time spent in DataLoader fetching data
- train_forward_pass_time: Time spent in forward pass (via nn.Module hooks)
- train_backward_pass_time: Time spent in backward pass (via autograd.backward)
- train_loss: Loss value (captured from loss modules like CrossEntropyLoss)
- training_step: Batch counter

For SLURM jobs, the following environment variables are automatically used:
- SLURM_JOB_ID
- SLURMD_NODENAME
- SCENV (cluster name)
- SLURM_PROCID (global rank)
- SLURM_LOCALID (local rank)

Usage:
    # Option 1: Zero-code-change (set env vars and run your existing script)
    export OTEL_EXPORTER_OTLP_ENDPOINT="http://<endpoint>"
    python your_existing_training_script.py

    # Option 2: Run this demo script
    python training_metrics_example.py
"""

import os
import sys

# Add the modified PyTorch to the path
sys.path.insert(0, "/checkpoint/fairinfra/fair_cw_admins/conda_build/third-party/pytorch")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def create_simple_model():
    """Create a simple neural network for demonstration."""
    return nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )


def create_dummy_data(num_samples=1000, input_dim=100, num_classes=10, batch_size=32):
    """Create dummy training data."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def example_zero_code_change():
    """
    Example 0: ZERO CODE CHANGE usage - the recommended approach!

    This demonstrates what happens when you run an unmodified training script
    with OTEL_EXPORTER_OTLP_ENDPOINT set. The metrics are collected AUTOMATICALLY:

    - Forward pass time: Captured via nn.Module hooks
    - Backward pass time: Captured via autograd.backward instrumentation
    - Optimizer step time: Captured via Optimizer hooks
    - Loss value: Captured via hooks on CrossEntropyLoss, MSELoss, etc.
    - Data loading time: Captured via DataLoader.__next__ instrumentation
    - Epochs: Detected automatically when DataLoader raises StopIteration

    No special imports, no context managers, no decorators needed!
    """
    print("=" * 60)
    print("Example 0: ZERO CODE CHANGE (Fully Automatic)")
    print("=" * 60)
    print()
    print("This example shows a COMPLETELY UNMODIFIED training loop.")
    print("All metrics are collected automatically via PyTorch hooks.")
    print()

    # This is a STANDARD PyTorch training script - no training_metrics imports!
    # The metrics are collected automatically because:
    # 1. PyTorch imports torch.training_metrics at startup
    # 2. If OTEL_EXPORTER_OTLP_ENDPOINT is set, hooks are registered automatically

    model = create_simple_model()
    train_loader = create_dummy_data(num_samples=320, batch_size=32)  # 10 batches
    criterion = nn.CrossEntropyLoss()  # Loss is auto-captured!
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 2

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        # Standard training loop - NO special calls needed!
        for data, target in train_loader:  # Data loading time auto-captured
            optimizer.zero_grad()

            output = model(data)  # Forward time auto-captured via hooks
            loss = criterion(output, target)  # Loss value auto-captured

            loss.backward()  # Backward time auto-captured via autograd hook
            optimizer.step()  # Optimizer time auto-captured via hook

            total_loss += loss.item()
            num_batches += 1

        # Epoch ends when DataLoader is exhausted - auto-detected!
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    print()
    print("Training complete!")
    print("All metrics were collected AUTOMATICALLY - no code changes!")
    print()


def example_basic_usage():
    """
    Example 1: Basic usage with minimal code changes.

    This approach uses the TrainingContext to track epochs and batches,
    while forward pass and optimizer step timing is automatic via hooks.
    """
    print("=" * 60)
    print("Example 1: Basic Usage with TrainingContext")
    print("=" * 60)

    from torch.training_metrics import (
        enable_training_metrics,
        disable_training_metrics,
        TrainingContext,
    )

    # Create model, data, optimizer
    model = create_simple_model()
    train_loader = create_dummy_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Enable automatic metrics collection
    # This registers global hooks on nn.Module and Optimizer
    enable_training_metrics()

    num_epochs = 3

    # Use TrainingContext to track epochs and batches
    with TrainingContext() as ctx:
        for epoch in range(num_epochs):
            ctx.set_epoch(epoch)
            total_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                # Start batch timing (also starts data loading timer)
                ctx.start_batch()

                # Forward pass (automatically timed via hooks)
                output = model(data)
                loss = criterion(output, target)

                # Backward pass (timed with start/end calls)
                ctx.start_backward()
                optimizer.zero_grad()
                loss.backward()
                ctx.end_backward()

                # Optimizer step (automatically timed via hooks)
                optimizer.step()

                # End batch and record metrics
                ctx.end_batch(loss=loss.item())

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # Disable metrics collection when done
    disable_training_metrics()

    print("Training complete!\n")


def example_context_manager_style():
    """
    Example 2: Using nested context managers for cleaner code.

    This approach uses ctx.batch() and batch.backward() context managers
    for automatic timing with less boilerplate.
    """
    print("=" * 60)
    print("Example 2: Nested Context Managers (Cleaner Style)")
    print("=" * 60)

    from torch.training_metrics import (
        enable_training_metrics,
        disable_training_metrics,
        TrainingContext,
    )

    model = create_simple_model()
    train_loader = create_dummy_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    enable_training_metrics()

    num_epochs = 2
    global_step = 0

    with TrainingContext() as ctx:
        for epoch in range(num_epochs):
            ctx.set_epoch(epoch)

            for batch_idx, (data, target) in enumerate(train_loader):
                # Using nested context managers for automatic timing
                with ctx.batch() as batch:
                    # Forward pass (auto-timed)
                    output = model(data)
                    loss = criterion(output, target)

                    # Backward pass with context manager
                    with batch.backward():
                        optimizer.zero_grad()
                        loss.backward()

                    # Optimizer step (auto-timed)
                    optimizer.step()

                    # Set the loss value
                    batch.set_loss(loss.item())

                global_step += 1

            print(f"Epoch {epoch + 1}/{num_epochs} completed")

    disable_training_metrics()
    print("Training complete!\n")


def example_custom_config():
    """
    Example 3: Using custom configuration.

    This shows how to customize the metrics collection behavior.
    """
    print("=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)

    from torch.training_metrics import (
        enable_training_metrics,
        disable_training_metrics,
        TrainingContext,
        MetricsConfig,
    )

    # Create custom configuration
    config = MetricsConfig(
        # Log every 5 batches to reduce overhead
        log_every_n_batches=5,
        # Export metrics every 10 seconds
        export_interval_ms=10000,
        # Custom job metadata
        job_id=os.getenv("SLURM_JOB_ID", "custom_job_123"),
        cluster="my_cluster",
        # Disable optimizer timing if not needed
        track_optimizer_time=True,
    )

    model = create_simple_model()
    train_loader = create_dummy_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Enable with custom config
    enable_training_metrics(config=config)

    with TrainingContext() as ctx:
        for epoch in range(2):
            ctx.set_epoch(epoch)

            for batch_idx, (data, target) in enumerate(train_loader):
                ctx.start_batch()

                output = model(data)
                loss = criterion(output, target)

                ctx.start_backward()
                optimizer.zero_grad()
                loss.backward()
                ctx.end_backward()

                optimizer.step()
                ctx.end_batch(loss=loss.item())

            print(f"Epoch {epoch + 1}/2 completed")

    disable_training_metrics()
    print("Training complete!\n")


def example_direct_collector_access():
    """
    Example 4: Direct access to the collector for advanced use cases.

    This shows how to directly access the metrics collector for
    custom timing or metrics not covered by the standard API.
    """
    print("=" * 60)
    print("Example 4: Direct Collector Access")
    print("=" * 60)

    from torch.training_metrics import enable_training_metrics, get_collector

    model = create_simple_model()
    train_loader = create_dummy_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Enable and get the collector
    collector = enable_training_metrics()

    for epoch in range(2):
        collector.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            collector.start_batch()

            # Forward
            output = model(data)
            loss = criterion(output, target)

            # Backward
            collector.start_backward()
            optimizer.zero_grad()
            loss.backward()
            collector.end_backward()

            # Step
            optimizer.step()

            # Record with explicit training step
            collector.end_batch(
                loss=loss.item(),
                training_step=epoch * len(train_loader) + batch_idx
            )

        print(f"Epoch {epoch + 1}/2 completed")

    # Shutdown properly to flush metrics
    collector.shutdown()
    print("Training complete!\n")


def main():
    """Run all examples."""
    # Check if OTEL endpoint is configured
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otel_endpoint:
        print("=" * 60)
        print("INFO: OTEL_EXPORTER_OTLP_ENDPOINT not set")
        print()
        print("For ZERO-CODE-CHANGE usage, set this environment variable:")
        print("  export OTEL_EXPORTER_OTLP_ENDPOINT='http://<endpoint>'")
        print()
        print("Metrics will still be collected in this demo,")
        print("but won't be exported to ODS3.")
        print("=" * 60)
        print()

        # Force enable for demo purposes
        os.environ["PYTORCH_TRAINING_METRICS_ENABLED"] = "1"
    else:
        print(f"Using OTEL endpoint: {otel_endpoint}")
        print()

    # Run examples - starting with the zero-code-change example!
    example_zero_code_change()
    example_basic_usage()
    example_context_manager_style()
    example_custom_config()
    example_direct_collector_access()

    print("=" * 60)
    print("All examples completed successfully!")
    print()
    print("KEY TAKEAWAY: For most use cases, you don't need to change")
    print("any code! Just set OTEL_EXPORTER_OTLP_ENDPOINT and run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
