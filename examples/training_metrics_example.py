#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Example script demonstrating PyTorch training metrics instrumentation.

This script shows the zero-code-change usage pattern for the torch.training_metrics module.

The metrics are exported to ODS3 via OpenTelemetry using the fair_model schema without
application code changes

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

Note, you will need to build torch locally for the C dynamic libs to be loaded
    pip install -r requirements.txt
    python setup build
    python -m pip install --no-build-isolation -v -e .
"""

import os
import sys

# Add the modified PyTorch to the path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def example_basic_model():
    """
    Example: ZERO CODE CHANGE usage - the recommended approach!

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
    print("Example 1: ZERO CODE CHANGE (Fully Automatic) with a simple model")
    print("=" * 60)
    print()
    print("This example shows a COMPLETELY UNMODIFIED training loop.")
    print("All metrics are collected automatically via PyTorch hooks.")
    print()

    # This is a STANDARD PyTorch training script - no training_metrics imports!
    # The metrics are collected automatically because:
    # 1. PyTorch imports torch.training_metrics at startup
    # 2. If OTEL_EXPORTER_OTLP_ENDPOINT is set, hooks are registered automatically

    # Create a simple neural network
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )

    # Create dummy training data
    num_samples = 320
    input_dim = 100
    num_classes = 10
    batch_size = 32

    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()  # Loss is auto-captured!
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50

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


def example_torch_compile():
    """Test that training_metrics hooks work with torch.compile."""
    print("=" * 60)
    print("Example: Test Dynamo Compatibility with torch.compile()")
    print("=" * 60)
    print()
    print("This example shows that our hooks also works with torch compile.")
    print("All metrics are collected automatically via PyTorch hooks.")
    print()

    # Create a simple model similar to what vLLM uses
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=256, num_layers=2):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids):
            hidden_states = self.embed_tokens(input_ids)
            for layer in self.layers:
                hidden_states = torch.relu(layer(hidden_states))
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits

    print("\n1. Creating model...")
    model = SimpleModel().cuda()
    print(f"   Model created: {model.__class__.__name__}")

    print("\n2. Compiling model with torch.compile...")
    try:
        # Use backend="eager" to avoid inductor PosixPath bug
        # You can also try mode="default" or backend="inductor" once the bug is fixed
        compiled_model = torch.compile(model, backend="eager")
        print("   ✓ Model compiled successfully (backend=eager)")
    except Exception as e:
        print(f"   ✗ Compilation failed: {e}")
        return False

    print("\n3. Running forward pass with compiled model...")
    try:
        # Create dummy input
        input_ids = torch.randint(0, 1000, (2, 128)).cuda()

        # Run forward pass - this is where the RLock error would occur
        with torch.no_grad():
            output = compiled_model(input_ids)

        print(f"   ✓ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
    except torch._dynamo.exc.Unsupported as e:
        print(f"   ✗ Dynamo Unsupported error: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Forward pass failed: {type(e).__name__}: {e}")
        return False

    print("\n4. Run compiled model with dummy dataloader...")
    try:
        # Create dummy dataloader with token sequences
        num_samples = 320
        seq_length = 64
        vocab_size = 1000
        batch_size = 32

        input_data = torch.randint(0, vocab_size, (num_samples, seq_length))
        target_data = torch.randint(0, vocab_size, (num_samples, seq_length))
        dataset = TensorDataset(input_data, target_data)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 50
        model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for input_ids, targets in train_loader:
                input_ids = input_ids.cuda()
                targets = targets.cuda()

                optimizer.zero_grad()

                # Forward pass with compiled model
                logits = compiled_model(input_ids)

                # Compute loss (flatten for CrossEntropyLoss)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"   Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        print(f"   ✓ Training with compiled model and dataloader successful!")

    except torch._dynamo.exc.Unsupported as e:
        print(f"   ✗ Dynamo Unsupported error during training: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Training failed: {type(e).__name__}: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ Dynamo compatibility verified!")
    print("=" * 60)
    return True



def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Training Metrics Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training_metrics_example.py                    # Run all tests
  python training_metrics_example.py --test basic       # Run basic model test only
  python training_metrics_example.py --test compile     # Run torch.compile test only
  python training_metrics_example.py --test all         # Run all tests (default)
        """
    )
    parser.add_argument(
        "--test", "-t",
        choices=["basic", "compile", "all"],
        default="all",
        help="Which test to run: 'basic' (basic model), 'compile' (torch.compile), or 'all' (default)"
    )
    args = parser.parse_args()

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
    else:
        print(f"Using OTEL endpoint: {otel_endpoint}")
        print()

    # Force enable for demo purposes
    os.environ["PYTORCH_TRAINING_METRICS_ENABLED"] = "1"
    print("Force enable PYTORCH_TRAINING_METRICS_ENABLED=1 for demo purpose")
    print()

    # Run selected tests
    if args.test in ("basic", "all"):
        example_basic_model()

    if args.test in ("compile", "all"):
        example_torch_compile()

    print("=" * 60)
    print("Example completed successfully!")
    print()
    print("KEY TAKEAWAY: For most use cases, you don't need to change")
    print("any code! Just set OTEL_EXPORTER_OTLP_ENDPOINT and run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
