#!/usr/bin/env python3
"""
Multi-rank distributed example using FakeProcessGroup for tlparse testing.
This simulates multiple ranks in a single process, making it easier to debug
and generate consistent trace logs.
"""

import os
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup


class SimpleModel(nn.Module):
    """Simple neural network for testing distributed training."""

    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def setup_tracing(trace_dir, rank):
    """Set up PyTorch tracing environment for the given rank."""
    # Create rank-specific trace directory
    rank_trace_dir = os.path.join(trace_dir, f"rank_{rank}")
    os.makedirs(rank_trace_dir, exist_ok=True)

    # Set environment variables for tracing
    os.environ["TORCH_TRACE"] = rank_trace_dir
    os.environ["TORCH_LOGS"] = "+dynamo,+inductor,+aot"
    os.environ["TORCH_LOGS_FORMAT"] = "json"

    # Additional tracing options to ensure logs are generated
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_COMPILE_DEBUG"] = "1"

    print(f"Rank {rank}: Tracing to {rank_trace_dir}")
    print(f"Rank {rank}: TORCH_TRACE={os.environ.get('TORCH_TRACE')}")
    print(f"Rank {rank}: TORCH_LOGS={os.environ.get('TORCH_LOGS')}")
    return rank_trace_dir


def simulate_rank_training(rank, world_size, trace_dir):
    """Simulate training for a specific rank using FakeProcessGroup."""
    print(f"\n=== Simulating Rank {rank}/{world_size-1} ===")

    # Set up tracing for this rank
    rank_trace_dir = setup_tracing(trace_dir, rank)

    # Create FakeProcessGroup for this rank
    fake_pg = FakeProcessGroup(rank=rank, world_size=world_size)

    # Create model with slight variations per rank
    torch.manual_seed(42 + rank)  # Different seed per rank
    model = SimpleModel()

    # Compile the model to generate trace logs (without DDP)
    print(f"Rank {rank}: Compiling model...")
    compiled_model = torch.compile(model, mode="default")

    # Create training data (slightly different per rank)
    batch_size = 16
    torch.manual_seed(100 + rank)
    input_data = torch.randn(batch_size, 10)
    target = torch.randn(batch_size, 1)

    # Create optimizer
    optimizer = optim.Adam(compiled_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Rank {rank}: Epoch {epoch + 1}/{num_epochs}")

        # Forward pass
        optimizer.zero_grad()
        output = compiled_model(input_data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Simulate distributed operations using FakeProcessGroup
        if epoch % 2 == 0:
            # Simulate all-reduce operation on gradients
            for param in compiled_model.parameters():
                if param.grad is not None:
                    # Create a copy of the gradient for all-reduce simulation
                    grad_tensor = param.grad.clone()
                    fake_pg.allreduce(grad_tensor)
                    # In real distributed training, this would average gradients across ranks

        # Simulate barrier synchronization
        dummy_tensor = torch.tensor([rank], dtype=torch.float32)
        fake_pg.allreduce(dummy_tensor)

        print(f"Rank {rank}: Epoch {epoch + 1}, Loss: {loss.item():.6f}")

    print(f"Rank {rank}: Training completed")
    return rank_trace_dir


def run_multi_rank_simulation(world_size=4, trace_base_dir=None):
    """Run multi-rank simulation using FakeProcessGroup."""

    # Create trace directory
    if trace_base_dir is None:
        trace_base_dir = tempfile.mkdtemp(prefix="tlparse_fakeprocessgroup_")

    print(f"=== Multi-Rank FakeProcessGroup Simulation ===")
    print(f"World size: {world_size}")
    print(f"Base trace directory: {trace_base_dir}")

    rank_trace_dirs = []

    # Simulate each rank sequentially
    for rank in range(world_size):
        try:
            rank_trace_dir = simulate_rank_training(rank, world_size, trace_base_dir)
            rank_trace_dirs.append(rank_trace_dir)
        except Exception as e:
            print(f"Error simulating rank {rank}: {e}")
            continue

    print(f"\n=== Simulation Complete ===")
    print(f"Generated traces for {len(rank_trace_dirs)} ranks")

    # Check for torch_compile_debug files (where traces actually go)
    debug_dir = os.path.join(os.getcwd(), "torch_compile_debug")
    total_files = 0

    if os.path.exists(debug_dir):
        print(f"\nChecking torch_compile_debug directory: {debug_dir}")

        # Find all JSON and log files
        import glob

        json_files = glob.glob(os.path.join(debug_dir, "**", "*.json"), recursive=True)
        log_files = glob.glob(os.path.join(debug_dir, "**", "*.log"), recursive=True)
        all_files = json_files + log_files

        total_files = len(all_files)
        print(f"Found {len(json_files)} JSON files and {len(log_files)} log files")

        # Show some example files
        for f in all_files[:5]:  # Show first 5 files
            rel_path = os.path.relpath(f, debug_dir)
            size = os.path.getsize(f)
            print(f"  📄 {rel_path} ({size} bytes)")
        if len(all_files) > 5:
            print(f"  ... and {len(all_files) - 5} more files")

        # Copy files to organized structure for tlparse
        if total_files > 0:
            organized_dir = os.path.join(trace_base_dir, "organized_traces")
            os.makedirs(organized_dir, exist_ok=True)

            import shutil

            for i, f in enumerate(all_files):
                filename = os.path.basename(f)
                # Add rank prefix to make it multi-rank-like
                rank_id = i % world_size
                new_filename = f"rank_{rank_id}_{filename}"
                dest_path = os.path.join(organized_dir, new_filename)
                shutil.copy2(f, dest_path)

            print(
                f"\nCopied {total_files} files to organized structure: {organized_dir}"
            )
            trace_base_dir = organized_dir

    # Also check the original rank directories
    for i, rank_dir in enumerate(rank_trace_dirs):
        if os.path.exists(rank_dir):
            files = [
                f
                for f in os.listdir(rank_dir)
                if os.path.isfile(os.path.join(rank_dir, f))
            ]
            if files:
                print(f"Rank {i}: {len(files)} files in {rank_dir}")

    print(f"\nTotal files found: {total_files}")

    if total_files > 0:
        print(f"\n🎉 Success! Generated trace logs from torch.compile debug output")
        print(f"\nTo analyze with tlparse:")
        print(f"  cd /Users/skarjala/Desktop/tlparse")
        print(
            f"  ./target/release/tlparse {trace_base_dir} -o tlparse_output --overwrite"
        )

        return trace_base_dir
    else:
        print(f"\n❌ No trace files were generated")
        return None


def run_tlparse_analysis(trace_dir):
    """Run tlparse on the generated trace files."""
    import subprocess

    output_dir = os.path.join(trace_dir, "analysis")
    tlparse_binary = "/Users/skarjala/Desktop/tlparse/target/release/tlparse"

    if not os.path.exists(tlparse_binary):
        print(f"❌ tlparse binary not found at {tlparse_binary}")
        print("Please build tlparse first: cargo build --release")
        return False

    cmd = [tlparse_binary, trace_dir, "-o", output_dir, "--overwrite"]

    print(f"Running tlparse: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print(f"✅ tlparse analysis completed successfully!")
            print(f"Output directory: {output_dir}")

            # Check for generated files
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                print(f"Generated {len(files)} analysis files:")
                for f in sorted(files)[:10]:  # Show first 10 files
                    print(f"  📄 {f}")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more files")

                # Look for key files
                index_file = os.path.join(output_dir, "index.html")
                if os.path.exists(index_file):
                    print(f"\n🌐 Open in browser: file://{index_file}")

            return True
        else:
            print(f"❌ tlparse failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ tlparse timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running tlparse: {e}")
        return False


def main():
    """Main function to run the FakeProcessGroup example."""
    print("PyTorch FakeProcessGroup Multi-Rank Example for tlparse")
    print("=" * 60)

    # Configuration
    world_size = 4  # Simulate 4 ranks

    # Run the simulation
    trace_dir = run_multi_rank_simulation(world_size)

    if trace_dir:
        print(f"\n📁 Trace directory: {trace_dir}")

        # Ask user if they want to run tlparse analysis
        try:
            response = (
                input("\nRun tlparse analysis on generated traces? (y/n): ")
                .strip()
                .lower()
            )
            if response in ["y", "yes"]:
                success = run_tlparse_analysis(trace_dir)
                if success:
                    print("\n🎉 Complete! Check the analysis output directory.")
                else:
                    print("\n⚠️  Analysis failed, but trace files are still available.")
            else:
                print(f"\nTrace files saved to: {trace_dir}")
                print("You can run tlparse manually later.")
        except KeyboardInterrupt:
            print("\nSkipped analysis.")

    print(f"\n✨ Example complete!")


if __name__ == "__main__":
    # Ensure we're using the right multiprocessing method
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
