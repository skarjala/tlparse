#!/usr/bin/env python3

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group(
        "nccl" if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def run_training(rank, world_size):
    """Run training on a specific rank."""
    print(f"Running on rank {rank}")
    setup(rank, world_size)

    # Create model and move it to GPU if available
    device = torch.device(
        f"cuda:{rank}"
        if torch.cuda.is_available() and torch.cuda.device_count() > rank
        else "cpu"
    )
    model = SimpleModel().to(device)

    # Wrap model with DDP
    ddp_model = DDP(
        model,
        device_ids=(
            [rank]
            if torch.cuda.is_available() and torch.cuda.device_count() > rank
            else None
        ),
    )

    # Compile the model - this should generate torch trace logs
    compiled_model = torch.compile(ddp_model)

    # Create some dummy data
    batch_size = 8
    input_data = torch.randn(batch_size, 10).to(device)
    target = torch.randn(batch_size, 1).to(device)

    # Create optimizer
    optimizer = optim.SGD(compiled_model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(3):
        print(f"Rank {rank}, Epoch {epoch}")

        # Forward pass
        output = compiled_model(input_data)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item():.4f}")

    cleanup()


def main():
    world_size = 2  # Use 2 processes/ranks

    # Check if we have TORCH_TRACE set
    if "TORCH_TRACE" not in os.environ:
        print("Warning: TORCH_TRACE environment variable not set!")
        print(
            "Please run with: TORCH_TRACE=/tmp/multi_rank_trace python multi_rank_example.py"
        )
        return

    print(f"TORCH_TRACE is set to: {os.environ['TORCH_TRACE']}")

    # Spawn processes for distributed training
    mp.spawn(run_training, args=(world_size,), nprocs=world_size, join=True)

    print("Multi-rank training completed!")
    print(f"Check the trace logs in: {os.environ['TORCH_TRACE']}")


if __name__ == "__main__":
    main()
