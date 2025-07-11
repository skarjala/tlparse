# multi_rank_fakePG.py  (patched)

import os, torch, torch.distributed as dist, torch.multiprocessing as mp

from torch.testing._internal.distributed.fake_pg import FakeProcessGroup, FakeStore

WORLD_SIZE = 4
OUT_DIR = "logs"


def setup_logging(rank: int):
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, f"rank_{rank}.log")
    # plain-message formatter so tlparse can read each line as raw JSON
    import logging, sys

    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    # silence duplicate stderr output
    root.propagate = False


def worker(rank: int, world_size: int):
    setup_logging(rank)

    # ───── Fake PG/bootstrap ─────
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("fake", rank=rank, world_size=world_size)

    # ───── Tiny workload with a collective ─────
    def f(x):
        y = x * 2
        dist.all_reduce(y)  # hits fake PG
        return y.relu()

    x = torch.randn(8, 8)

    # Try without compilation first to test fake PG
    result = f(x)
    print(f"Rank {rank}: Function executed successfully, result shape: {result.shape}")

    # Now try with compilation
    try:
        torch._dynamo.reset()
        torch._dynamo.config.dynamic_shapes = False
        compiled = torch.compile(f)
        compiled_result = compiled(x)
        print(f"Rank {rank}: Compiled function executed successfully")
    except Exception as e:
        print(f"Rank {rank}: Compilation failed: {e}")
        print(f"Rank {rank}: Continuing without compilation...")

    print(f"Rank {rank} completed")
    dist.destroy_process_group()


if __name__ == "__main__":
    # Use TORCH_TRACE instead of TORCH_LOGS for better compatibility
    os.environ.setdefault("TORCH_TRACE", "./logs/trace")
    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    print(f"✓ generated {WORLD_SIZE} logs in ./{OUT_DIR}")
    print("  now run: tlparse ./logs --all-ranks-html -o tl_out")
