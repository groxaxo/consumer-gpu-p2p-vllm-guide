"""
Standalone NCCL all-reduce test for TP=2.

Verifies that NCCL can correctly perform all-reduce operations across two GPUs
using SHM transport. This is the actual transport path used by vLLM on Intel
consumer platforms where BAR-mapped P2P doesn't work.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python test_nccl_tp2.py
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_DEBUG_SUBSYS"] = os.environ.get("NCCL_DEBUG_SUBSYS", "INIT,P2P,SHM")

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    tensor = torch.ones(1024, device=f"cuda:{rank}") * (rank + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(rank)

    expected = sum(range(1, world_size + 1))
    correct = torch.allclose(tensor, torch.full_like(tensor, expected))
    nans = tensor.isnan().sum().item()

    if rank == 0:
        print(f"\n=== NCCL ALL-REDUCE RESULT ===")
        print(f"Expected: {expected}, Got: {tensor[0].item()}, Correct: {correct}, NaNs: {nans}")
        if correct and nans == 0:
            print("SUCCESS: NCCL all-reduce works with TP=2!")
        else:
            print(f"FAIL: values wrong. First 5: {tensor[:5].tolist()}")
            raise SystemExit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
