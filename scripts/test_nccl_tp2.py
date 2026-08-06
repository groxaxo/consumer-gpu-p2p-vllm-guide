#!/usr/bin/env python3
"""Standalone NCCL correctness test with P2P enabled.

For the complete gate (driver, boot config, direct peer kernels, CUDA IPC,
NCCL, and profile generation), use p2p_doctor.py validate instead.
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import sys
import tempfile
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int, port: int, output_path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    local_pass = torch.ones(1, dtype=torch.int32, device=rank)
    rows: list[dict[str, Any]] = []
    for elements in (1024, 1 << 20, 8 << 20):
        tensor = torch.full(
            (elements,), float(rank + 1), dtype=torch.float32, device=rank
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(rank)
        expected = float(world_size * (world_size + 1) // 2)
        finite = bool(torch.isfinite(tensor).all().item())
        correct = bool(torch.all(tensor == expected).item())
        if not (finite and correct):
            local_pass.zero_()
        rows.append(
            {
                "rank": rank,
                "elements": elements,
                "expected": expected,
                "finite": finite,
                "correct": correct,
            }
        )

    dist.all_reduce(local_pass, op=dist.ReduceOp.MIN)
    if rank == 0:
        payload = {
            "world_size": world_size,
            "passed": bool(local_pass.item()),
            "rank0_checks": rows,
        }
        pathlib.Path(output_path).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    dist.barrier()
    dist.destroy_process_group()


def main() -> int:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("At least two visible CUDA devices are required.", file=sys.stderr)
        return 1

    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,GRAPH,P2P,SHM")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.NamedTemporaryFile(prefix="nccl-p2p-", delete=False) as handle:
        output_path = handle.name
    try:
        mp.spawn(worker, args=(world_size, port, output_path), nprocs=world_size, join=True)
        payload = json.loads(pathlib.Path(output_path).read_text(encoding="utf-8"))
        print(json.dumps(payload, indent=2))
        return 0 if payload.get("passed") is True else 1
    finally:
        pathlib.Path(output_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
