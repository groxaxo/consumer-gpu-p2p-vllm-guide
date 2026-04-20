#!/usr/bin/env bash
set -euo pipefail
#
# Post-reboot GPU validation for Intel platforms with consumer NVIDIA GPUs.
# Checks boot args, driver params, and runs NCCL SHM all-reduce to confirm
# TP=2 transport works.

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VLLM_VENV_PATH="${VLLM_VENV_PATH:-/home/op/venvs/vllm}"
CMDLINE="$(</proc/cmdline)"

echo "=== Step 1: Check required boot args ==="
printf '%s\n' "$CMDLINE"

if [[ " $CMDLINE " != *" intel_iommu=on "* || " $CMDLINE " != *" iommu=pt "* ]]; then
  echo ""
  echo "FAIL: expected intel_iommu=on iommu=pt."
  exit 1
fi
echo "PASS: boot args OK"

echo ""
echo "=== Step 2: Check RMForceP2PType is 0 ==="
grep RegistryDwords /proc/driver/nvidia/params || true

echo ""
echo "=== Step 3: Clear vLLM caches ==="
rm -f ~/.cache/vllm/gpu_p2p_access_cache_for_*.json
rm -rf ~/.cache/vllm/torch_compile_cache
echo "Caches cleared"

echo ""
echo "=== Step 4: NCCL SHM all-reduce test (${CUDA_DEVICES}) ==="
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$VLLM_VENV_PATH/bin/python" -c "
import os, torch, torch.distributed as dist, torch.multiprocessing as mp

def worker(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    tensor = torch.ones(1024, device=f'cuda:{rank}') * (rank + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(rank)
    expected = sum(range(1, world_size + 1))
    correct = torch.allclose(tensor, torch.full_like(tensor, expected))
    nans = tensor.isnan().sum().item()
    if rank == 0:
        if correct and nans == 0:
            print('PASS: NCCL all-reduce via SHM transport works.')
        else:
            print(f'FAIL: expected {expected}, got {tensor[0].item()}, NaNs={nans}')
            raise SystemExit(1)
    dist.destroy_process_group()

mp.spawn(worker, args=(2,), nprocs=2, join=True)
"
echo ""
echo "=== All post-reboot checks passed ==="
