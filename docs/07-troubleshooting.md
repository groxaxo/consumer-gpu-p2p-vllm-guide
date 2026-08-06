# 7. Troubleshooting

Start with the failed line from:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  ~/venvs/vllm/bin/python scripts/p2p_doctor.py validate \
    --venv ~/venvs/vllm
```

Do not skip a failed gate and continue calling the result P2P-enabled.

## Driver/userspace version mismatch

Symptoms:

- `nvidia-smi`: `Failed to initialize NVML: Driver/library version mismatch`
- `modinfo nvidia` reports `595.58.03`, but `nvidia-smi` reports another version
- the installer refuses to build the patch

Compare all layers:

```bash
modinfo -F version nvidia
grep -E 'NVRM version|Kernel Module' /proc/driver/nvidia/version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

All must be exactly `595.58.03` after reboot. Reinstall matching userspace from
the exact official runfile when necessary:

```bash
sudo sh /opt/nvidia-p2p/NVIDIA-Linux-x86_64-595.58.03.run \
  --silent --ui=none --no-questions --accept-license --no-kernel-modules
sudo reboot
```

The reboot is required when the loaded kernel module was a different version.

## Patched module built but stock module loads

Inspect module paths and license:

```bash
modinfo -n nvidia
modinfo -F version nvidia
modinfo -F license nvidia
dkms status
find /lib/modules/"$(uname -r)" -name 'nvidia*.ko*' -print
```

Expected license is `Dual MIT/GPL`, and the DKMS status must show the current
kernel installed. Rebuild dependency maps and initramfs:

```bash
sudo depmod -a
sudo update-initramfs -u
sudo reboot
```

If an apt `nvidia-dkms-*` package keeps replacing the module, remove the
conflict or use the installer's explicit `--lock-driver` mode after the exact
stack is working.

## Boot configuration fails

```bash
cat /proc/cmdline
find /sys/kernel/iommu_groups -mindepth 1 -maxdepth 1 -type d | wc -l
```

Required:

- Intel: `intel_iommu=on iommu=pt`
- AMD: `amd_iommu=on iommu=pt`
- populated IOMMU groups

Regenerate GRUB and reboot:

```bash
sudo update-grub
sudo reboot
```

The guide no longer hides PCIe errors with `pci=noaer`. AER messages are useful
evidence; fix the link/riser/slot issue rather than suppressing it.

## Direct kernel peer read/write fails

A capability report of `1` followed by `read=FAIL` or `write=FAIL` means the
mapping is advertised but not correct. Do not set `VLLM_SKIP_P2P_CHECK=1`.

Collect:

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
lspci -tv
sudo lspci -vv -s <each-GPU-BDF> | grep -E 'LnkCap:|LnkSta:|ACSCtl:'
journalctl -k -b | grep -Ei 'NVRM|Xid|AER|BadTLP|Unsupported Request'
```

Then check:

1. both cards use CPU-connected slots when possible;
2. slot bifurcation matches the board layout;
3. Above 4G Decoding is enabled;
4. ACS firmware settings do not force an unusable route;
5. risers are stable at the negotiated generation/width;
6. forcing Gen3 removes signal-integrity errors;
7. an RTX 3090 NVLink bridge is correctly seated when that path is intended.

If the platform cannot route the pair correctly, use a different pair or the
explicit SHM fallback. Software cannot validate corrupted hardware traffic into
correctness.

## vLLM CUDA IPC fails while direct kernel access passes

This narrows the issue to cross-process CUDA IPC/runtime behavior. Remove only
the profile-specific vLLM cache and regenerate it:

```bash
find ~/.cache/vllm -maxdepth 1 -name 'gpu_p2p_access_cache_for_*.json' -print
rm ~/.cache/vllm/gpu_p2p_access_cache_for_<exact-device-key>.json

CUDA_VISIBLE_DEVICES=0,1 \
VLLM_SKIP_P2P_CHECK=0 \
  ~/venvs/vllm/bin/python scripts/p2p_doctor.py validate \
    --venv ~/venvs/vllm
```

Do not delete every cache on each start. The cache is keyed by the visible
device mapping and is useful once a pair has passed.

Check for mixed CUDA libraries:

```bash
~/venvs/vllm/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
PY

echo "$LD_LIBRARY_PATH"
ldd ~/venvs/vllm/lib/python*/site-packages/torch/lib/libtorch_cuda.so | grep -E 'cuda|nccl'
```

## NCCL exact-value test fails

Run the standalone test with logs:

```bash
source ~/venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_P2P_DISABLE=0 \
NCCL_SHM_DISABLE=0 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM \
  python scripts/test_nccl_tp2.py 2>&1 | tee /tmp/nccl-p2p.log
```

Investigate the first NCCL or CUDA error, not only the final timeout. Check for
another workload on the selected GPUs and for stale processes:

```bash
nvidia-smi
ps -ef | grep -E '[v]llm|[t]orchrun|[p]ython.*nccl'
```

When direct peer and CUDA IPC pass but NCCL explicitly selects SHM, keep the
result labelled as SHM. Strict profile generation rejects that state unless
`--allow-nccl-shm` is deliberately supplied.

## vLLM logs “custom all-reduce is disabled”

Interpret the reason:

- **world size 3**: expected; vLLM custom all-reduce does not support TP=3.
  NCCL can still use validated P2P.
- **P2P test failed**: not expected for a validated TP=2 profile. Re-run the
  doctor and inspect the vLLM IPC gate.
- **more than two PCIe-only GPUs / not fully connected**: vLLM may choose NCCL
  even when individual peer pairs work.

Do not suppress the message without understanding which branch produced it.

## vLLM OOM during graph capture

P2P can be correct while the model still lacks graph-capture headroom. Use:

```bash
VLLM_ENFORCE_EAGER=1 \
VLLM_GPU_MEMORY_UTILIZATION=0.88 \
  bash scripts/manage_vllm_safe_tp2.sh restart <model-id>
```

Then tune memory utilization and context independently of P2P.

## Profile is stale

This is intentional after a kernel, driver, boot, GPU-order, slot, or selected
set change. Regenerate:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/manage_vllm_safe_tp2.sh revalidate
```

Never hand-edit the fingerprint to bypass testing.

## Port is occupied

The launcher refuses to adopt or kill an unmanaged listener:

```bash
sudo lsof -nP -iTCP:8000 -sTCP:LISTEN
ps -fp <pid>
```

Stop that service explicitly or choose another port:

```bash
VLLM_PORT=8001 bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

## Emergency SHM fallback

```bash
VLLM_P2P_MODE=shm \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

This is designed to preserve correct inference while P2P is repaired. It is not
a successful P2P outcome and should be benchmarked against single-GPU or other
parallelism strategies.
