# 3. P2P Transport Diagnostics

## Understanding the Three P2P Paths

There are three distinct "P2P" mechanisms in CUDA. They behave very differently
on Intel consumer platforms:

### 1. `cudaMemcpyPeer` (DMA Staging)

```
GPU0 VRAM → PCIe → Host RAM → PCIe → GPU1 VRAM
```

The driver stages data through system RAM. This is not true peer-to-peer, but
it works on any multi-GPU system. Throughput is ~6 GB/s, limited by PCIe +
DDR bandwidth. The patched driver enables this path on consumer GPUs.

**Status on Intel consumer: WORKS**

### 2. Direct BAR-Mapped Access

```
GPU0 → PCIe TLP → GPU1 VRAM (via BAR window)
```

One GPU maps a region of another GPU's VRAM into its own PCIe address space.
Reads/writes go directly over PCIe without touching host RAM. This requires the
root complex to route peer TLPs between different root ports.

**Status on Intel consumer: FAILS** — the CPU root complex drops or misroutes
the TLPs. Data comes back as NaN with no error. This is a silicon-level
limitation of Intel consumer root complexes; it is not fixable in software.

### 3. CUDA IPC (Inter-Process Communication)

```
Process A (GPU0) → cudaIpcGetMemHandle → Process B (GPU1)
```

Shares GPU memory between OS processes via BAR-mapped memory handles. Internally
depends on the same BAR-mapped access as path 2.

**Status on Intel consumer: FAILS** — same root complex limitation. This is
what vLLM's `can_actually_p2p()` test catches.

---

## Quick Diagnostic: The Bandwidth Benchmark

The repo includes a CUDA benchmark that tests all GPU pairs and produces a full
system report. This is the fastest way to characterize your hardware:

```bash
python3 scripts/p2p_bandwidth_bench.py
# optionally save the report
python3 scripts/p2p_bandwidth_bench.py --save bench_results.txt
```

The script compiles the CUDA source (`scripts/p2p_bandwidth_bench.cu`),
runs all GPU pairs, and emits:

- Unidirectional bandwidth at 1 / 16 / 64 / 256 MiB
- Bidirectional bandwidth (simultaneous both directions)
- Round-trip latency (4-byte ping, 200 rounds)
- Host↔device baseline per GPU
- Full system header: driver version, `lspci LnkSta`, `nvidia-smi topo`

Look for the `[Summary & Recommendations]` section at the end. Pairs below
~3 GB/s are bandwidth-starved and should be avoided for TP=2 inference.

---

## NCCL Transport Detection

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM \
  python3 scripts/test_nccl_tp2.py
```

Key lines in the output:

```
Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 1
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
```

- `isAllDirectP2p 0` — NCCL confirmed direct BAR P2P is not available
- `via SHM/direct/direct` — NCCL selected shared-host-memory transport ✓

If you see `via P2P` instead, NCCL believes P2P works. On Intel consumer
hardware this typically leads to hangs or data corruption — set
`NCCL_P2P_DISABLE=1` to force SHM.

---

## vLLM P2P Detection

vLLM runs its own IPC test at startup. With `VLLM_SKIP_P2P_CHECK=0`:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=0 VLLM_SKIP_P2P_CHECK=0 \
  ~/venvs/vllm/bin/vllm serve Qwen/Qwen3.5-9B \
    --tensor-parallel-size 2 --enforce-eager
```

Watch the startup log for:

```
Custom allreduce is disabled because your platform lacks GPU P2P capability
or P2P test failed.
```

This message confirms vLLM's `can_actually_p2p()` function spawned two
processes, tested CUDA IPC handle sharing, caught the failure, and disabled
custom all-reduce. Inference continues using NCCL.

---

## Checking P2P Capability Reporting

```bash
# Quick Python check — shows what the driver reports
python3 - <<'EOF'
import torch
for i in range(torch.cuda.device_count()):
    for j in range(torch.cuda.device_count()):
        if i != j:
            can = torch.cuda.can_device_access_peer(i, j)
            print(f"GPU{i} → GPU{j}: cudaDeviceCanAccessPeer = {can}")
EOF
```

On a patched-driver system you will see `True` for all pairs. This means the
driver reports P2P capability and the DMA staging path is available. It does
**not** mean direct BAR-mapped reads work — see section 2 above.

---

## Verifying DMA Copy Correctness

```python
import torch

# Enable peer access (uses DMA staging on Intel consumer platforms)
torch.cuda.set_device(0)

a = torch.randn(100_000, device='cuda:0')
b = a.to('cuda:1')          # cudaMemcpyPeer under the hood

# If NaN: RMForceP2PType=1 is set (forces BAR P2P, which corrupts data)
# If correct: DMA staging path is working properly
assert not b.isnan().any(), "NaN detected — check RMForceP2PType in modprobe config"
print(f"Max absolute error: {(b.cpu() - a.cpu()).abs().max().item():.2e}")
# Expected: < 1e-6  (FP32 rounding only)
```

---

## Summary Table

| Test                         | Mechanism                   | Result on Intel consumer        |
|---                           |---                          |---                              |
| `cudaDeviceCanAccessPeer`    | Driver capability query     | **true** (patched driver)       |
| `cudaMemcpyPeer`             | DMA staging via host RAM    | **WORKS** @ ~6 GB/s             |
| Direct BAR reads/writes      | PCIe TLP routing            | **FAILS** (NaN, no error)       |
| CUDA IPC handle sharing      | BAR-mapped cross-process    | **FAILS**                       |
| NCCL all-reduce              | Auto-selects SHM transport  | **WORKS**                       |
| vLLM TP=2 inference          | NCCL-delegated all-reduce   | **WORKS**                       |
