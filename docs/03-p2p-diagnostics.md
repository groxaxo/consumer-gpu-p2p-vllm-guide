# 3. P2P Transport Diagnostics

## Understanding the Three P2P Paths

There are three distinct "P2P" mechanisms in CUDA, and they behave very
differently on consumer Intel platforms:

### 1. `cudaMemcpyPeer` (DMA Staging)

```
GPU0 → Host RAM → GPU1
```

The driver stages data through system RAM. This is NOT true P2P but it works
on any multi-GPU system. Performance is ~6 GB/s (limited by PCIe + DDR
bandwidth).

**Status: WORKS**

### 2. Direct BAR-Mapped Access

```
GPU0 → PCIe BAR window → GPU1 VRAM
```

One GPU maps a region of another GPU's VRAM into its own address space via
PCIe BAR windows. Reads/writes go directly over the PCIe bus without host
involvement.

**Status: FAILS on Intel consumer platforms** — the root complex cannot
route these TLPs between different root ports. Data comes back as NaN.

### 3. CUDA IPC (Inter-Process Communication)

```
Process A on GPU0 → IPC handle → Process B on GPU1
```

Uses `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` to share GPU memory
between processes. Internally uses BAR-mapped P2P.

**Status: FAILS** — same root complex limitation.

## Diagnostic Commands

### Check P2P Capability

```bash
CUDA_VISIBLE_DEVICES=0,1 /path/to/simpleP2P
```

Output on this system:
```
GPU0 CAN access GPU1 via peer access
GPU1 CAN access GPU0 via peer access
```

`cudaDeviceCanAccessPeer` returns true — but this only means the driver
reports capability, NOT that BAR-mapped reads work.

### Check Actual Data Transfer

```python
import torch
# DMA staging path (works)
a = torch.randn(1000, device='cuda:0')
b = a.to('cuda:1')  # Uses cudaMemcpyPeer internally
print(b.isnan().sum())  # 0 — correct

# Direct access path (fails on Intel consumer)
a = torch.randn(1000, device='cuda:0')
# If you could do direct BAR reads, they'd return NaN
```

### NCCL Transport Detection

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM \
  python test_nccl_tp2.py
```

Key output:
```
Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 1
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
```

`isAllDirectP2p 0` confirms NCCL detected that direct P2P doesn't work, and
it fell back to SHM transport.

### vLLM P2P Detection

vLLM uses a multi-process IPC test in `vllm/distributed/communication_op/
all_reduce_utils.py`. It spawns two processes, one on each GPU, and tests
CUDA IPC handle sharing. When this fails:

```
Custom allreduce is disabled because your platform lacks GPU P2P capability
or P2P test failed.
```

vLLM then falls back to NCCL for all-reduce operations.

### Bandwidth Test

```bash
CUDA_VISIBLE_DEVICES=0,1 /path/to/p2pBandwidthLatencyTest
```

This will show:
- Bidirectional BAR P2P: fails or shows very low bandwidth
- Unidirectional BAR P2P: fails or shows NaN
- Host-staged copy: ~6 GB/s (correct)

## Why simpleP2P "Passes" But P2P Doesn't Work

The CUDA `simpleP2P` sample only tests:
1. `cudaDeviceCanAccessPeer` — returns true (driver capability)
2. `cudaMemcpyPeer` — works (DMA staging)

It does NOT test direct BAR-mapped reads. The sample's "P2P access enabled"
message is misleading on platforms where only DMA staging works.

## Summary Table

| Test | Mechanism | Result |
|---|---|---|
| `cudaDeviceCanAccessPeer` | Driver capability query | TRUE |
| `cudaMemcpyPeer` | DMA staging via host RAM | WORKS @ ~6 GB/s |
| Direct BAR reads | PCIe TLP routing | FAILS (NaN) |
| CUDA IPC | BAR-mapped cross-process | FAILS |
| NCCL all-reduce | Auto-selects SHM | WORKS |
| vLLM inference | Auto-disables custom all-reduce | WORKS |
