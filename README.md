# Consumer GPU P2P & vLLM Tensor-Parallel Guide

Running vLLM with tensor parallelism (TP=2) across consumer NVIDIA GPUs on an
Intel desktop platform. This guide documents the full journey: patched drivers,
P2P expectations vs reality, what actually works, and production-ready scripts.

## TL;DR

**It works — but not how you'd expect.**

On Intel consumer platforms (Alder Lake, Raptor Lake, etc.), the CPU root
complex **cannot** route BAR-mapped PCIe peer-to-peer (P2P) TLPs between
different root ports. This means:

- Direct GPU-to-GPU memory access via PCIe BAR windows: **FAILS** (data corruption)
- `cudaMemcpyPeer` (DMA staging through host RAM): **WORKS** at ~6 GB/s
- NCCL all-reduce via SHM (shared host memory): **WORKS** (auto-detected)
- vLLM TP=2 inference: **WORKS** (auto-disables custom all-reduce, uses NCCL)

You do **not** need working BAR P2P for multi-GPU inference. NCCL and vLLM
figure it out automatically. The patched driver is still required for the CUDA
`cudaDeviceCanAccessPeer`/`cudaMemcpyPeer` path to function.

## Hardware

| Component | Details |
|---|---|
| CPU | Intel Core i7-12700KF (Alder Lake, 12th gen) |
| GPUs | 2x RTX 3090 (24 GiB) + 2x RTX 3060 (12 GiB) + 1x RTX 3090 (24 GiB) |
| PCIe topology | All 5 GPUs on separate root ports, no PLX switch |
| OS | Ubuntu 22.04, kernel 6.x |
| Driver | NVIDIA 595.58.03 (aikitoria/open-gpu-kernel-modules `595.58.03-p2p` branch) |

### PCIe Topology

```
CPU PEG lanes (direct to CPU):
  00:01.0 → [01] EMPTY SLOT (x16 electrical)
  00:01.1 → [02] GPU0: RTX 3090  (x16 electrical, x8 negotiated)

PCH lanes (cross DMI bridge):
  00:1b.4 → [04] GPU1: RTX 3090  (x4)
  00:1c.0 → [05] GPU2: RTX 3060  (x1)
  00:1c.1 → [06] GPU3: RTX 3060  (x1)
  00:1c.4 → [08] GPU4: RTX 3090  (x4)
```

## Table of Contents

1. [Prerequisites & Boot Configuration](docs/01-boot-config.md)
2. [Patched NVIDIA Driver](docs/02-patched-driver.md)
3. [P2P Transport Diagnostics](docs/03-p2p-diagnostics.md)
4. [vLLM Setup & Configuration](docs/04-vllm-setup.md)
5. [Production Launcher Script](docs/05-launcher.md)
6. [Ollama Multi-GPU](docs/06-ollama.md)
7. [Troubleshooting](docs/07-troubleshooting.md)

## Quick Start

### 1. Install the patched driver

```bash
git clone -b 595.58.03-p2p https://github.com/aikitoria/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules
make -j$(nproc) modules
sudo make modules_install
sudo depmod -a
```

### 2. Configure boot args

Edit `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX_DEFAULT="... intel_iommu=on iommu=pt pci=noaer pcie_aspm=off"
```

```bash
sudo update-grub
sudo reboot
```

### 3. Set driver registry key

Create `/etc/modprobe.d/nvidia.conf`:

```
options nvidia NVreg_RegistryDwords="RMForceP2PType=0"
```

### 4. Install vLLM

```bash
python -m venv ~/venvs/vllm
source ~/venvs/vllm/bin/activate
pip install vllm
```

### 5. Launch with TP=2

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  NCCL_IB_DISABLE=1 \
  NCCL_P2P_DISABLE=0 \
  NCCL_SHM_DISABLE=0 \
  VLLM_SKIP_P2P_CHECK=0 \
  vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --enforce-eager \
    --max-model-len 32768
```

## What NOT to Set

These are stale workarounds from before the hardware limitation was understood.
**Do not use them** — they break auto-detection:

```bash
# ❌ WRONG — prevents NCCL from probing transport capabilities
NCCL_P2P_DISABLE=1

# ❌ WRONG — hides real P2P failures instead of letting vLLM handle them
VLLM_SKIP_P2P_CHECK=1

# ❌ WRONG — vLLM disables custom all-reduce automatically when needed
--disable-custom-all-reduce
```

## Key Findings

### The Intel Root Complex Limitation

Consumer Intel platforms (Alder Lake, Raptor Lake, etc.) have a root complex that
cannot route BAR-mapped peer TLPs between different PCIe root ports. This is a
**silicon-level limitation**, not fixable in software.

What this means:
- `cudaDeviceCanAccessPeer()` returns **true** (driver reports capability)
- `cudaMemcpyPeer()` **works** (uses DMA staging through host RAM)
- Direct BAR-mapped reads/writes from one GPU to another: **data corruption (NaN)**
- CUDA IPC (cross-process BAR mapping): **fails**

### NCCL Handles It Automatically

NCCL probes transport capabilities at startup. When direct P2P fails, it
automatically falls back to SHM (shared host memory) transport:

```
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
```

No environment variable overrides needed.

### vLLM Handles It Automatically

vLLM's `can_actually_p2p()` function tests CUDA IPC in separate processes.
When it fails, vLLM disables custom all-reduce and delegates to NCCL:

```
Custom allreduce is disabled because your platform lacks GPU P2P capability
or P2P test failed.
```

No flags needed.

### ACS is Read-Only (and Irrelevant)

Access Control Services (ACS) registers on Intel consumer root complexes are
**read-only**. `setpci` writes succeed (exit 0) but the register reads back
unchanged (`0xffff`). This is moot — since the root complex can't route P2P
TLPs at all, ACS state doesn't matter.

## When Would True BAR P2P Work?

You'd need one of:
1. **AMD Threadripper / EPYC** — their root complexes support P2P routing
2. **Intel Xeon** (server platforms) — same
3. **PLX/PEX PCIe switch** — all GPUs behind the same switch can do P2P
4. **NVLink bridge** — directly connects GPUs (RTX 3090 supports this in pairs)

Two GPUs on the **same CPU PEG controller** (e.g., 00:01.0 and 00:01.1) might
theoretically support P2P, but this is untested and may still not work on
consumer platforms.

## Could True P2P Help Performance?

On this setup, NCCL SHM transport achieves ~6 GB/s between GPUs (DMA through
host RAM). True BAR P2P would be faster for small-message all-reduce operations
(lower latency), but for the large tensor shards in vLLM TP inference, the
throughput difference is modest. The bottleneck is typically compute, not
inter-GPU bandwidth.

## Scripts

| Script | Purpose |
|---|---|
| [`scripts/manage_vllm_safe_tp2.sh`](scripts/manage_vllm_safe_tp2.sh) | Canonical vLLM launcher with boot arg gates |
| [`scripts/require-gpu-pair.sh`](scripts/require-gpu-pair.sh) | Pre-flight check for TP=2 GPU pair |
| [`scripts/post-reboot-test.sh`](scripts/post-reboot-test.sh) | Full post-reboot validation (boot args + NCCL test) |
| [`scripts/test_nccl_tp2.py`](scripts/test_nccl_tp2.py) | Standalone NCCL all-reduce test |

## License

This guide and all scripts are provided under the MIT License. The patched
NVIDIA driver is subject to its own license (Dual MIT/GPL) — see
[aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules).
