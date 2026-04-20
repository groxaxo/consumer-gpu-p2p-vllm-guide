# Guide: Running multi-GPU inference on consumer Intel hardware (Alder Lake/Raptor Lake) with Ollama and vLLM — including the thing that nobody explains clearly

I spent the last few weeks banging my head against "just enable P2P" advice that
doesn't actually apply to consumer Intel platforms. Eventually got it all working
— Ollama and vLLM both running tensor-parallel across two RTX 3090s on an
i7-12700KF — and documented the whole journey so you don't have to repeat it.

**Repo:** https://github.com/groxaxo/consumer-gpu-p2p-vllm-guide

---

## The thing nobody explains

On Intel consumer platforms (Alder Lake, Raptor Lake — basically any Z690/B660/Z790
board), the CPU root complex **physically cannot route PCIe peer-to-peer traffic
between different root ports**. Not a BIOS setting. Not a Linux config. A
silicon-level decision by Intel for consumer-grade root complexes.

What this means:

- Direct GPU-to-GPU memory via PCIe BAR windows: **silent data corruption** (NaN tensors, no error)
- `cudaDeviceCanAccessPeer()` still returns **true** (the driver lies)
- ACS registers read back `0xffff` regardless of what you write with `setpci`
- CUDA IPC (cross-process BAR mapping): **fails**

But here's what nobody says in the same breath: **this doesn't stop you from
doing multi-GPU inference**.

---

## What actually works

**`cudaMemcpyPeer`** with the patched NVIDIA driver: DMA staging through host RAM
at ~6 GB/s per direction. Not PCIe P2P in the traditional sense, but enough for
NCCL all-reduce.

**NCCL SHM transport**: NCCL probes at startup, finds P2P broken, and
automatically switches to shared-host-memory transport. No flags needed. You'll
see this in the logs:

```
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
```

**vLLM TP=2**: vLLM's `can_actually_p2p()` function spawns separate processes,
catches the IPC failure, and disables custom all-reduce automatically. You'll see:

```
Custom allreduce is disabled because your platform lacks GPU P2P capability
or P2P test failed.
```

That's the success message. Both adapters detected the limitation and handled it.

**Ollama multi-GPU**: Ollama uses the same `cudaMemcpyPeer` DMA path. Simple
systemd drop-in:

```ini
# /etc/systemd/system/ollama.service.d/gpu-pair.conf
[Service]
Environment="CUDA_DEVICE_ORDER=PCI_BUS_ID"
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="OLLAMA_NUM_GPU=2"
Environment="OLLAMA_KEEP_ALIVE=5m"
```

---

## The counterintuitive performance trick

Explicitly telling NCCL that P2P doesn't exist is faster than letting it
auto-detect. By setting `NCCL_P2P_DISABLE=1`, you skip the probe entirely —
NCCL goes straight to SHM and optimizes the path more aggressively than it
would if it had to fall back.

Benchmarks on **Qwen3.6-35B-A3B-FP8 (MoE)** on **2× RTX 3090** with vLLM 0.19.0:

| Workload | Auto-detect | P2P Disabled | Gain |
|---|---:|---:|---:|
| Short prompts (64 in/out) | 197 tok/s | **231 tok/s** | +17% |
| Medium prompts (512/256) | 203 tok/s | **227 tok/s** | +12% |
| Concurrent ×4 (256/128) | 174 tok/s | **217 tok/s** | +24% |
| Long sequences (1024/512) | 48 tok/s | **115 tok/s** | **+142%** |

Long sequences are wild. Auto-detect TPOT variance: 46–105 ms (very jittery).
P2P disabled TPOT: 42.7–43.2 ms (rock stable). The jitter comes from NCCL's
fallback path struggling as the KV-cache grows.

Cold start is also 6 seconds faster because vLLM skips P2P cache generation
(`VLLM_SKIP_P2P_CHECK=1`).

**Production config** — set these and forget them:

```bash
NCCL_P2P_DISABLE=1        # skip probe, use SHM directly
VLLM_SKIP_P2P_CHECK=1     # skip vLLM P2P cache gen, saves 5s startup
```

---

## PCIe slot assignment matters a lot

On Z690/B660 boards the PCH shares a DMI 3.0 x4 link (~16 GB/s total) across
all PCH-attached slots. If you have multiple GPUs on PCH ports, they compete for
that bandwidth. Some PCH slots physically look like x16 but the card negotiates
Gen1 x1 because the PCH ran out of lanes.

My two RTX 3060s came up at Gen1 x1 — **~0.8 GB/s**. Completely useless for
inference. The 3090s on the CPU PEG and PCH Gen4 slots came up at ~6.6 GB/s.

The repo includes a bandwidth benchmark (`scripts/p2p_bandwidth_bench.py`) that
compiles a CUDA test, runs all GPU pairs, and spits out a full system report.
Run it before picking your TP=2 pair.

---

## The one-command installer

Getting this setup to survive kernel upgrades was the hardest part. Without DKMS,
the next `apt upgrade` that lands a new kernel leaves you with missing modules and
a broken inference server.

```bash
git clone https://github.com/groxaxo/consumer-gpu-p2p-vllm-guide.git
cd consumer-gpu-p2p-vllm-guide
python3 install.py
```

The installer (with a full-screen animated UI) handles:

- Cloning `aikitoria/open-gpu-kernel-modules` (`595.58.03-p2p` branch)
- Registering with DKMS — patched modules rebuild automatically on kernel upgrade, signed with your MOK key (Secure Boot compatible)
- Patching GRUB with `intel_iommu=on iommu=pt` + modprobe config
- apt pin (`Pin-Priority: -1`) + `apt-mark hold` so stock `nvidia-driver-*` packages can't replace the patched driver
- Full CUDA 12.8 vLLM stack via `uv`
- `p2p-healthcheck` script for post-reboot validation

After install: `sudo apt update && sudo apt upgrade` is safe to run any time.

---

## Things this doesn't apply to

If you're on **AMD Threadripper, EPYC, or Intel Xeon**, your root complex handles
P2P natively. The SHM fallback path and patched driver are specific to Intel
consumer platforms.

If you have a **PLX/PEX PCIe switch** or **NVLink bridge** (the RTX 3090
supports NVLink in pairs), you'd have true BAR P2P and could skip all of this.
True BAR P2P would further reduce per-token latency by eliminating the host RAM
bounce entirely.

---

## TL;DR

- Intel consumer root complexes can't route BAR P2P. This is fine.
- NCCL and vLLM both auto-detect and use SHM instead. No config needed.
- **Disabling P2P probing is 10–15% faster** than auto-detect (up to +142% on long sequences).
- The patched NVIDIA driver (`595.58.03-p2p`) is required to enable `cudaMemcpyPeer`.
- One command installs + locks everything down so it survives kernel upgrades.

Repo + docs + scripts: https://github.com/groxaxo/consumer-gpu-p2p-vllm-guide

Happy to answer questions. If you're hitting the "works on one GPU, silent NaN on two" problem — that's this.
