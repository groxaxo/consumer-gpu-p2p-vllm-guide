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
8. [Lockdown: Surviving `apt upgrade`](docs/08-lockdown.md)

## Quick Start

```bash
git clone https://github.com/groxaxo/consumer-gpu-p2p-vllm-guide.git
cd consumer-gpu-p2p-vllm-guide
python3 install.py
```

That is the only command you need. The installer will:

1. Bootstrap `asciimatics` automatically if it is not installed
2. Show you a plan of what will change and ask for confirmation
3. Launch a full-screen animated display while the installation runs in the background
4. Install OS prerequisites (`apt`)
5. Clone the patched NVIDIA P2P driver source
6. **Register the driver with DKMS** so kernel upgrades auto-rebuild it
7. Patch GRUB boot args and write `/etc/modprobe.d/nvidia.conf`
8. Lock down apt (`apt-mark hold` + `/etc/apt/preferences.d/00-nvidia-p2p-pin`) so userspace `nvidia-*` / `libnccl*` packages can't replace the `.run`-installed driver
9. Stash the `.run` installer at `/opt/nvidia-p2p/` and install `/usr/local/sbin/p2p-healthcheck`
10. Create `~/venvs/vllm` and install a CUDA 12.8 vLLM stack via `uv`
11. Print the full install log on completion (or on error)

### Surviving `apt upgrade`

Out-of-tree NVIDIA drivers normally die the next time you run
`sudo apt upgrade`, because a new kernel lands without matching modules. This
installer fixes that by:

- **DKMS** rebuilds the patched 595.58.03 modules (signed with your MOK key,
  Secure Boot–compatible) every time apt installs a new kernel — *before* you
  reboot.
- An **apt preferences pin** with `Pin-Priority: -1` blocks any new
  `nvidia-driver-*` / `libnvidia-compute-*` / `linux-modules-nvidia-*` package
  from landing as a transitive dependency.
- **apt-mark hold** freezes the currently-installed userspace `nvidia-*` /
  `libnccl*` packages so a new CUDA-13 NCCL or stock NVIDIA driver can't
  replace them.

Result: `sudo apt update && sudo apt upgrade` is safe to run any time. Verify
with `sudo p2p-healthcheck`. See [docs/08-lockdown.md](docs/08-lockdown.md) for
full details and the recovery procedure.

**Flags:**

| Flag | Effect |
|---|---|
| `--dry-run` | Show what would happen — make no changes |
| `--yes` | Skip the confirmation prompt |
| `--skip-driver` | Skip driver clone + DKMS build |
| `--skip-grub` | Skip GRUB / modprobe changes |
| `--skip-vllm` | Skip venv + vLLM install |
| `--skip-lockdown` | Skip apt holds, apt pin, healthcheck, `.run` stash |
| `--driver-dir PATH` | Override driver checkout location |
| `--venv-dir PATH` | Override venv location |

After install, **reboot** and validate:

```bash
bash scripts/post-reboot-test.sh
```

Then launch vLLM:

```bash
bash scripts/manage_vllm_safe_tp2.sh start
```

### Launch with TP=2 manually

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  NCCL_IB_DISABLE=1 \
  NCCL_P2P_DISABLE=1 \
  NCCL_SHM_DISABLE=0 \
  VLLM_SKIP_P2P_CHECK=1 \
  vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --enforce-eager \
    --max-model-len 32768
```

> **Manual steps** (what the installer does under the hood) are documented in
> [`docs/`](docs/) for reference.

## What to Set (and Why)

### For production (recommended — fastest):

```bash
NCCL_P2P_DISABLE=1        # Skip P2P probe — 10-15% faster TPOT
VLLM_SKIP_P2P_CHECK=1     # Skip vLLM P2P cache generation — saves 5s startup
```

Benchmarks show explicitly disabling P2P gives 10–15% better throughput because
NCCL optimizes the SHM transport path when it knows P2P isn't available.

### For initial validation (first-time setup):

```bash
NCCL_P2P_DISABLE=0        # Let NCCL probe — confirms fallback works
VLLM_SKIP_P2P_CHECK=0     # Let vLLM test P2P — confirms auto-detection
```

Use this once to verify everything works, then switch to the production config.

### Do not use:

```bash
# ❌ WRONG — vLLM disables custom all-reduce automatically when needed
--disable-custom-all-reduce
```

## Benchmarks

Tested with `Qwen/Qwen3.6-35B-A3B-FP8` (MoE, FP8) on 2x RTX 3090 via vLLM 0.19.0.
Two configurations compared:

- **P2P Auto-detect**: `NCCL_P2P_DISABLE=0` — NCCL probes P2P, fails, falls back to SHM
- **P2P Disabled**: `NCCL_P2P_DISABLE=1` — NCCL skips P2P probe, uses SHM directly

### Short Prompts (64 in / 64 out, sequential)

| Metric | P2P Auto-detect | P2P Disabled | Delta |
|---|---:|---:|---:|
| TTFT (median) | 75.2 ms | 68.8 ms | **-8.5%** |
| TPOT (median) | 47.7 ms | 42.6 ms | **-10.7%** |
| Throughput | 197 tok/s | 231 tok/s | **+17%** |

### Medium Prompts (512 in / 256 out, sequential)

| Metric | P2P Auto-detect | P2P Disabled | Delta |
|---|---:|---:|---:|
| TTFT (median) | 76.2 ms | 69.5 ms | **-8.8%** |
| TPOT (median) | 46.8 ms | 43.0 ms | **-8.1%** |
| Throughput | 203 tok/s | 227 tok/s | **+12%** |

### Concurrent (256 in / 128 out, 4 concurrent)

| Metric | P2P Auto-detect | P2P Disabled | Delta |
|---|---:|---:|---:|
| TTFT (median) | 194 ms | 136 ms | **-30%** |
| TPOT (median) | 52.8 ms | 45.3 ms | **-14%** |
| Throughput | 174 tok/s | 217 tok/s | **+24%** |

### Long Sequences (1024 in / 512 out, sequential)

| Metric | P2P Auto-detect | P2P Disabled | Delta |
|---|---:|---:|---:|
| TTFT (median) | 204 ms | 103 ms | **-50%** |
| TPOT (median) | 58.1 ms | 42.9 ms | **-26%** |
| Throughput | 48 tok/s | 115 tok/s | **+142%** |

> **Note**: The long-sequence P2P auto-detect test showed high variance (TPOT
> 46–105 ms) while P2P disabled was rock-stable (42.7–43.2 ms). The variance
> likely comes from NCCL's fallback path struggling with larger KV-cache
> working sets.

### Startup Time

| Config | Cold start to ready |
|---|---|
| P2P Auto-detect | ~92 s (includes ~5 s P2P cache generation) |
| P2P Disabled | ~86 s |

### Takeaway

**Explicitly disabling P2P is 10–15% faster for typical workloads and up to
40%+ faster under concurrent/long-sequence loads.** When you know your hardware
can't do BAR P2P, tell NCCL upfront — it optimizes the SHM path more
aggressively and avoids per-token fallback overhead.

For production use, set `NCCL_P2P_DISABLE=1` and `VLLM_SKIP_P2P_CHECK=1`. For
initial validation, leave them at 0 to confirm auto-detection works, then switch.

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

## Could True BAR P2P Help Performance?

With NCCL P2P explicitly disabled (SHM-only transport), vLLM achieves
**~43 ms TPOT** (23 tok/s per GPU) with the Qwen3.6-35B-A3B MoE model.
That's ~231 tok/s aggregate throughput for sequential short prompts, and
~227 tok/s for medium prompts — on two consumer RTX 3090s.

True BAR P2P (e.g., behind a PLX switch or on Threadripper/EPYC) would:
- Reduce per-token latency by eliminating the host RAM bounce
- Enable vLLM's custom all-reduce kernel (lower-overhead than NCCL)
- Likely improve concurrent throughput significantly (less host bus contention)

On this Intel setup, the DMI 3.0 x4 link between CPU and PCH is the
bottleneck for SHM transport when both GPUs are on PCH root ports.
The two GPUs used for TP=2 here are GPU0 (CPU PEG) and GPU1 (PCH),
so data crosses the DMI bridge on every all-reduce step.

## Scripts

| Script | Purpose |
|---|---|
| [`install.py`](install.py) | Interactive autoinstaller for the full setup |
| [`scripts/manage_vllm_safe_tp2.sh`](scripts/manage_vllm_safe_tp2.sh) | Canonical vLLM launcher with boot arg gates |
| [`scripts/require-gpu-pair.sh`](scripts/require-gpu-pair.sh) | Pre-flight check for TP=2 GPU pair |
| [`scripts/post-reboot-test.sh`](scripts/post-reboot-test.sh) | Full post-reboot validation (boot args + NCCL test) |
| [`scripts/test_nccl_tp2.py`](scripts/test_nccl_tp2.py) | Standalone NCCL all-reduce test |
| [`scripts/p2p_bandwidth_bench.py`](scripts/p2p_bandwidth_bench.py) | **Full P2P + PCIe bandwidth benchmark** (compiles + runs CUDA benchmark, emits system report) |
| [`scripts/p2p_bandwidth_bench.cu`](scripts/p2p_bandwidth_bench.cu) | CUDA benchmark source (unidirectional, bidirectional, latency, all GPU pairs) |

## P2P Bandwidth Benchmark

Run a full diagnostic across all GPU pairs:

```bash
python3 scripts/p2p_bandwidth_bench.py
# optionally save results
python3 scripts/p2p_bandwidth_bench.py --save bench_results.txt
```

The benchmark measures:
- **Unidirectional bandwidth** at 1 / 16 / 64 / 256 MiB transfer sizes
- **Bidirectional bandwidth** (simultaneous both directions)
- **Round-trip latency** (4-byte ping, 200 rounds)
- **Host↔Device baseline** per GPU
- **P2P vs CPU-bounce comparison** for every pair where P2P is available

It also prints a full system context header: driver version, PCIe link state
(`lspci LnkSta`), `nvidia-smi topo`, and GPU inventory — so you can paste the
entire output as a single reproducible report.

### Why PCIe slot assignment matters

On Intel consumer platforms the DMI link (CPU ↔ PCH) and the number of CPU-attached
PEG lanes are both finite. A mixed 3090+3060 rig on Z690/B660 will typically end
up with some cards on PCH root ports running at **x1** even though the slot is
physically x16. Example from a real 5-GPU system:

| GPU | Card | PCIe link | P2P bandwidth |
|---|---|---|---|
| GPU0 | RTX 3090 | Gen4 x8 (CPU PEG) | ~6.6 GB/s |
| GPU1 | RTX 3090 | Gen4 x4 (PCH) | ~6.6 GB/s |
| GPU2 | RTX 3060 | **Gen1 x1 (PCH)** | **~0.8 GB/s** |
| GPU3 | RTX 3060 | **Gen1 x1 (PCH)** | **~0.8 GB/s** |
| GPU4 | RTX 3090 | Gen4 x4 (CPU PEG) | ~6.6 GB/s |

The 3060s physically support Gen3 x16 but the PCH slots were bandwidth-starved.
The benchmark `[Summary & Recommendations]` section flags pairs below threshold
and prints the relevant `lspci` command to investigate.

**Fix**: move GPUs to CPU PEG slots, or use a PCIe bifurcation riser to give
both cards at least x4 each from a single x16 slot.

## License

This guide and all scripts are provided under the MIT License. The patched
NVIDIA driver is subject to its own license (Dual MIT/GPL) — see
[aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules).
