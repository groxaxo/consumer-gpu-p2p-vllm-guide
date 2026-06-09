# Running a 35B MoE model on two consumer RTX 3090s — what actually works (and what doesn't)

I've spent the last few weeks turning a pile of consumer NVIDIA GPUs on an
Intel Alder Lake desktop into a working multi-GPU inference server. The short
version: **it works, but not the way you'd expect**. I documented the whole
journey and open-sourced the tooling so you don't have to repeat my mistakes.

---

## The problem nobody talks about clearly

Everyone says "just enable P2P" and move on. What they don't say is that on
Intel consumer platforms (Alder Lake, Raptor Lake — basically every Z690/B660
board out there), the CPU root complex **physically cannot route PCIe P2P
traffic between different root ports**. It's a silicon-level decision, not a
BIOS setting, not something you can patch with `setpci`. The ACS registers read
back as `0xffff` no matter what you write.

What this means in practice: direct GPU-to-GPU memory access via BAR mapping
causes **silent data corruption** — your tensors come out full of NaN. There's
no error. The computation just gives you garbage.

But here's the thing — that doesn't actually stop you from running multi-GPU
inference.

---

## What actually works

NCCL and vLLM both handle this gracefully if you understand the fallback path:

**`cudaMemcpyPeer`** — the patched driver enables DMA staging through host RAM.
Throughput is around 6 GB/s per direction, which is plenty for NCCL all-reduce
operations between two GPUs.

**NCCL SHM transport** — when NCCL probes at startup and finds P2P unavailable,
it automatically selects shared-host-memory transport. No flags needed, no
config. You'll see this in the logs:
```
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
```

**vLLM TP=2** — vLLM runs `can_actually_p2p()` in separate processes at
startup, catches the IPC failure, and disables custom all-reduce automatically.
You'll see:
```
Custom allreduce is disabled because your platform lacks GPU P2P capability
or P2P test failed.
```

That's vLLM telling you everything's fine.

---

## The counterintuitive performance trick

Here's where it gets interesting: **explicitly disabling P2P is faster than
letting the system auto-detect it**.

When NCCL probes for P2P and finds it doesn't work, it falls back to SHM — but
it doesn't optimize as aggressively as it would if it knew from the start that
P2P wasn't an option. By setting `NCCL_P2P_DISABLE=1`, you're telling NCCL
"don't bother probing, go straight to SHM and optimize for it."

The results on `Qwen3.6-35B-A3B-FP8` (MoE) across two RTX 3090s:

| Workload | P2P auto-detect | P2P disabled | Gain |
|---|---:|---:|---:|
| Short prompts (64/64) | 197 tok/s | 231 tok/s | **+17%** |
| Medium prompts (512/256) | 203 tok/s | 227 tok/s | **+12%** |
| Concurrent ×4 (256/128) | 174 tok/s | 217 tok/s | **+24%** |
| Long sequences (1024/512) | 48 tok/s | 115 tok/s | **+142%** |

The long-sequence case is wild. Auto-detect showed TPOT variance of 46–105 ms
(bad jitter). P2P disabled: 42.7–43.2 ms, rock stable. The fallback path under
auto-detect struggles with larger KV-cache working sets, and it compounds with
sequence length.

Cold start is also 6 seconds faster because vLLM skips P2P cache generation.

So the two-line production config is:
```bash
NCCL_P2P_DISABLE=1
VLLM_SKIP_P2P_CHECK=1
```

---

## Ollama

Ollama uses the same `cudaMemcpyPeer` path under the hood (it doesn't use NCCL
directly). Getting it to use both GPUs is a simple systemd drop-in:

```ini
# /etc/systemd/system/ollama.service.d/gpu-pair.conf
[Service]
Environment="CUDA_DEVICE_ORDER=PCI_BUS_ID"
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="OLLAMA_NUM_GPU=2"
Environment="OLLAMA_KEEP_ALIVE=5m"
```

One caveat: Ollama and vLLM cannot run on the same GPUs simultaneously. Both
will try to claim VRAM and one of them will OOM. Pick one per session, or pin
them to different GPU pairs if you have more than two cards.

On a 5-GPU rig, I run the two 3090s (GPU0 + GPU1) for vLLM and keep the others
available for Ollama quick queries.

---

## The installer

I got tired of manually rebuilding after kernel updates, so I wrote an
automated installer. One command:

```bash
git clone https://github.com/groxaxo/consumer-gpu-p2p-vllm-guide.git
cd consumer-gpu-p2p-vllm-guide
python3 install.py
```

It handles:
- Cloning the patched NVIDIA driver from `aikitoria/open-gpu-kernel-modules`
  (`595.58.03-p2p` branch)
- Registering with DKMS so kernel upgrades auto-rebuild the patched modules
  (MOK-signed, Secure Boot compatible)
- Patching GRUB with the required `intel_iommu=on iommu=pt` args
- Writing `/etc/modprobe.d/nvidia.conf` with `RMForceP2PType=0`
- Setting up an apt pin + `apt-mark hold` so `apt upgrade` can't silently
  replace the patched driver with a stock one
- Creating `~/venvs/vllm` with the full CUDA 12.8 vLLM stack via `uv`
- Installing a `p2p-healthcheck` script you can run after any kernel update

The DKMS + apt lockdown piece took the most iteration. Without it, the next
`apt upgrade` that lands a new kernel leaves you with missing modules and a
broken setup. Now `sudo apt update && sudo apt upgrade` is just fine.

---

## Hardware notes

A few things I learned about PCIe slot assignment that aren't obvious:

On Z690/B660 boards, the PCH can only deliver about 16 GB/s total across its
DMI link. If you have multiple GPUs on PCH root ports, they share that
bandwidth. Some slots physically look like x16 but the card negotiates Gen1 x1
because the PCH ran out of lanes. My two RTX 3060s ended up at Gen1 x1 — that's
~0.8 GB/s per GPU, useless for inference.

The 3090s came up at Gen4 x8 (CPU PEG) and Gen4 x4 (PCH) — about 6.6 GB/s
each, which is workable. The bandwidth benchmark script in the repo (`p2p_bandwidth_bench.py`)
will tell you exactly what you have before you commit to a GPU pair.

Best pair for TP=2: one GPU on a CPU PEG slot, one on a PCH slot, both at
Gen4. The DMI link carries the all-reduce traffic but Gen4 x4 is fast enough
that it doesn't bottleneck you meaningfully.

---

## Repo

**https://github.com/groxaxo/consumer-gpu-p2p-vllm-guide**

It includes:
- The automated installer (`install.py`)
- 8 detailed docs covering each part of the setup
- The production launch script with boot arg gates
- NCCL all-reduce test, P2P bandwidth benchmark (CUDA), post-reboot validation

Happy to answer questions. If you're on AMD Threadripper or EPYC and wondering
if any of this applies to you — it doesn't, your root complex handles P2P
natively. This is specifically the Intel consumer platform problem.

---

*Tested on: i7-12700KF, 5× NVIDIA RTX GPUs, Ubuntu 22.04, vLLM 0.19.0,
NCCL + CUDA 12.8*
