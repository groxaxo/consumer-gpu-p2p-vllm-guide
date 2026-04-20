# 4. vLLM Setup & Configuration

## Installation

The installer (`python3 install.py`) handles this automatically. If you prefer
to install manually:

```bash
# Install uv — fast Python package manager
python3 -m pip install --user uv

# Create an isolated venv
uv venv ~/venvs/vllm

# Install PyTorch with CUDA 12.8 wheels first
# (cu128 = CUDA 12.8 — use this exact index URL to avoid CUDA 13.x wheels)
uv pip install --python ~/venvs/vllm/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 \
  torchvision==0.26.0+cu128 \
  torchaudio==2.11.0+cu128

# Install vLLM
uv pip install --python ~/venvs/vllm/bin/python vllm==0.21.0
```

> Do **not** use plain `pip install vllm` — it can pull in CUDA 13.x wheels
> that are incompatible with the rest of this stack.

Verify the runtime resolves to CUDA 12.x before proceeding:

```bash
~/venvs/vllm/bin/python -c \
  'import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)'
# Expected: 2.11.0+cu128  12.8  0.21.0
```

---

## Running with TP=2

Using the production launcher (recommended):

```bash
bash scripts/manage_vllm_safe_tp2.sh start
```

Or manually with full control over all flags:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  NCCL_IB_DISABLE=1 \
  NCCL_P2P_DISABLE=1 \
  NCCL_SHM_DISABLE=0 \
  VLLM_SKIP_P2P_CHECK=1 \
  ~/venvs/vllm/bin/vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --enforce-eager \
    --max-model-len 32768
```

---

## Environment Variables Explained

| Variable                   | Value | Why                                                           |
|---                         |---    |---                                                            |
| `NCCL_IB_DISABLE=1`        | 1     | No InfiniBand on consumer hardware; disable IB probing        |
| `NCCL_P2P_DISABLE=1`       | 1     | Skip P2P probe — 10–15% faster TPOT vs probe-and-fallback     |
| `NCCL_SHM_DISABLE=0`       | 0     | Keep SHM enabled — this is the transport that actually works  |
| `VLLM_SKIP_P2P_CHECK=1`    | 1     | Skip vLLM's IPC test — saves ~5 s at startup                  |

For **first-time validation**, set `NCCL_P2P_DISABLE=0` and
`VLLM_SKIP_P2P_CHECK=0` once. NCCL will probe P2P, fail, log
`via SHM/direct/direct`, and fall back. vLLM's `can_actually_p2p()` will
run, fail, and log "Custom allreduce is disabled". Both log lines confirm
the auto-detection pipeline is working. Then switch back to the production
values above for all subsequent runs.

---

## Why `--enforce-eager` Is Hardcoded

For models that fill most of VRAM (e.g. Qwen3.6-35B FP8 uses ~17.5 GiB per
GPU on 24 GiB RTX 3090s at 0.92 utilization), CUDA graph warmup requires
additional memory for profiling. With only ~700 MiB free per GPU, the graph
capture OOMs.

`--enforce-eager` disables CUDA graphs, trading ~5–10% throughput for
stability. On smaller models that leave more VRAM free, you can omit this flag.

---

## What Happens at Startup

**Validation config** (`NCCL_P2P_DISABLE=0`, `VLLM_SKIP_P2P_CHECK=0`):

1. vLLM detects 2 GPUs from `CUDA_VISIBLE_DEVICES=0,1`
2. Runs `can_actually_p2p()` — spawns two processes, tests CUDA IPC handles
3. IPC test fails (Intel root complex can't route BAR P2P)
4. vLLM logs "Custom allreduce is disabled" and delegates all-reduce to NCCL
5. NCCL probes transport: direct P2P fails; logs `via SHM/direct/direct`; selects SHM
6. Model loads: ~17.5 GiB per GPU
7. Server starts; correct inference begins

**Production config** (`NCCL_P2P_DISABLE=1`, `VLLM_SKIP_P2P_CHECK=1`):

Steps 2–3 and 5 are skipped entirely. vLLM trusts the env vars; NCCL goes
straight to SHM. Same end state, ~6 s faster startup, 10–15% lower per-token
latency overhead.

---

## Model Selection for TP=2 on 24 GiB GPUs

Available VRAM at 0.92 utilization: `2 × 24 × 0.92 ≈ 44 GiB` total.

| Model                        | Approx. size  | Fits at 0.92? |
|---                           |---            |---            |
| Qwen/Qwen3.5-9B              | ~9 GiB        | Yes (lots of headroom) |
| Qwen/Qwen3.6-35B-A3B-FP8    | ~35 GiB       | Yes (tight)   |
| Larger dense 70B models      | ~70+ GiB      | No — need TP=4 or lower util |

The MoE architecture of Qwen3.6-35B means only a subset of parameters are
active per token, which is why it fits despite its total parameter count.

---

## Verification

After launch, confirm the server is healthy:

```bash
# List loaded models
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool

# Smoke test
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "Hello, what model are you?"}],
    "max_tokens": 32
  }' | python3 -m json.tool
```

---

## Monitoring

```bash
# Watch GPU memory and utilization in real time
watch -n1 nvidia-smi

# Check vLLM's log for transport confirmation (launcher writes to ~/logs/)
grep -E "P2P|allreduce|NCCL|SHM" ~/logs/vllm.log | head -20

# Or if running vLLM directly in terminal, filter stdout:
# ... | grep -E "P2P|allreduce|NCCL|SHM"
```

---

## Version Notes

Benchmarks in the README were collected on vLLM 0.19.0. The installer ships
0.21.0. Results should be equivalent or better on 0.21.0; the NCCL transport
path is unchanged between these versions.
