# 4. vLLM Setup & Configuration

## Installation

```bash
python3 -m venv ~/venvs/vllm
source ~/venvs/vllm/bin/activate
pip install vllm
```

## Running with TP=2

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

## Environment Variables Explained

| Variable | Value | Why |
|---|---|---|
| `NCCL_IB_DISABLE=1` | No InfiniBand | Consumer hardware has no IB |
| `NCCL_P2P_DISABLE=0` | Let NCCL probe | Auto-selects SHM when BAR P2P fails |
| `NCCL_SHM_DISABLE=0` | Enable SHM | The actual transport that works |
| `VLLM_SKIP_P2P_CHECK=0` | Don't skip | vLLM correctly detects and handles failure |

## Why `--enforce-eager` Is Hardcoded

For models that fill most of VRAM (e.g., Qwen3.6-35B FP8 at ~17.5 GiB per GPU
on 24 GiB RTX 3090s at 0.92 utilization), CUDA graph warmup requires additional
memory for profiling. With only ~700 MiB free per GPU, the graph capture OOMs.

`--enforce-eager` disables CUDA graphs entirely, trading ~5-10% throughput for
stability. On smaller models that leave more VRAM free, you can omit this flag.

## What Happens at Startup

1. vLLM detects 2 GPUs with `CUDA_VISIBLE_DEVICES=0,1`
2. It runs `can_actually_p2p()` — spawns two processes, tests CUDA IPC
3. IPC test fails (Intel root complex limitation)
4. vLLM disables custom all-reduce, delegates to NCCL
5. NCCL probes transport: direct P2P fails, selects SHM
6. Model loads: ~17.5 GiB per GPU
7. Server starts, correct inference begins

## Model Selection Considerations

For TP=2 on 24 GiB GPUs:
- Model must fit in `2 × 24 × utilization` GiB
- At 0.92 utilization: ~44 GiB total
- Qwen3.6-35B-A3B-FP8: ~35 GiB (fits comfortably)
- Larger models may require TP=4 or lower utilization

## Verification

```bash
# Check health
curl http://127.0.0.1:1234/v1/models

# Smoke test
curl http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32
  }'
```

## Monitoring

```bash
# Watch GPU memory
watch -n1 nvidia-smi

# Check vLLM log for transport info
grep -E "P2P|allreduce|NCCL|SHM" /path/to/vllm.log
```
