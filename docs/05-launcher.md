# 5. Production Launcher Script

The canonical launcher is `manage_vllm_safe_tp2.sh`. It enforces boot args,
manages PID files, handles stale processes, and refuses to adopt unmanaged
vLLM instances.

## Design Principles

1. **Hard-fail on missing prerequisites** — no silent degradation
2. **Never adopt unmanaged processes** — prevents config bypass
3. **Single source of truth** — all other scripts delegate here
4. **No stale fallback flags** — `NCCL_P2P_DISABLE=1`, `VLLM_SKIP_P2P_CHECK=1`,
   and `--disable-custom-all-reduce` are never set

## Usage

```bash
# Start with default model
manage_vllm_safe_tp2.sh start

# Start with a different model
manage_vllm_safe_tp2.sh start meta-llama/Meta-Llama-3.1-8B-Instruct

# Start with extra args
manage_vllm_safe_tp2.sh start Qwen/Qwen3.6-35B-A3B-FP8 --max-model-len 16384

# Stop
manage_vllm_safe_tp2.sh stop

# Restart
manage_vllm_safe_tp2.sh restart

# Check status
manage_vllm_safe_tp2.sh status

# Health check
manage_vllm_safe_tp2.sh health
```

## Environment Overrides

```bash
VLLM_PORT=1235
VLLM_HOST=0.0.0.0
VLLM_MAX_MODEL_LEN=32768
VLLM_GPU_MEMORY_UTILIZATION=0.92
CUDA_VISIBLE_DEVICES=0,1
VLLM_VENV_PATH=/path/to/venv
VLLM_LOG_DIR=/home/user/logs
VLLM_STARTUP_TIMEOUT_SECONDS=180
```

## Startup Flow

```
start command
  ├─ remove_pid_file_if_stale()
  ├─ check if already running → exit 0
  ├─ check if unmanaged vLLM on same port → REFUSE
  ├─ require_p2p_boot_args()
  │   ├─ check intel_iommu=on in /proc/cmdline → REFUSE if missing
  │   └─ check iommu=pt in /proc/cmdline → REFUSE if missing
  ├─ clear vLLM P2P cache files
  └─ launch vllm serve with hardcoded env vars
      ├─ NCCL_P2P_DISABLE=0
      ├─ NCCL_SHM_DISABLE=0
      ├─ VLLM_SKIP_P2P_CHECK=0
      ├─ --enforce-eager
      └─ --tensor-parallel-size 2
```

## Qwen-Specific Flags

When the model matches `Qwen/Qwen3.6-35B-A3B-FP8` and
`VLLM_USE_QWEN_TOOLING_DEFAULTS=1` (default), the launcher adds:

```
--moe-backend marlin
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

For other models, these are omitted. Set `VLLM_USE_QWEN_TOOLING_DEFAULTS=0`
to disable even for the Qwen model.
