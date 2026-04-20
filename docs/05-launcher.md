# 5. Production Launcher Script

The canonical launcher is `scripts/manage_vllm_safe_tp2.sh`. It enforces boot
args, manages PID files, handles stale processes, and refuses to adopt
unmanaged vLLM instances.

## Design Principles

1. **Hard-fail on missing prerequisites** — if the required boot args aren't
   present, the script exits with a clear error rather than launching into a
   broken state.
2. **Never adopt unmanaged processes** — if a vLLM process is already listening
   on the configured port but wasn't started by this script, the start command
   refuses. This prevents config bypass by background processes.
3. **Single source of truth** — all NCCL, transport, and model flags are set
   here. You don't need to pass env vars manually.
4. **Production transport defaults** — `NCCL_P2P_DISABLE=1` and
   `VLLM_SKIP_P2P_CHECK=1` are on by default. These skip P2P probing entirely
   and give 10–15% better TPOT than letting NCCL probe and fall back. Set
   either to `0` to validate the auto-detect path. `--disable-custom-all-reduce`
   is never passed — vLLM disables it automatically when P2P is unavailable.

## Usage

```bash
# Start with default model (Qwen/Qwen3.5-9B)
bash scripts/manage_vllm_safe_tp2.sh start

# Start with a different model
bash scripts/manage_vllm_safe_tp2.sh start Qwen/Qwen3.6-35B-A3B-FP8

# Start with extra vLLM flags (appended after the model)
bash scripts/manage_vllm_safe_tp2.sh start Qwen/Qwen3.6-35B-A3B-FP8 --max-model-len 16384

# Other commands
bash scripts/manage_vllm_safe_tp2.sh stop
bash scripts/manage_vllm_safe_tp2.sh restart
bash scripts/manage_vllm_safe_tp2.sh status
bash scripts/manage_vllm_safe_tp2.sh health
```

## Environment Overrides

These can be exported before running the script to change its defaults:

```bash
VLLM_MODEL=Qwen/Qwen3.5-9B            # model to serve
VLLM_PORT=8000                         # port (default 8000)
VLLM_HOST=127.0.0.1                    # bind address
VLLM_MAX_MODEL_LEN=32768               # max context length
VLLM_GPU_MEMORY_UTILIZATION=0.92       # fraction of VRAM to allocate
CUDA_VISIBLE_DEVICES=0,1               # GPU pair to use
VLLM_VENV_PATH=~/venvs/vllm            # path to vLLM virtualenv
VLLM_LOG_DIR=~/logs                    # where vllm stdout/stderr is written
VLLM_RUN_DIR=~/.run                    # PID file directory
VLLM_STARTUP_TIMEOUT_SECONDS=180       # seconds to wait for /health to respond

# Feature toggles
VLLM_USE_QWEN_TOOLING_DEFAULTS=1       # add Qwen3 parser/tool flags (see below)
VLLM_USE_UNSLOTH_DEFAULTS=1            # add prefix caching + batching flags
VLLM_UNSLOTH_ARGS=""                   # override the built-in Unsloth defaults

# Transport / P2P
NCCL_P2P_DISABLE=1                     # 1 = skip P2P probe (default, faster)
                                        # 0 = let NCCL probe (for validation)
VLLM_SKIP_P2P_CHECK=1                  # 1 = skip vLLM P2P cache gen (default)
                                        # 0 = run vLLM's IPC test (for validation)

# MoE
VLLM_MOE_BACKEND=marlin                # force a specific MoE kernel backend
                                        # omit to use auto-detection (see below)
```

## Startup Flow

```
start command
  ├─ remove_pid_file_if_stale()          — clean up leftover PID from last crash
  ├─ already running?                    → exit 0 (idempotent)
  ├─ unmanaged vLLM on same port?        → REFUSE (prevents config bypass)
  ├─ require_p2p_boot_args()
  │   ├─ intel_iommu=on in /proc/cmdline → REFUSE if missing
  │   └─ iommu=pt in /proc/cmdline       → REFUSE if missing
  ├─ clear vLLM P2P cache files          — stale cache can cause 5s startup delay
  └─ launch vllm serve
      ├─ NCCL_P2P_DISABLE=1   (default; set to 0 to validate auto-detect)
      ├─ NCCL_IB_DISABLE=1
      ├─ NCCL_SHM_DISABLE=0
      ├─ VLLM_SKIP_P2P_CHECK=1 (default; set to 0 to validate)
      ├─ --enforce-eager        (prevents OOM on CUDA graph capture)
      └─ --tensor-parallel-size 2
```

## Qwen-Specific Flags

When the model is `Qwen/Qwen3.5-9B` (the default) and
`VLLM_USE_QWEN_TOOLING_DEFAULTS=1` (the default), the launcher adds:

```
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--language-model-only
```

**What these do:**

- `--reasoning-parser qwen3` — enables parsing of Qwen3's chain-of-thought
  `<think>...</think>` blocks from the output.
- `--enable-auto-tool-choice` — allows the model to call tools based on user
  intent, without requiring the client to explicitly pass `tool_choice`.
- `--tool-call-parser qwen3_coder` — uses Qwen3's specific tool call output
  format (JSON-in-special-tokens) rather than the generic OpenAI format.
- `--language-model-only` — Qwen3.5-9B is a vision-language checkpoint that
  includes vision encoder weights. This flag tells vLLM to load only the
  language model weights, skipping the vision encoder. Saves memory and
  eliminates image-token processing overhead when you're only doing text.

Set `VLLM_USE_QWEN_TOOLING_DEFAULTS=0` to disable all of these even for the
default model (e.g. when using the model through a client that handles tool
parsing itself).

## Unsloth Performance Defaults

When `VLLM_USE_UNSLOTH_DEFAULTS=1` (the default), the launcher adds vLLM
flags optimized for throughput:

- Prefix caching (`--enable-prefix-caching`) — reuses KV cache across requests
  that share a common system prompt, reducing TTFT for multi-turn conversations.
- Chunked prefill tuning — better batching of prefill and decode phases.

Set `VLLM_USE_UNSLOTH_DEFAULTS=0` to disable. Set `VLLM_UNSLOTH_ARGS="..."` to
completely replace the built-in Unsloth defaults with your own flags.

## MoE Backend Auto-Detection

vLLM supports multiple backends for Mixture-of-Experts routing kernels. The
launcher adds `--moe-backend marlin` automatically only when the model looks
like a **quantized MoE** model — specifically, when the model name contains
both:

1. An active-parameter suffix: `-A3B`, `-A22B`, `-A47B`, etc. (the number of
   active params in the MoE sparse forward pass)
2. A quantization token: `FP8`, `GPTQ`, `AWQ`, `INT4`, `INT8`, `W4`, `W8`, `BNB`

Examples:
- `Qwen3.6-35B-A3B-FP8` → quantized MoE → `--moe-backend marlin` added
- `Qwen3.5-9B` → dense model → no MoE backend flag
- `Qwen3.6-35B-A3B` → MoE but not quantized → no flag (uses vLLM default)

Marlin is a high-performance quantized matmul kernel tuned for A100/3090-class
GPUs. On unquantized (full-precision) MoE models, Marlin is not needed and the
flag is omitted.

Set `VLLM_MOE_BACKEND=<value>` to force a specific backend regardless of
auto-detection, or leave it unset to let the launcher decide.
