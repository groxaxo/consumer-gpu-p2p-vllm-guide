#!/usr/bin/env bash
set -euo pipefail
#
# Canonical vLLM TP=2 launcher for consumer GPUs on Intel platforms.
#
# GATES:
#   - /proc/cmdline must contain intel_iommu=on iommu=pt
#   - Refuses to adopt unmanaged vLLM processes
#
# RUNTIME:
#   - NCCL_P2P_DISABLE=1 (default) — BAR P2P never works on Intel consumer
#     root complexes, and telling NCCL upfront is 10-15% faster than letting
#     it probe and fall back to SHM (see README benchmarks). Export
#     NCCL_P2P_DISABLE=0 once to validate the auto-detect path.
#   - VLLM_SKIP_P2P_CHECK=1 (default) — skips ~5s of P2P cache generation.
#     Export VLLM_SKIP_P2P_CHECK=0 once to validate vLLM's own detection.
#   - NCCL_SHM_DISABLE=0
#   - --enforce-eager hardcoded (24 GiB 3090s OOM during CUDA graph warmup)
#
# See docs/05-launcher.md for full documentation.

usage() {
  cat <<'EOF'
Usage:
  manage_vllm_safe_tp2.sh start [model] [extra vllm serve args...]
  manage_vllm_safe_tp2.sh stop [model]
  manage_vllm_safe_tp2.sh restart [model] [extra vllm serve args...]
  manage_vllm_safe_tp2.sh status [model]
  manage_vllm_safe_tp2.sh health [model]

Environment overrides:
  VLLM_VENV_PATH                  Python venv to activate
  VLLM_MODEL                      Default model id
  VLLM_PORT                       Default port (1234)
  VLLM_HOST                       Bind host (0.0.0.0)
  VLLM_MAX_MODEL_LEN              Default max model len (32768)
  VLLM_GPU_MEMORY_UTILIZATION     Default GPU memory utilization (0.92)
  VLLM_LOG_DIR                    Log directory
  VLLM_RUN_DIR                    PID directory
  VLLM_STARTUP_TIMEOUT_SECONDS    Startup wait timeout (180)
  VLLM_UNSLOTH_ARGS               Optional custom Unsloth-style args (space-separated)
  VLLM_USE_UNSLOTH_DEFAULTS       1 to add built-in Unsloth-style defaults (default 1)
  VLLM_MOE_BACKEND                Optional explicit MoE backend
  VLLM_USE_QWEN_TOOLING_DEFAULTS  1 to add qwen3 parser/tool flags (default 1)
  NCCL_P2P_DISABLE                Default 1 (SHM-only, fastest); 0 to let NCCL probe
  VLLM_SKIP_P2P_CHECK             Default 1 (skip vLLM P2P check); 0 to validate
EOF
}

ACTION="${1:-}"
if [[ -z "$ACTION" ]]; then
  usage
  exit 1
fi
shift || true

QWEN_DEFAULT_MODEL="Qwen/Qwen3.5-9B"
DEFAULT_MODEL="$QWEN_DEFAULT_MODEL"
MODEL="${VLLM_MODEL:-$DEFAULT_MODEL}"

if [[ "$ACTION" == "start" || "$ACTION" == "restart" || "$ACTION" == "status" || "$ACTION" == "health" || "$ACTION" == "stop" ]]; then
  if [[ $# -gt 0 && "$1" != --* ]]; then
    MODEL="$1"
    shift
  fi
fi

PORT="${VLLM_PORT:-1234}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VENV_PATH="${VLLM_VENV_PATH:-/home/op/venvs/vllm}"
LOG_DIR="${VLLM_LOG_DIR:-/home/op/logs}"
RUN_DIR="${VLLM_RUN_DIR:-/home/op/.openjaw/run}"
STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-180}"
USE_QWEN_DEFAULTS="${VLLM_USE_QWEN_TOOLING_DEFAULTS:-1}"
USE_UNSLOTH_DEFAULTS="${VLLM_USE_UNSLOTH_DEFAULTS:-1}"
UNSLOTH_ARGS="${VLLM_UNSLOTH_ARGS:-}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
VLLM_SKIP_P2P_CHECK="${VLLM_SKIP_P2P_CHECK:-1}"

MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/: ' '___' | tr -cd 'A-Za-z0-9._-')"
PID_FILE="${RUN_DIR}/${MODEL_SLUG}-tp2-safe-${PORT}.pid"
LOG_FILE="${LOG_DIR}/${MODEL_SLUG}-tp2-safe-${PORT}.log"
HEALTH_URL="http://127.0.0.1:${PORT}/v1/models"

# Returns true when the model name looks like a quantized MoE model.
# Heuristic: active-param suffix (-A<N>B, e.g. -A3B, -A22B) AND a
# quantization token (FP8, GPTQ, AWQ, INT4, INT8, W4, W8).
# Full-precision MoE models (no quant token) do NOT match.
_is_quantized_moe() {
  local m="${1,,}"
  [[ "$m" =~ -a[0-9]+b ]] && [[ "$m" =~ (fp8|gptq|awq|int4|int8|w4|w8|bnb) ]]
}

# Qwen3.5-9B is a vision-language checkpoint; --language-model-only loads just
# the LM weights so it fits the TP=2 VRAM budget.
qwen_extra_args=()
if [[ "$USE_QWEN_DEFAULTS" == "1" && "$MODEL" == "$QWEN_DEFAULT_MODEL" ]]; then
  qwen_extra_args=(
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --language-model-only
  )
fi

# MoE backend: explicit env override wins; otherwise auto-detect from model name.
# --moe-backend marlin is only appropriate for quantized MoE models.
# Full-precision (FP16/BF16) dense or MoE models should use the vLLM default.
if [[ -n "${VLLM_MOE_BACKEND:-}" ]]; then
  qwen_extra_args+=(--moe-backend "$VLLM_MOE_BACKEND")
elif _is_quantized_moe "$MODEL"; then
  qwen_extra_args+=(--moe-backend marlin)
fi

unsloth_extra_args=()
if [[ -n "$UNSLOTH_ARGS" ]]; then
  read -r -a unsloth_extra_args <<< "$UNSLOTH_ARGS"
elif [[ "$USE_UNSLOTH_DEFAULTS" == "1" ]]; then
  unsloth_extra_args=(
    --dtype auto
    --kv-cache-dtype auto
    --enable-prefix-caching
    --max-num-batched-tokens 2048
    --max-num-seqs 16
  )
fi

is_running() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(<"$PID_FILE")"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

discover_listener_pid() {
  local pid=""
  read -r pid < <(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
  if [[ -n "$pid" ]]; then
    printf '%s\n' "$pid"
  fi
}

listener_is_vllm() {
  local pid
  pid="$(discover_listener_pid)"
  [[ -n "$pid" ]] || return 1
  local cmd
  cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ "$cmd" == *"vllm serve"* || "$cmd" == *"vllm.entrypoints"* ]]
}

remove_pid_file_if_stale() {
  if [[ -f "$PID_FILE" ]] && ! is_running; then
    rm -f "$PID_FILE"
  fi
}

wait_for_health() {
  local elapsed=0
  while (( elapsed < STARTUP_TIMEOUT_SECONDS )); do
    if curl --silent --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -f "$PID_FILE" ]]; then
      local pid
      pid="$(<"$PID_FILE")"
      if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
        return 1
      fi
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  return 1
}

require_p2p_boot_args() {
  local cmdline
  cmdline="$(</proc/cmdline)"

  if [[ " $cmdline " != *" intel_iommu=on "* ]]; then
    echo "Refusing start: /proc/cmdline is missing intel_iommu=on." >&2
    echo "Current cmdline: $cmdline" >&2
    return 1
  fi

  if [[ " $cmdline " != *" iommu=pt "* ]]; then
    echo "Refusing start: /proc/cmdline is missing iommu=pt." >&2
    echo "Current cmdline: $cmdline" >&2
    return 1
  fi
}

start_server() {
  remove_pid_file_if_stale
  if is_running; then
    echo "Already running: pid $(<"$PID_FILE")"
    echo "Health: $HEALTH_URL"
    echo "Log: $LOG_FILE"
    return 0
  fi
  if listener_is_vllm && curl --silent --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "Refusing to adopt unmanaged vLLM on port $PORT" >&2
    echo "PID: $(discover_listener_pid)" >&2
    echo "Health: $HEALTH_URL" >&2
    echo "Stop it and relaunch through this script." >&2
    return 1
  fi

  mkdir -p "$LOG_DIR" "$RUN_DIR"
  require_p2p_boot_args
  rm -f ~/.cache/vllm/gpu_p2p_access_cache_for_*.json

  local cmd=(
    vllm serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size 2
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --generation-config vllm
    --disable-log-stats
    --enforce-eager
  )
  cmd+=("${qwen_extra_args[@]}")
  cmd+=("${unsloth_extra_args[@]}")
  cmd+=("$@")

  (
    source "$VENV_PATH/bin/activate"
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    export OMP_NUM_THREADS=1
    export GLOO_SOCKET_IFNAME=lo
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE"
    export NCCL_SHM_DISABLE=0
    export VLLM_SKIP_P2P_CHECK="$VLLM_SKIP_P2P_CHECK"
    export VLLM_MARLIN_USE_ATOMIC_ADD=1
    nohup "${cmd[@]}" >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
  )

  if wait_for_health; then
    echo "Started $MODEL"
    echo "PID: $(<"$PID_FILE")"
    echo "Health: $HEALTH_URL"
    echo "Log: $LOG_FILE"
    return 0
  fi

  echo "Startup failed for $MODEL" >&2
  echo "Log: $LOG_FILE" >&2
  remove_pid_file_if_stale
  return 1
}

stop_server() {
  remove_pid_file_if_stale
  if ! is_running; then
    if listener_is_vllm; then
      local unmanaged_pid
      unmanaged_pid="$(discover_listener_pid)"
      kill "$unmanaged_pid" 2>/dev/null || true
      for _ in {1..10}; do
        if ! kill -0 "$unmanaged_pid" 2>/dev/null; then
          echo "Stopped unmanaged $MODEL on port $PORT"
          return 0
        fi
        sleep 1
      done
      kill -9 "$unmanaged_pid" 2>/dev/null || true
      echo "Force-stopped unmanaged $MODEL on port $PORT"
      return 0
    fi
    echo "Not running"
    return 0
  fi

  local pid
  pid="$(<"$PID_FILE")"
  kill "$pid" 2>/dev/null || true
  for _ in {1..10}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Stopped $MODEL"
      return 0
    fi
    sleep 1
  done

  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "Force-stopped $MODEL"
}

status_server() {
  remove_pid_file_if_stale
  if is_running; then
    echo "Running"
    echo "PID: $(<"$PID_FILE")"
    echo "Health: $HEALTH_URL"
    echo "Log: $LOG_FILE"
    return 0
  fi
  if listener_is_vllm && curl --silent --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "Running (unmanaged)"
    echo "PID: $(discover_listener_pid)"
    echo "Health: $HEALTH_URL"
    echo "Expected log: $LOG_FILE"
    return 0
  fi
  echo "Stopped"
  echo "Expected log: $LOG_FILE"
  return 1
}

health_server() {
  curl --silent --max-time 10 "$HEALTH_URL"
}

case "$ACTION" in
  start)
    start_server "$@"
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server "$@"
    ;;
  status)
    status_server
    ;;
  health)
    health_server
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
