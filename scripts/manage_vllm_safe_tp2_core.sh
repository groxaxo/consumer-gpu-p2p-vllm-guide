#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible filename; the launcher now supports any TP size that
# matches CUDA_VISIBLE_DEVICES.  P2P is enabled only from a machine-bound
# profile produced by p2p_doctor.py, unless the operator explicitly selects
# auto or SHM fallback mode.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DOCTOR="${P2P_DOCTOR_PATH:-$SCRIPT_DIR/p2p_doctor.py}"

usage() {
  cat <<'USAGE'
Usage:
  manage_vllm_safe_tp2.sh validate [doctor options...]
  manage_vllm_safe_tp2.sh revalidate [doctor options...]
  manage_vllm_safe_tp2.sh start [model] [extra vllm serve args...]
  manage_vllm_safe_tp2.sh stop [model]
  manage_vllm_safe_tp2.sh restart [model] [extra vllm serve args...]
  manage_vllm_safe_tp2.sh status [model]
  manage_vllm_safe_tp2.sh health [model]
  manage_vllm_safe_tp2.sh transport

P2P modes:
  VLLM_P2P_MODE=validated  Require a current p2p_doctor profile (default).
  VLLM_P2P_MODE=auto       Enable NCCL P2P and vLLM's real IPC check without
                           requiring a saved profile. Useful while diagnosing.
  VLLM_P2P_MODE=shm        Explicit recovery mode: disable NCCL P2P and vLLM
                           custom all-reduce. This is not peer-to-peer.

Important environment variables:
  CUDA_VISIBLE_DEVICES              GPU order, e.g. 0,1 or 0,1,2
  VLLM_TENSOR_PARALLEL_SIZE         Defaults to visible-device count
  VLLM_VENV_PATH                    Defaults to ~/venvs/vllm
  P2P_PROFILE_PATH                  Defaults to ~/.config/vllm/consumer-p2p.env
  VLLM_MODEL                        Default model id
  VLLM_HOST / VLLM_PORT             Bind address and port (0.0.0.0:8000)
  VLLM_MAX_MODEL_LEN                Default 32768
  VLLM_GPU_MEMORY_UTILIZATION       Default 0.92
  VLLM_ENFORCE_EAGER                1 by default; set 0 when graph capture fits
  VLLM_LOG_DIR / VLLM_RUN_DIR       Log and PID directories
  VLLM_STARTUP_TIMEOUT_SECONDS      Default 300
  VLLM_USE_QWEN_TOOLING_DEFAULTS    Default 1
  VLLM_USE_UNSLOTH_DEFAULTS         Default 1
  VLLM_UNSLOTH_ARGS                 Replacement space-separated vLLM args
  VLLM_MOE_BACKEND                  Optional explicit backend
USAGE
}

ACTION="${1:-}"
if [[ -z "$ACTION" ]]; then
  usage
  exit 64
fi
shift || true

DEFAULT_MODEL="Qwen/Qwen3.5-9B"
MODEL="${VLLM_MODEL:-$DEFAULT_MODEL}"
case "$ACTION" in
  start|restart|stop|status|health)
    if [[ $# -gt 0 && "$1" != --* ]]; then
      MODEL="$1"
      shift
    fi
    ;;
esac

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a DEVICE_ARRAY <<< "$CUDA_DEVICES"
VISIBLE_COUNT=0
for device in "${DEVICE_ARRAY[@]}"; do
  [[ -n "${device//[[:space:]]/}" ]] && VISIBLE_COUNT=$((VISIBLE_COUNT + 1))
done
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-$VISIBLE_COUNT}"
if ! [[ "$TP_SIZE" =~ ^[0-9]+$ ]] || (( TP_SIZE < 2 || TP_SIZE > VISIBLE_COUNT )); then
  echo "Invalid VLLM_TENSOR_PARALLEL_SIZE=$TP_SIZE for CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" >&2
  exit 64
fi

VENV_PATH="${VLLM_VENV_PATH:-$HOME/venvs/vllm}"
PROFILE_PATH="${P2P_PROFILE_PATH:-$HOME/.config/vllm/consumer-p2p.env}"
P2P_MODE="${VLLM_P2P_MODE:-validated}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-300}"
LOG_DIR="${VLLM_LOG_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/vllm}"
RUN_DIR="${VLLM_RUN_DIR:-${XDG_RUNTIME_DIR:-$HOME/.cache}/vllm-run}"
USE_QWEN_DEFAULTS="${VLLM_USE_QWEN_TOOLING_DEFAULTS:-1}"
USE_UNSLOTH_DEFAULTS="${VLLM_USE_UNSLOTH_DEFAULTS:-1}"
UNSLOTH_ARGS="${VLLM_UNSLOTH_ARGS:-}"

MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/: ' '___' | tr -cd 'A-Za-z0-9._-')"
DEVICE_SLUG="$(printf '%s' "$CUDA_DEVICES" | tr ',:/ ' '____' | tr -cd 'A-Za-z0-9._-')"
PID_FILE="${RUN_DIR}/${MODEL_SLUG}-tp${TP_SIZE}-${DEVICE_SLUG}-${PORT}.pid"
LOG_FILE="${LOG_DIR}/${MODEL_SLUG}-tp${TP_SIZE}-${DEVICE_SLUG}-${PORT}.log"
HEALTH_URL="http://127.0.0.1:${PORT}/v1/models"

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $description: $path" >&2
    exit 1
  fi
}

require_runtime() {
  require_file "$DOCTOR" "P2P doctor"
  require_file "$VENV_PATH/bin/activate" "vLLM virtual environment"
  require_file "$VENV_PATH/bin/vllm" "vLLM executable"
  command -v curl >/dev/null 2>&1 || {
    echo "curl is required" >&2
    exit 1
  }
  command -v lsof >/dev/null 2>&1 || {
    echo "lsof is required" >&2
    exit 1
  }
}

profile_check() {
  "$VENV_PATH/bin/python" "$DOCTOR" check-profile \
    --devices "$CUDA_DEVICES" \
    --profile "$PROFILE_PATH"
}

configure_transport() {
  TRANSPORT_EXTRA_ARGS=()
  case "$P2P_MODE" in
    validated)
      profile_check
      # p2p_doctor writes only strict KEY=value exports and validates this file
      # before it is sourced here.
      # shellcheck disable=SC1090
      source "$PROFILE_PATH"
      if [[ "${P2P_PROFILE_STATUS:-}" != "validated" || \
            "${P2P_PROFILE_DEVICES:-}" != "$CUDA_DEVICES" || \
            "${NCCL_P2P_DISABLE:-}" != "0" || \
            "${VLLM_SKIP_P2P_CHECK:-}" != "0" ]]; then
        echo "Validated profile contains unsafe or mismatched values: $PROFILE_PATH" >&2
        exit 1
      fi
      ;;
    auto)
      export NCCL_P2P_DISABLE=0
      export NCCL_SHM_DISABLE=0
      export VLLM_SKIP_P2P_CHECK=0
      echo "WARNING: VLLM_P2P_MODE=auto has no persisted hardware gate." >&2
      echo "         vLLM's real CUDA IPC checker remains enabled." >&2
      ;;
    shm)
      export NCCL_P2P_DISABLE=1
      export NCCL_SHM_DISABLE=0
      export VLLM_SKIP_P2P_CHECK=0
      TRANSPORT_EXTRA_ARGS=(--disable-custom-all-reduce)
      echo "WARNING: SHM fallback selected. Direct GPU P2P is disabled." >&2
      ;;
    *)
      echo "Unknown VLLM_P2P_MODE=$P2P_MODE (expected validated, auto, or shm)" >&2
      exit 64
      ;;
  esac
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

remove_stale_pid() {
  if [[ -f "$PID_FILE" ]] && ! is_running; then
    rm -f "$PID_FILE"
  fi
}

listener_pid() {
  lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n1 || true
}

listener_is_vllm() {
  local pid
  pid="$(listener_pid)"
  [[ -n "$pid" ]] || return 1
  local command_line
  command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ "$command_line" == *"vllm serve"* || "$command_line" == *"vllm.entrypoints"* ]]
}

wait_for_health() {
  local elapsed=0
  while (( elapsed < STARTUP_TIMEOUT_SECONDS )); do
    if curl --silent --fail --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -f "$PID_FILE" ]]; then
      local pid
      pid="$(<"$PID_FILE")"
      if ! kill -0 "$pid" 2>/dev/null; then
        return 1
      fi
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  return 1
}

_is_quantized_moe() {
  local lowered="${1,,}"
  [[ "$lowered" =~ -a[0-9]+b ]] && \
    [[ "$lowered" =~ (fp8|gptq|awq|int4|int8|w4|w8|bnb) ]]
}

build_model_args() {
  MODEL_EXTRA_ARGS=()
  if [[ "$USE_QWEN_DEFAULTS" == "1" && "$MODEL" == "$DEFAULT_MODEL" ]]; then
    MODEL_EXTRA_ARGS+=(
      --reasoning-parser qwen3
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --language-model-only
    )
  fi

  if [[ -n "${VLLM_MOE_BACKEND:-}" ]]; then
    MODEL_EXTRA_ARGS+=(--moe-backend "$VLLM_MOE_BACKEND")
  elif _is_quantized_moe "$MODEL"; then
    MODEL_EXTRA_ARGS+=(--moe-backend marlin)
  fi

  PERFORMANCE_ARGS=()
  if [[ -n "$UNSLOTH_ARGS" ]]; then
    # Intentional operator-provided word splitting for vLLM CLI options.
    # shellcheck disable=SC2206
    PERFORMANCE_ARGS=($UNSLOTH_ARGS)
  elif [[ "$USE_UNSLOTH_DEFAULTS" == "1" ]]; then
    PERFORMANCE_ARGS=(
      --dtype auto
      --kv-cache-dtype auto
      --enable-prefix-caching
      --max-num-batched-tokens 2048
      --max-num-seqs 16
    )
  fi
}

start_server() {
  require_runtime
  remove_stale_pid
  if is_running; then
    echo "Already running: PID $(<"$PID_FILE")"
    echo "Health: $HEALTH_URL"
    echo "Log: $LOG_FILE"
    return 0
  fi

  local existing_pid
  existing_pid="$(listener_pid)"
  if [[ -n "$existing_pid" ]]; then
    echo "Refusing to start: port $PORT is already owned by PID $existing_pid." >&2
    echo "This launcher never adopts or kills an unmanaged listener." >&2
    return 1
  fi

  configure_transport
  build_model_args
  mkdir -p "$LOG_DIR" "$RUN_DIR"

  local command=(
    vllm serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --generation-config vllm
    --disable-log-stats
  )
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    command+=(--enforce-eager)
  fi
  command+=("${TRANSPORT_EXTRA_ARGS[@]}")
  command+=("${MODEL_EXTRA_ARGS[@]}")
  command+=("${PERFORMANCE_ARGS[@]}")
  command+=("$@")

  echo "Launching $MODEL on CUDA_VISIBLE_DEVICES=$CUDA_DEVICES (TP=$TP_SIZE)"
  echo "P2P mode: $P2P_MODE; NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE; VLLM_SKIP_P2P_CHECK=$VLLM_SKIP_P2P_CHECK"
  if (( TP_SIZE == 3 )); then
    echo "Note: vLLM custom all-reduce does not support world size 3; NCCL can still use validated P2P."
  fi

  (
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    export NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_SHM_DISABLE VLLM_SKIP_P2P_CHECK
    export VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-1}"
    nohup setsid "${command[@]}" >"$LOG_FILE" 2>&1 < /dev/null &
    echo $! >"$PID_FILE"
  )

  if wait_for_health; then
    echo "Started: PID $(<"$PID_FILE")"
    echo "Health: $HEALTH_URL"
    echo "Log: $LOG_FILE"
    return 0
  fi

  echo "Startup failed. Last log lines:" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  stop_server >/dev/null 2>&1 || true
  return 1
}

stop_server() {
  remove_stale_pid
  if ! is_running; then
    if listener_is_vllm; then
      echo "Refusing to stop unmanaged vLLM PID $(listener_pid) on port $PORT." >&2
      return 1
    fi
    echo "Not running"
    return 0
  fi

  local pid
  pid="$(<"$PID_FILE")"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Stopped $MODEL"
      return 0
    fi
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "Force-stopped $MODEL"
}

status_server() {
  remove_stale_pid
  if is_running; then
    echo "Running"
    echo "PID: $(<"$PID_FILE")"
    echo "Health: $HEALTH_URL"
    echo "Log: $LOG_FILE"
    return 0
  fi
  local pid
  pid="$(listener_pid)"
  if [[ -n "$pid" ]]; then
    echo "Port $PORT is occupied by unmanaged PID $pid"
    return 2
  fi
  echo "Stopped"
  return 1
}

health_server() {
  curl --silent --fail --max-time 10 "$HEALTH_URL"
  printf '\n'
}

validate_transport() {
  require_runtime
  "$VENV_PATH/bin/python" "$DOCTOR" validate \
    --devices "$CUDA_DEVICES" \
    --venv "$VENV_PATH" \
    --profile "$PROFILE_PATH" \
    "$@"
}

revalidate_transport() {
  validate_transport --write-profile "$@"
}

show_transport() {
  echo "mode=$P2P_MODE"
  echo "devices=$CUDA_DEVICES"
  echo "tp_size=$TP_SIZE"
  echo "profile=$PROFILE_PATH"
  if [[ -f "$PROFILE_PATH" ]]; then
    profile_check
    grep -E '^(export )?(P2P_PROFILE_[A-Z0-9_]*|NCCL_P2P_DISABLE|NCCL_SHM_DISABLE|VLLM_SKIP_P2P_CHECK)=' "$PROFILE_PATH"
  else
    echo "profile_status=missing"
  fi
}

case "$ACTION" in
  validate)
    validate_transport "$@"
    ;;
  revalidate)
    revalidate_transport "$@"
    ;;
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
  transport)
    show_transport
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 64
    ;;
esac
