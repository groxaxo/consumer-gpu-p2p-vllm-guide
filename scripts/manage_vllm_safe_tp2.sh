#!/usr/bin/env bash
set -euo pipefail

# Fail-closed front-end. The lifecycle implementation remains in the reviewed
# core script; this boundary canonicalizes vLLM's numeric GPU ordering and
# requires the hardened machine-bound profile before a validated start.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CORE="$SCRIPT_DIR/manage_vllm_safe_tp2_core.sh"
DOCTOR="${P2P_DOCTOR_PATH:-$SCRIPT_DIR/p2p_doctor.py}"
VENV_PATH="${VLLM_VENV_PATH:-$HOME/venvs/vllm}"
PROFILE_PATH="${P2P_PROFILE_PATH:-$HOME/.config/vllm/consumer-p2p.env}"
ACTION="${1:-}"
MODE="${VLLM_P2P_MODE:-validated}"
RAW_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

[[ -x "$CORE" ]] || { echo "Missing launcher core: $CORE" >&2; exit 1; }

IFS=',' read -r -a RAW_DEVICE_ARRAY <<< "$RAW_DEVICES"
declare -A SEEN_DEVICES=()
NORMALIZED_DEVICES=()
for raw_device in "${RAW_DEVICE_ARRAY[@]}"; do
  device="$raw_device"
  device="${device#"${device%%[![:space:]]*}"}"
  device="${device%"${device##*[![:space:]]}"}"
  [[ "$device" =~ ^[0-9]+$ ]] || {
    echo "CUDA_VISIBLE_DEVICES must contain numeric physical GPU indices; got: $raw_device" >&2
    exit 64
  }
  device="$((10#$device))"
  [[ -z "${SEEN_DEVICES[$device]+x}" ]] || {
    echo "Duplicate GPU index in CUDA_VISIBLE_DEVICES: $device" >&2
    exit 64
  }
  SEEN_DEVICES[$device]=1
  NORMALIZED_DEVICES+=("$device")
done
(( ${#NORMALIZED_DEVICES[@]} >= 2 )) || {
  echo "At least two physical GPU indices are required." >&2
  exit 64
}
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${NORMALIZED_DEVICES[*]}")"

case "$ACTION" in
  start|restart)
    if [[ "$MODE" == "validated" ]]; then
      [[ -x "$VENV_PATH/bin/python" ]] || {
        echo "vLLM Python not found: $VENV_PATH/bin/python" >&2
        exit 1
      }
      "$VENV_PATH/bin/python" "$DOCTOR" check-profile \
        --devices "$CUDA_VISIBLE_DEVICES" \
        --profile "$PROFILE_PATH"
    fi
    ;;
esac

exec "$CORE" "$@"
