#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VLLM_VENV_PATH:-$HOME/venvs/vllm}"
DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PROFILE_PATH="${P2P_PROFILE_PATH:-$HOME/.config/vllm/consumer-p2p.env}"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "vLLM Python not found: $VENV_PATH/bin/python" >&2
  exit 1
fi

exec "$VENV_PATH/bin/python" "$SCRIPT_DIR/p2p_doctor.py" validate \
  --devices "$DEVICES" \
  --venv "$VENV_PATH" \
  --profile "$PROFILE_PATH" \
  --write-profile \
  "$@"
