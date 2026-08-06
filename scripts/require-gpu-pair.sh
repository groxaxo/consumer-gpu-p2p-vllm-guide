#!/usr/bin/env bash
set -euo pipefail

# Compatibility preflight. The old implementation checked only Intel boot
# arguments. This replacement validates the complete machine-bound profile for
# the requested device set.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEVICES="${1:-${CUDA_VISIBLE_DEVICES:-0,1}}"
VENV_PATH="${VLLM_VENV_PATH:-$HOME/venvs/vllm}"
PROFILE_PATH="${P2P_PROFILE_PATH:-$HOME/.config/vllm/consumer-p2p.env}"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "vLLM Python not found: $VENV_PATH/bin/python" >&2
  exit 1
fi

exec "$VENV_PATH/bin/python" "$SCRIPT_DIR/p2p_doctor.py" check-profile \
  --devices "$DEVICES" \
  --profile "$PROFILE_PATH"
