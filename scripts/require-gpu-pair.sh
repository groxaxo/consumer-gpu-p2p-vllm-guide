#!/usr/bin/env bash
set -euo pipefail
#
# Pre-flight check for TP=2 GPU pair.
# Validates boot args required for the patched NVIDIA driver.
#
# ACS control registers are read-only on Intel consumer root complexes —
# writes are silently ignored. NCCL SHM transport works regardless.

GPU_PAIR="${1:-${CUDA_VISIBLE_DEVICES:-0,1}}"
CMDLINE="$(</proc/cmdline)"

if [[ "$GPU_PAIR" != *,* ]]; then
  echo "Refusing validation: expected a two-GPU pair, got '$GPU_PAIR'." >&2
  exit 1
fi

if [[ " $CMDLINE " != *" intel_iommu=on "* ]]; then
  echo "Refusing start: /proc/cmdline is missing intel_iommu=on." >&2
  echo "Current cmdline: $CMDLINE" >&2
  exit 1
fi

if [[ " $CMDLINE " != *" iommu=pt "* ]]; then
  echo "Refusing start: /proc/cmdline is missing iommu=pt." >&2
  echo "Current cmdline: $CMDLINE" >&2
  exit 1
fi

echo "GPU pair $GPU_PAIR: boot args OK."
