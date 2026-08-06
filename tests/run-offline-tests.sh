#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m compileall -q install.py install_core.py scripts tests
bash -n scripts/manage_vllm_safe_tp2.sh
bash -n scripts/manage_vllm_safe_tp2_core.sh
bash -n scripts/post-reboot-test.sh
python3 -m unittest discover -s tests -p 'test_*.py' -v

if command -v nvcc >/dev/null 2>&1; then
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT
  nvcc -O2 -std=c++17 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_86,code=compute_86 \
    scripts/p2p_probe.cu -o "$tmpdir/p2p_probe"
  nvcc -O2 -std=c++17 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_86,code=compute_86 \
    scripts/p2p_bandwidth_bench.cu -o "$tmpdir/p2p_bandwidth_bench"
else
  echo "nvcc not present; CUDA compilation test skipped"
fi
