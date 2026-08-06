# 6. Ollama note

This repository's acceptance gate is designed for **vLLM, CUDA IPC, and NCCL**.
It no longer claims that an Ollama release uses a particular internal transport
or environment variable without version-specific evidence.

A successful `p2p_doctor.py` run proves that the selected CUDA devices can
perform direct peer kernel reads/writes and cross-process CUDA IPC on this
host. Whether a particular Ollama/llama.cpp build uses that path, host staging,
or another backend is determined by that build and its launch parameters.

## Safe coexistence

Do not run Ollama and vLLM on the same GPUs simultaneously unless memory has
been explicitly partitioned and tested. The usual operational pattern is:

```bash
sudo systemctl stop ollama
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

Or pin each service to a disjoint GPU set.

## Validate before benchmarking Ollama

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/post-reboot-test.sh
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/p2p_bandwidth_bench.py
```

Then enable Ollama's debug logging for the installed version and inspect its
actual backend selection. Avoid treating VRAM allocation on two cards as proof
that direct P2P is used.

For vLLM, continue with [04 — vLLM setup](04-vllm-setup.md).
