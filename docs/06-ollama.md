# 6. Ollama Multi-GPU

Ollama can also use multiple GPUs for inference. Since it doesn't use NCCL
directly, it relies on the same CUDA `cudaMemcpyPeer` path (DMA staging).

## Configuration

Create a systemd drop-in at `/etc/systemd/system/ollama.service.d/gpu-pair.conf`:

```ini
[Service]
Environment="CUDA_DEVICE_ORDER=PCI_BUS_ID"
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="OLLAMA_NUM_GPU=2"
Environment="OLLAMA_KEEP_ALIVE=5m"
```

Reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## Pre-flight Script

If you want Ollama to validate boot args before starting, create a wrapper:

```bash
#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OLLAMA_NUM_GPU="${OLLAMA_NUM_GPU:-2}"

# Verify boot args
CMDLINE="$(</proc/cmdline)"
if [[ " $CMDLINE " != *" intel_iommu=on "* ]]; then
  echo "ERROR: intel_iommu=on not in /proc/cmdline" >&2
  exit 1
fi

exec /usr/local/bin/ollama serve
```

Point the systemd unit at this wrapper via an `ExecStart=` override.

## Resource Conflict with vLLM

**Ollama and vLLM cannot run simultaneously on the same GPUs.** Both will
attempt to allocate VRAM, causing OOM. Options:

1. Run only one at a time
2. Pin Ollama to different GPUs (e.g., `CUDA_VISIBLE_DEVICES=2,3`)
3. Use vLLM exclusively and skip Ollama

## GPU Selection

The 2x RTX 3090 (24 GiB each) are the best pair for inference. The RTX 3060s
(12 GiB each) have insufficient VRAM for most TP=2 models.

```bash
# Use the two 3090s on PCH lanes (GPU 1 and GPU 4)
CUDA_VISIBLE_DEVICES=1,4 OLLAMA_NUM_GPU=2 ollama serve
```
