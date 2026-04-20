# 6. Ollama Multi-GPU

Ollama can spread a single model across multiple GPUs for inference. Unlike
vLLM, Ollama does not use NCCL — it relies directly on the CUDA
`cudaMemcpyPeer` DMA staging path (the same path the patched driver enables).
This means Ollama works out of the box once the patched driver is installed,
without any NCCL environment variables.

> **Ollama vs vLLM:** Use Ollama for quick interactive queries and simple
> API access. Use vLLM when you need tensor parallelism, higher throughput,
> batching, or OpenAI-compatible chat completions at scale. Both can use the
> same GPU pair, but **not simultaneously** — see the resource conflict section.

---

## Prerequisites

Complete docs [01](01-boot-config.md), [02](02-patched-driver.md) first. The
patched NVIDIA driver must be loaded before Ollama can use `cudaMemcpyPeer`.

Verify the driver is active:

```bash
modinfo nvidia | grep version
# version:        595.58.03
```

---

## Systemd Configuration

Create a systemd drop-in to tell Ollama which GPUs to use:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/gpu-pair.conf > /dev/null <<'EOF'
[Service]
Environment="CUDA_DEVICE_ORDER=PCI_BUS_ID"
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="OLLAMA_NUM_GPU=2"
Environment="OLLAMA_KEEP_ALIVE=5m"
EOF
```

Apply and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo systemctl status ollama
```

**Variable reference:**

| Variable                   | Value | What it does                                                  |
|---                         |---    |---                                                            |
| `CUDA_DEVICE_ORDER`        | `PCI_BUS_ID` | GPU indices are assigned by PCIe slot order, not discovery order. Makes GPU numbering stable across reboots. |
| `CUDA_VISIBLE_DEVICES`     | `0,1` | Restricts Ollama to these two GPU indices. Use the pair with the best bandwidth (run `scripts/p2p_bandwidth_bench.py` to find it). |
| `OLLAMA_NUM_GPU`           | `2`   | Explicitly tells Ollama to split the model across 2 GPUs.    |
| `OLLAMA_KEEP_ALIVE`        | `5m`  | How long to keep a loaded model in VRAM after the last request before unloading. Set to `-1` to keep forever. |

---

## GPU Selection

The best GPU pair for inference is the two RTX 3090s — 24 GiB each gives 48 GiB
total, enough for large models. The RTX 3060s (12 GiB each) have insufficient
VRAM for most useful multi-GPU models and their PCIe slots are bandwidth-starved
on this board (Gen1 x1, ~0.8 GB/s).

```bash
# See which GPU indices correspond to which cards
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv

# Run the bandwidth benchmark to find your best pair
python3 scripts/p2p_bandwidth_bench.py
```

To use a specific pair (e.g. GPU1 and GPU4 if those are both 3090s):

```bash
CUDA_VISIBLE_DEVICES=1,4 OLLAMA_NUM_GPU=2 ollama serve
```

Or update the systemd drop-in's `CUDA_VISIBLE_DEVICES` line.

---

## Pre-flight Boot Arg Validation

If you want Ollama to refuse to start when the required boot args are missing
(e.g. after a kernel update that removed them), create a wrapper script:

```bash
sudo tee /usr/local/bin/ollama-p2p-serve > /dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OLLAMA_NUM_GPU="${OLLAMA_NUM_GPU:-2}"

CMDLINE="$(</proc/cmdline)"
if [[ " $CMDLINE " != *" intel_iommu=on "* ]]; then
  echo "ERROR: intel_iommu=on not in /proc/cmdline — patched driver won't work" >&2
  exit 1
fi
if [[ " $CMDLINE " != *" iommu=pt "* ]]; then
  echo "ERROR: iommu=pt not in /proc/cmdline" >&2
  exit 1
fi

exec /usr/local/bin/ollama serve
EOF
sudo chmod +x /usr/local/bin/ollama-p2p-serve
```

Then update the systemd unit to use the wrapper:

```bash
sudo tee /etc/systemd/system/ollama.service.d/preflight.conf > /dev/null <<'EOF'
[Service]
ExecStart=
ExecStart=/usr/local/bin/ollama-p2p-serve
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

---

## Verification

```bash
# Check the service started and is using both GPUs
sudo systemctl status ollama

# Watch GPU memory — both GPUs should show usage when a model is loaded
watch -n1 nvidia-smi

# Pull a small model and run a query
ollama pull qwen3:8b
ollama run qwen3:8b "Hello — how many GPUs are you running on?"

# Check API directly
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

After `ollama run`, `nvidia-smi` should show VRAM allocated on both GPUs.

---

## Resource Conflict with vLLM

**Ollama and vLLM cannot run on the same GPUs at the same time.** Both try to
allocate the majority of VRAM on their assigned GPUs. If both are running on
the same pair, one will OOM.

Options:

1. **Run only one at a time** — stop Ollama before starting vLLM, and vice versa:
   ```bash
   sudo systemctl stop ollama
   bash scripts/manage_vllm_safe_tp2.sh start
   ```

2. **Pin them to different GPU pairs** — if you have 4+ GPUs, give each service
   its own pair:
   ```bash
   # Ollama on GPU 2 and 3
   CUDA_VISIBLE_DEVICES=2,3 in the systemd drop-in

   # vLLM on GPU 0 and 1
   CUDA_VISIBLE_DEVICES=0,1 in manage_vllm_safe_tp2.sh
   ```

3. **Use vLLM exclusively** — vLLM's OpenAI-compatible API covers all Ollama
   use cases, and vLLM generally gives higher throughput and lower latency.

---

## Troubleshooting

**Ollama not using both GPUs:**
- Check `OLLAMA_NUM_GPU=2` is set in the systemd drop-in
- Verify with `watch nvidia-smi` that VRAM appears on both GPUs after loading
- Ensure the patched driver is active (`modinfo nvidia | grep version`)

**Model loads slowly or only one GPU is active:**
- Run `python3 scripts/p2p_bandwidth_bench.py` — if the pair bandwidth is below
  ~3 GB/s, the GPUs may be on bandwidth-starved PCIe slots. Try a different pair.

**`ollama serve` exits immediately:**
- Check `journalctl -u ollama -n 50` for error messages
- If using the pre-flight wrapper, verify boot args: `cat /proc/cmdline`
