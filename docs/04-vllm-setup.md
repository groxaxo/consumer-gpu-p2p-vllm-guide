# 4. vLLM setup with validated P2P

## Install the exact CUDA 12.9 runtime

vLLM `0.21.0` publishes its normal Linux precompiled wheel for CUDA 12.9. The
installer therefore resolves one exact ABI-compatible transaction:

```text
torch==2.11.0+cu129
torchvision==0.26.0+cu129
torchaudio==2.11.0+cu129
vllm==0.21.0
```

Automated installation:

```bash
python3 install.py --yes
```

Manual equivalent:

```bash
python3 -m venv ~/venvs/vllm
~/venvs/vllm/bin/python -m pip install --upgrade pip setuptools wheel
~/venvs/vllm/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  torch==2.11.0+cu129 \
  torchvision==0.26.0+cu129 \
  torchaudio==2.11.0+cu129 \
  vllm==0.21.0
~/venvs/vllm/bin/python -m pip check
```

Verify every layer rather than checking only that imports succeed:

```bash
~/venvs/vllm/bin/python - <<'PY'
import json
import torch, torchvision, torchaudio, vllm
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
    "vllm": vllm.__version__,
}, indent=2, sort_keys=True))
PY
```

Expected:

```json
{
  "torch": "2.11.0+cu129",
  "torch_cuda": "12.9",
  "torchaudio": "2.11.0+cu129",
  "torchvision": "0.26.0+cu129",
  "vllm": "0.21.0"
}
```

The host NVIDIA driver, PyTorch CUDA runtime, and precompiled vLLM extension are
separate version layers. The P2P patch requires host driver/userspace
`595.58.03`; the Python environment must keep PyTorch and vLLM on the same CUDA
12.9 ABI.

## Generate a validated profile first

For a pair:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
P2P_PROFILE_PATH=~/.config/vllm/p2p-0-1.env \
  bash scripts/post-reboot-test.sh
```

For all three GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
P2P_PROFILE_PATH=~/.config/vllm/p2p-0-1-2.env \
  bash scripts/post-reboot-test.sh
```

A validated profile contains:

```bash
NCCL_P2P_DISABLE=0
NCCL_SHM_DISABLE=0
VLLM_SKIP_P2P_CHECK=0
```

The last setting is critical. It keeps vLLM's real cross-process CUDA IPC
validation active. vLLM caches successful directed device-pair results instead
of blindly trusting the driver's capability report.

## Launch through the profile gate

Pair / TP=2:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
P2P_PROFILE_PATH=~/.config/vllm/p2p-0-1.env \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

Three GPUs / TP=3:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
P2P_PROFILE_PATH=~/.config/vllm/p2p-0-1-2.env \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

The backward-compatible filename does not hardcode TP=2. Tensor-parallel size
defaults to the visible device count. Numeric physical GPU indices are required
because vLLM 0.21's custom all-reduce device mapping parses
`CUDA_VISIBLE_DEVICES` as integers.

Override TP only when the visible set intentionally contains spare devices:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
VLLM_TENSOR_PARALLEL_SIZE=2 \
P2P_PROFILE_PATH=~/.config/vllm/p2p-0-1-2.env \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

The profile still validates the complete visible set. Prefer exposing only the
pair when a pair-only run is intended.

## Manual launch

After profile validation:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export P2P_PROFILE_PATH="$HOME/.config/vllm/p2p-0-1.env"

~/venvs/vllm/bin/python scripts/p2p_doctor.py check-profile \
  --devices "$CUDA_VISIBLE_DEVICES" \
  --profile "$P2P_PROFILE_PATH"

# The file is owner-only, allowlisted, and checked before the managed launcher
# sources it. For a manual launch, inspect it before sourcing.
source "$P2P_PROFILE_PATH"

~/venvs/vllm/bin/vllm serve <model-id> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --enforce-eager
```

Do not replace the validated values with `NCCL_P2P_DISABLE=1` or
`VLLM_SKIP_P2P_CHECK=1` and continue describing the run as P2P-enabled.

## TP=2 versus TP=3

### TP=2

vLLM 0.21's ordinary custom all-reduce path supports world size 2. On Ampere it
uses the standard custom-all-reduce buffer path, provided the two GPUs are fully
connected and real P2P validation succeeds.

### TP=3

vLLM custom all-reduce supports world sizes 2, 4, 6, and 8, not 3. A three-GPU
run therefore uses NCCL. NCCL can still use validated P2P between all three
GPUs. A log explaining that custom all-reduce is disabled at world size 3 is an
expected backend distinction, not a P2P failure.

## CUDA graphs and memory

The launcher defaults to:

```bash
VLLM_ENFORCE_EAGER=1
```

Near-full 24 GiB cards can OOM during CUDA graph capture. This setting is
independent of P2P. When the model leaves sufficient headroom, benchmark graph
mode explicitly:

```bash
VLLM_ENFORCE_EAGER=0 \
  bash scripts/manage_vllm_safe_tp2.sh restart <model-id>
```

Graph benefit varies by model, batch shape, context length, and vLLM release.
Do not attach a universal performance percentage to this toggle.

## Health and logs

```bash
bash scripts/manage_vllm_safe_tp2.sh status <model-id>
bash scripts/manage_vllm_safe_tp2.sh health <model-id>
grep -Ei 'P2P|NCCL|all.?reduce|SHM|NET|IPC' ~/.local/state/vllm/*.log
```

The launcher never adopts or kills an unmanaged process already listening on
the configured port.

## Explicit fallback

When P2P validation fails but correct inference must continue:

```bash
VLLM_P2P_MODE=shm \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

This sets `NCCL_P2P_DISABLE=1` and adds `--disable-custom-all-reduce`. It is a
host-memory fallback, not a successful P2P configuration.
