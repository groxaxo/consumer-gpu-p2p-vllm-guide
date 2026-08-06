# 4. vLLM setup with validated P2P

## Install the pinned environment

`install.py` creates `~/venvs/vllm` by default and installs the runtime in one
resolver transaction:

```bash
python3 install.py --yes
```

Manual equivalent:

```bash
python3 -m venv ~/venvs/vllm
~/venvs/vllm/bin/python -m pip install --upgrade pip setuptools wheel
~/venvs/vllm/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 \
  torchvision==0.26.0+cu128 \
  torchaudio==2.11.0+cu128 \
  vllm==0.21.0
~/venvs/vllm/bin/python -m pip check
```

Verify:

```bash
~/venvs/vllm/bin/python - <<'PY'
import torch, vllm
print("torch", torch.__version__)
print("CUDA runtime", torch.version.cuda)
print("vLLM", vllm.__version__)
PY
```

The NVIDIA kernel/userspace driver and PyTorch's bundled CUDA runtime are
separate version layers. The host driver must be new enough for the runtime and,
for this patch, must remain exactly `595.58.03`.

## Generate a validated profile first

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/post-reboot-test.sh
```

The profile contains:

```bash
NCCL_P2P_DISABLE=0
NCCL_SHM_DISABLE=0
VLLM_SKIP_P2P_CHECK=0
```

The last setting is critical. In vLLM it enables the real CUDA IPC validation
path. Once generated, vLLM reuses its directed-pair cache instead of repeating
the expensive subprocess test.

## Launch through the profile-gated manager

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

The backward-compatible filename no longer hardcodes TP=2. It defaults TP size
to the number of visible devices:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

Override only when the visible set intentionally contains spare devices:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
VLLM_TENSOR_PARALLEL_SIZE=2 \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

The saved profile still validates all visible devices in that command. For a
pair-only profile and test, expose only the pair.

## Manual launch

After `check-profile` succeeds, source the profile and keep the real vLLM check
enabled:

```bash
export CUDA_VISIBLE_DEVICES=0,1
~/venvs/vllm/bin/python scripts/p2p_doctor.py check-profile \
  --devices "$CUDA_VISIBLE_DEVICES"
source ~/.config/vllm/consumer-p2p.env

~/venvs/vllm/bin/vllm serve <model-id> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --enforce-eager
```

Do not replace the validated values with `NCCL_P2P_DISABLE=1` or
`VLLM_SKIP_P2P_CHECK=1` and still call the run P2P-enabled.

## TP=2 versus TP=3

vLLM custom all-reduce supports selected world sizes, including 2, but not 3.
On three PCIe GPUs:

- NCCL can still use validated peer access for TP=3.
- vLLM logs that custom all-reduce is disabled because world size 3 is
  unsupported.
- This is a backend limitation, not evidence that NCCL P2P failed.

Use TP=2 when low-latency custom all-reduce and model fit allow it. Use TP=3
when capacity requires all three GPUs, and benchmark NCCL communication.

## CUDA graphs and memory

The launcher defaults `VLLM_ENFORCE_EAGER=1` because near-full 24 GiB cards can
OOM during CUDA graph capture. This setting is independent of P2P. When a model
leaves enough headroom, benchmark:

```bash
VLLM_ENFORCE_EAGER=0 \
  bash scripts/manage_vllm_safe_tp2.sh restart <model-id>
```

Do not present a fixed percentage difference as universal; graph benefits vary
by model, batch shape, context length, and vLLM release.

## Health and logs

```bash
bash scripts/manage_vllm_safe_tp2.sh status <model-id>
bash scripts/manage_vllm_safe_tp2.sh health <model-id>

# The status command prints the exact log path.
grep -Ei 'P2P|NCCL|all.?reduce|SHM|IPC' ~/.local/state/vllm/*.log
```

The launcher never adopts or kills an unmanaged listener on the configured
port. Stop that process explicitly or use another port.

## Fallback

When validation fails and the model must run before the platform is repaired:

```bash
VLLM_P2P_MODE=shm \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

This sets `NCCL_P2P_DISABLE=1` and passes
`--disable-custom-all-reduce`. It is a safe, explicit host-memory fallback, not
P2P.
