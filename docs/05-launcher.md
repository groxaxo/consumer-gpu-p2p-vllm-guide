# 5. Profile-gated vLLM launcher

`scripts/manage_vllm_safe_tp2.sh` keeps its original filename for compatibility,
but it now supports the visible GPU count rather than hardcoding two GPUs.

## Safety invariants

The launcher:

1. requires a machine-bound P2P profile in `validated` mode;
2. checks the profile against the current driver, loaded kernel, boot arguments,
   GPU UUIDs, PCI bus IDs, selected order, and expected transport variables;
3. leaves vLLM's real CUDA IPC checker enabled;
4. never deletes vLLM's P2P cache during an ordinary start;
5. never adopts or kills an unmanaged process on the configured port;
6. runs vLLM in its own process group so managed worker processes stop together;
7. labels SHM fallback as SHM and disables vLLM custom all-reduce in that mode.

## Commands

```bash
# Run validation without writing a profile
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/manage_vllm_safe_tp2.sh validate

# Validate and write/update the profile
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/manage_vllm_safe_tp2.sh revalidate

# Start
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>

# Add model-specific vLLM flags
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id> \
    --max-num-seqs 8

# Lifecycle
bash scripts/manage_vllm_safe_tp2.sh status <model-id>
bash scripts/manage_vllm_safe_tp2.sh health <model-id>
bash scripts/manage_vllm_safe_tp2.sh restart <model-id>
bash scripts/manage_vllm_safe_tp2.sh stop <model-id>

# Show the selected profile and transport values
bash scripts/manage_vllm_safe_tp2.sh transport
```

## P2P modes

### Validated — default

```bash
VLLM_P2P_MODE=validated
```

Startup fails when the profile is missing or stale. Regenerate it after a
kernel, driver, GPU order, slot, or firmware/boot change.

### Auto — diagnostic

```bash
VLLM_P2P_MODE=auto
```

The launcher enables NCCL P2P and vLLM's actual checker but does not require a
saved profile. This is useful while collecting evidence; it is not the default
production contract.

### SHM — recovery

```bash
VLLM_P2P_MODE=shm
```

The launcher sets:

```bash
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=0
VLLM_SKIP_P2P_CHECK=0
```

and adds `--disable-custom-all-reduce`. This prevents a broken peer route from
being used. It also means the run is not P2P-enabled.

## Environment reference

| Variable | Default | Meaning |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `0,1` | Exact GPU identifiers and order to expose |
| `VLLM_TENSOR_PARALLEL_SIZE` | visible count | Tensor-parallel world size |
| `VLLM_P2P_MODE` | `validated` | `validated`, `auto`, or `shm` |
| `P2P_PROFILE_PATH` | `~/.config/vllm/consumer-p2p.env` | Machine-bound profile |
| `VLLM_VENV_PATH` | `~/venvs/vllm` | Runtime environment |
| `VLLM_MODEL` | `Qwen/Qwen3.5-9B` | Default model when omitted |
| `VLLM_HOST` | `0.0.0.0` | API bind address |
| `VLLM_PORT` | `8000` | API port |
| `VLLM_MAX_MODEL_LEN` | `32768` | Maximum context |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.92` | vLLM memory allocation target |
| `VLLM_ENFORCE_EAGER` | `1` | Disable graph capture when set |
| `VLLM_LOG_DIR` | `~/.local/state/vllm` | Logs |
| `VLLM_RUN_DIR` | XDG runtime/cache path | PID files |
| `VLLM_STARTUP_TIMEOUT_SECONDS` | `300` | Health wait limit |
| `VLLM_MOE_BACKEND` | unset | Optional explicit MoE backend |
| `VLLM_USE_QWEN_TOOLING_DEFAULTS` | `1` | Default-model parser flags |
| `VLLM_USE_UNSLOTH_DEFAULTS` | `1` | Prefix cache/batching defaults |
| `VLLM_UNSLOTH_ARGS` | unset | Replacement operator-supplied CLI args |

The launcher does not accept `NCCL_P2P_DISABLE` or `VLLM_SKIP_P2P_CHECK` as
unsafe overrides in validated mode; the checked profile is the source of truth.

## Profile lifecycle

The profile contains no secret. Its mode is `0600` because it is executable
shell configuration and should not be writable by another local user.

It is intentionally invalidated by:

- running-kernel change;
- NVIDIA driver version change;
- GPU UUID, bus, or order change;
- selected device-list change;
- missing IOMMU passthrough boot state; or
- edits that turn P2P or vLLM verification off.

Revalidate:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
P2P_PROFILE_PATH=~/.config/vllm/p2p-0-1.env \
  bash scripts/manage_vllm_safe_tp2.sh revalidate
```

Use the same variables to launch.
