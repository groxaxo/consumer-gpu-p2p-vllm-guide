# Validated consumer NVIDIA P2P for vLLM on Ampere

This repository installs and validates the patched NVIDIA open kernel modules
needed to expose peer access on supported consumer GPUs, then prevents vLLM
from using that route until the selected GPUs pass destructive data-integrity,
CUDA IPC, and NCCL tests.

The primary target is **Linux with two or three RTX 3090 / Ampere GPUs**. The
upstream patch also discusses newer GeForce generations, but this guide's
strict profile deliberately requires compute capability 8.x.

> [!CAUTION]
> This changes kernel modules and enables IOMMU passthrough. Keep a working
> kernel entry and remote recovery path. `iommu=pt` weakens DMA isolation and is
> inappropriate for hosts running untrusted devices or software.

## What version 2 fixes

The old guide could finish successfully while leaving P2P disabled or unsafe:

1. It built patched `595.58.03` kernel modules without requiring the matching
   NVIDIA `595.58.03` userspace driver.
2. It exported `NCCL_P2P_DISABLE=1` while calling the runtime P2P-enabled.
3. It exported `VLLM_SKIP_P2P_CHECK=1`, which tells vLLM to trust the driver's
   capability report instead of performing the real cross-process CUDA IPC
   mutation test.
4. It treated `cudaDeviceCanAccessPeer()` and `cudaMemcpyPeer()` as proof of
   direct mapped-memory access.
5. It deleted vLLM's P2P cache and then skipped the verification that recreates
   it.
6. It installed PyTorch CUDA 12.8 beside the default vLLM 0.21.0 Linux wheel,
   whose compiled CUDA variant is 12.9.
7. It made blanket topology claims instead of testing the actual motherboard,
   firmware, slots, ACS route, and GPU pair.

The corrected path fails closed.

## Exact supported stack

- NVIDIA userspace driver: **595.58.03**
- Patched open modules:
  [`aikitoria/open-gpu-kernel-modules`](https://github.com/aikitoria/open-gpu-kernel-modules/tree/595.58.03-p2p)
- Reviewed upstream source revision:
  `6dd6ba34a4abfb3761797b26102094b856b01edd`
- PyTorch: **2.11.0+cu129**
- torchvision: **0.26.0+cu129**
- torchaudio: **2.11.0+cu129**
- vLLM: **0.21.0**, official Linux CUDA 12.9 wheel
- Primary OS path: Ubuntu 22.04/24.04-class systems using GRUB and DKMS

Do not combine this kernel patch with another NVIDIA userspace version. When
upstream publishes a patch for a newer driver, review and update the userspace
version, kernel-module source revision, Python/CUDA runtime, and validation
evidence together.

## Acceptance gates

A validated launcher profile is written only when every required gate passes:

| Gate | Required result |
|---|---|
| Driver stack | `modinfo`, loaded NVRM, and `nvidia-smi` all report `595.58.03` |
| Boot state | CPU-specific IOMMU enablement, `iommu=pt`, and active IOMMU groups |
| Architecture | Every selected visible GPU reports compute capability 8.x |
| Direct peer memory | Kernels on every GPU read and write exact `uint64_t` patterns in every other GPU's allocation |
| vLLM CUDA IPC | vLLM's own two-process peer-memory mutation test passes in every direction |
| NCCL correctness | Exact finite all-reduce results with `NCCL_P2P_DISABLE=0` |
| NCCL transport | At least one P2P channel is logged and no SHM/NET fallback channel is observed |

The profile is bound to the selected device order, GPU UUIDs, PCI bus IDs,
driver version, running kernel, and boot state. Any relevant change makes it
stale and blocks launch until revalidation.

## Installation

### 1. Get the exact NVIDIA runfile

Download the official **NVIDIA Linux x86_64 595.58.03** runfile from NVIDIA.
This repository deliberately does not mirror the proprietary file or guess its
checksum.

Clone the guide:

```bash
git clone https://github.com/groxaxo/consumer-gpu-p2p-vllm-guide.git
cd consumer-gpu-p2p-vllm-guide
```

Review the plan without changing the host:

```bash
python3 install.py --dry-run --yes
```

When `nvidia-smi` already reports exactly `595.58.03`:

```bash
python3 install.py --lock-driver --yes
```

When another driver is active:

```bash
python3 install.py \
  --driver-runfile "$HOME/Downloads/NVIDIA-Linux-x86_64-595.58.03.run" \
  --install-userspace \
  --lock-driver \
  --yes
```

The runfile is used for matching userspace only (`--no-kernel-modules`). The
installer then checks out the reviewed patch revision and installs its open
kernel modules through DKMS. NVIDIA apt packages can be pinned explicitly;
NCCL is deliberately left upgradeable and must continue to pass validation.

### 2. Reboot

```bash
sudo reboot
```

A reboot is mandatory. Do not validate against the old loaded kernel module.

### 3. Validate the exact GPU set

All three RTX 3090s:

```bash
cd ~/consumer-gpu-p2p-vllm-guide
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/post-reboot-test.sh
```

A selected pair:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
P2P_PROFILE_PATH="$HOME/.config/vllm/p2p-0-1.env" \
  bash scripts/post-reboot-test.sh
```

Strict validation requires `nvcc` to compile the direct peer load/store probe.
A reduced-evidence diagnostic is available when a CUDA toolkit is genuinely
unavailable:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/post-reboot-test.sh --allow-missing-nvcc
```

The vLLM cross-process CUDA IPC and NCCL gates remain mandatory. A successful
strict run ends with:

```text
RESULT=PASS
Wrote validated profile: ~/.config/vllm/consumer-p2p.env
```

No validated profile is written after a required failure, mixed transport, SHM
fallback, NET fallback, or unreported NCCL P2P transport.

### 4. Launch vLLM

Three GPUs / TP=3:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

Selected pair / TP=2:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
P2P_PROFILE_PATH="$HOME/.config/vllm/p2p-0-1.env" \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

The backward-compatible launcher filename no longer hardcodes TP=2. Tensor
parallel size defaults to the visible GPU count.

Validated mode exports:

```bash
NCCL_P2P_DISABLE=0
NCCL_SHM_DISABLE=0
VLLM_SKIP_P2P_CHECK=0
```

`VLLM_SKIP_P2P_CHECK=0` is intentional. vLLM performs the real peer CUDA IPC
check and caches the directed-pair result under `~/.cache/vllm/`. The launcher
no longer deletes that cache on every start.

## TP=2 versus TP=3

- **TP=2:** vLLM 0.21 can use its ordinary custom all-reduce path after the pair
  passes real peer access and is fully connected.
- **TP=3:** NCCL can use validated P2P, but vLLM custom all-reduce rejects world
  size 3. The launcher reports this as an expected backend limitation rather
  than a P2P failure.
- Generate a separate profile for each exact device order you intend to use.

## NVLink versus PCIe BAR1

For a supported RTX 3090 pair, the upstream patch chooses NVLink when available
and PCIe BAR1 otherwise. No registry override is needed for normal operation.

To intentionally force PCIe instead of NVLink for a controlled test:

```bash
python3 install.py --force-pcie --yes
sudo reboot
```

That writes `RMForceP2PType=1`. Do not use it merely because a pair is
PCIe-only; auto-selection already uses PCIe when no NVLink path exists.

Topology labels such as `PIX`, `PXB`, `PHB`, and `SYS` describe a route; they do
not prove correctness. ACS, firmware, slot wiring, link width, risers, and the
root complex can change the result. The integrity gates decide whether the
selected pair is accepted.

## Runtime modes

### `validated` — default

Requires a current machine-bound profile:

```bash
VLLM_P2P_MODE=validated \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

### `auto` — diagnostics

Enables NCCL P2P and keeps vLLM's real checker active, but does not require a
saved profile:

```bash
VLLM_P2P_MODE=auto \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

### `shm` — explicit recovery mode

Disables NCCL P2P and vLLM custom all-reduce:

```bash
VLLM_P2P_MODE=shm \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

This is a correctness-preserving host-memory fallback, **not P2P**.

## Diagnostics

Full validation without writing a profile:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  ~/venvs/vllm/bin/python scripts/p2p_doctor.py validate \
    --venv ~/venvs/vllm
```

Check a saved profile:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  ~/venvs/vllm/bin/python scripts/p2p_doctor.py check-profile \
    --profile ~/.config/vllm/p2p-0-1.env
```

Integrity-gated bandwidth report:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  python3 scripts/p2p_bandwidth_bench.py --save p2p-report.txt
```

Topology and ACS evidence:

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
lspci -tv
sudo lspci -vv | grep -E '^[0-9a-f]|LnkCap:|LnkSta:|ACSCtl:'
```

The benchmark labels `cudaMemcpyPeer` correctly as an API throughput
measurement. It is never treated as proof of direct physical BAR1 traffic.

## Recovery sequence

When validation fails:

1. Confirm every NVIDIA driver layer is exactly `595.58.03`.
2. Confirm `modinfo -F license nvidia` reports the open module and DKMS installed
   it for the running kernel.
3. Confirm the machine rebooted after module installation.
4. Confirm the CPU-specific IOMMU argument and `iommu=pt` are present in
   `/proc/cmdline`.
5. Inspect slot wiring, link generation/width, Above 4G Decoding, Resizable BAR,
   ACS controls, and risers.
6. Re-run direct peer, CUDA IPC, and NCCL gates for the exact device order.
7. Use `VLLM_P2P_MODE=shm` only as a clearly labelled fallback.

Never hand-edit a profile fingerprint or set `VLLM_SKIP_P2P_CHECK=1` to hide a
failed pair.

## File map

| Path | Purpose |
|---|---|
| `install.py` | Exact CUDA-runtime front-end |
| `install_core.py` | Driver, DKMS, GRUB, Secure Boot, package lock, and venv workflow |
| `scripts/p2p_probe.cu` | Direct peer kernel read/write integrity test |
| `scripts/p2p_doctor.py` | Fail-closed transport/profile policy |
| `scripts/p2p_doctor_core.py` | Driver, boot, CUDA IPC, and NCCL workers |
| `scripts/post-reboot-test.sh` | Reboot-time validation/profile wrapper |
| `scripts/manage_vllm_safe_tp2.sh` | Numeric-device/profile front-end |
| `scripts/manage_vllm_safe_tp2_core.sh` | vLLM lifecycle implementation |
| `scripts/p2p_bandwidth_bench.py` | Integrity-gated benchmark/report |
| `tests/` | Offline syntax, policy, spawn, and pure-function tests |

## Security and maintenance

- Patched kernel modules are not an NVIDIA-supported configuration.
- Secure Boot requires an enrolled signing key; the installer refuses to assume
  that signing is configured.
- Check DKMS before every kernel reboot.
- Revalidate after kernel, driver, CUDA/PyTorch/vLLM/NCCL, firmware, slot, or GPU
  order changes.
- Do not run the validator while production workloads occupy the selected GPUs;
  it creates CUDA contexts, allocates memory, and performs peer writes.

See the numbered procedures in [`docs/`](docs/) for deeper troubleshooting and
lifecycle details.
