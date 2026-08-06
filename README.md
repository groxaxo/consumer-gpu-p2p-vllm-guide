# Consumer NVIDIA P2P for vLLM — validated Ampere setup

This repository installs and validates the patched NVIDIA open kernel modules
required to expose peer access on supported consumer GPUs, then prevents vLLM
from using that access until the data path has passed destructive integrity
tests.

The primary target is **RTX 3090 / Ampere on Linux**, including two- and
three-GPU workstations. The upstream patch also documents RTX 4090 and RTX 5090,
but this guide's strict default profile requires compute capability 8.x.

> [!CAUTION]
> This changes kernel modules and requires `iommu=pt`. IOMMU passthrough reduces
> DMA isolation and is unsafe for hosts that run untrusted devices or software.
> Keep a working kernel entry and remote recovery path before changing a
> production machine.

## What was wrong with the previous guide

The original version could report a successful installation while leaving P2P
unavailable or unsafe:

1. It built **595.58.03 patched kernel modules without installing or requiring
   the matching 595.58.03 NVIDIA userspace driver**. That creates a
   driver/library mismatch or leaves the stock module active.
2. Its production launcher exported `NCCL_P2P_DISABLE=1`, so NCCL P2P was
   disabled even though the guide claimed to enable it.
3. It exported `VLLM_SKIP_P2P_CHECK=1`. In vLLM this means **trust the driver's
   capability report**; it does not mean “use a cached validated result.” That
   is the unsafe choice for a patched consumer driver.
4. It treated `cudaDeviceCanAccessPeer()` and `cudaMemcpyPeer()` as proof of
   direct peer access. Neither proves that a GPU can correctly load from and
   store to another GPU's mapped memory.
5. It declared PCIe P2P impossible on all Intel consumer root complexes. The
   upstream patch explicitly supports RTX 3090 PCIe BAR1 P2P; actual success is
   topology-, firmware-, ACS-, and motherboard-dependent and must be measured.
6. It deleted vLLM's P2P cache on every launch and then skipped the test that
   recreates it.

This revision replaces those assumptions with fail-closed validation.

## The acceptance gates

A launcher profile is written only when all required gates pass:

| Gate | What is verified | Why it matters |
|---|---|---|
| Exact driver stack | `modinfo`, the loaded NVRM module, and `nvidia-smi` all report `595.58.03` | Open kernel modules do not include NVIDIA userspace libraries; both layers must match exactly. |
| Boot configuration | CPU-specific IOMMU enablement, `iommu=pt`, and active IOMMU groups | The upstream BAR1 path requires passthrough mappings. |
| Ampere identity | Every selected visible device reports compute capability 8.x | Prevents accidentally applying the Ampere profile to another GPU set. |
| Direct kernel integrity | Kernels on each GPU read and write exact `uint64_t` patterns in every peer GPU's memory | Catches silent corruption that capability queries and copy APIs miss. |
| vLLM CUDA IPC integrity | vLLM's own two-process `can_actually_p2p()` mechanism opens, mutates, and verifies peer memory in every direction | This is the path vLLM custom all-reduce depends on. |
| NCCL correctness | Multi-size all-reduce runs with `NCCL_P2P_DISABLE=0`, produces finite exact values, and records the reported transport when available | Confirms the distributed runtime is correct with P2P enabled. |

The resulting profile is bound to the selected device order, GPU UUIDs, PCI bus
IDs, driver version, and running kernel. A kernel, driver, GPU-order, slot, or
boot-argument change makes the profile stale and blocks launch until it is
regenerated.

## Supported stack

The patch currently pinned by this guide is:

- NVIDIA userspace driver: **595.58.03**
- Patched open kernel modules:
  [`aikitoria/open-gpu-kernel-modules`, branch `595.58.03-p2p`](https://github.com/aikitoria/open-gpu-kernel-modules/tree/595.58.03-p2p)
- Reviewed upstream revision:
  `6dd6ba34a4abfb3761797b26102094b856b01edd`
- Default Python runtime: PyTorch `2.11.0+cu128`, vLLM `0.21.0`
- Primary OS path: Ubuntu 22.04/24.04-class systems using GRUB and DKMS

Do not combine this kernel patch with a different NVIDIA userspace version.
When upstream publishes a patch for another driver, review and update all three
pins together: userspace version, source revision, and validation evidence.

## Fast path

### 1. Prepare the exact NVIDIA runfile

Download the official **NVIDIA Linux x86_64 595.58.03** runfile from NVIDIA.
The upstream patch README links the matching driver details page. This
repository deliberately does not mirror the proprietary runfile or guess its
URL/checksum.

If `nvidia-smi` already reports exactly `595.58.03`, the runfile argument can be
omitted. Otherwise:

```bash
git clone https://github.com/groxaxo/consumer-gpu-p2p-vllm-guide.git
cd consumer-gpu-p2p-vllm-guide

python3 install.py \
  --driver-runfile "$HOME/Downloads/NVIDIA-Linux-x86_64-595.58.03.run" \
  --install-userspace \
  --lock-driver \
  --yes
```

The installer uses the runfile only for matching userspace libraries
(`--no-kernel-modules`). It then builds the reviewed patched open modules via
DKMS. It does **not** pin NCCL.

When the exact userspace driver is already installed:

```bash
python3 install.py --lock-driver --yes
```

Review without changing the host:

```bash
python3 install.py --dry-run --yes
```

### 2. Reboot

```bash
sudo reboot
```

A reboot is mandatory. Do not validate against the old in-memory kernel module.

### 3. Validate the exact GPU set

For two RTX 3090s:

```bash
cd ~/consumer-gpu-p2p-vllm-guide
CUDA_VISIBLE_DEVICES=0,1 bash scripts/post-reboot-test.sh
```

For all three RTX 3090s:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/post-reboot-test.sh
```

Strict validation requires `nvcc` so it can compile the direct peer load/store
probe. When a CUDA toolkit is genuinely unavailable, the vLLM CUDA IPC test is
still mandatory and the kernel test can be made advisory:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/post-reboot-test.sh --allow-missing-nvcc
```

That weaker profile is suitable only when you understand the reduced evidence.

A successful run ends with:

```text
RESULT=PASS
Wrote validated profile: ~/.config/vllm/consumer-p2p.env
```

No profile is written on a failed required gate.

### 4. Launch vLLM with validated P2P

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

Despite its backward-compatible filename, the launcher derives TP size from the
visible GPU count. For three GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/manage_vllm_safe_tp2.sh start <model-id>
```

The validated runtime exports:

```bash
NCCL_P2P_DISABLE=0
NCCL_SHM_DISABLE=0
VLLM_SKIP_P2P_CHECK=0
```

`VLLM_SKIP_P2P_CHECK=0` is intentional. vLLM performs the real IPC test once
and stores its directed device-pair results under `~/.cache/vllm/`; later
starts reuse the cache. The launcher no longer deletes it.

## RTX 3090 topology: NVLink, PCIe, TP=2, and TP=3

The upstream patch chooses NVLink for an RTX 3090 pair when NVLink is present
and falls back to PCIe BAR1 otherwise. No `RMForceP2PType` parameter is required
for normal operation.

To intentionally force PCIe instead of NVLink for testing:

```bash
python3 install.py --force-pcie --yes
sudo reboot
```

That writes `RMForceP2PType=1`, matching the upstream test mode. Do not use the
flag merely because the GPUs are PCIe-only; auto-selection already uses PCIe
when no NVLink path exists.

For vLLM:

- **TP=2** can use vLLM's custom all-reduce after the CUDA IPC gate passes.
- **TP=3** can use NCCL P2P, but vLLM's custom all-reduce does not support world
  size 3. The launcher reports this distinction instead of calling TP=3 broken.
- A three-GPU machine can validate all three devices and still run a selected
  two-GPU pair by generating a separate profile for that exact
  `CUDA_VISIBLE_DEVICES` order.

Example profiles are intentionally separate:

```bash
# Pair profile
CUDA_VISIBLE_DEVICES=0,1 \
P2P_PROFILE_PATH=~/.config/vllm/p2p-0-1.env \
  bash scripts/post-reboot-test.sh

# Three-GPU profile
CUDA_VISIBLE_DEVICES=0,1,2 \
P2P_PROFILE_PATH=~/.config/vllm/p2p-0-1-2.env \
  bash scripts/post-reboot-test.sh
```

Use the same `P2P_PROFILE_PATH` when launching.

## Launcher modes

### `validated` — default

Requires a current machine-bound profile and refuses startup when it is missing
or stale:

```bash
VLLM_P2P_MODE=validated bash scripts/manage_vllm_safe_tp2.sh start <model>
```

### `auto` — diagnostics

Enables NCCL P2P and leaves vLLM's real checker enabled, but does not require a
saved profile:

```bash
VLLM_P2P_MODE=auto bash scripts/manage_vllm_safe_tp2.sh start <model>
```

Use this while diagnosing, not as the normal production path.

### `shm` — explicit recovery mode

Disables NCCL P2P and vLLM custom all-reduce:

```bash
VLLM_P2P_MODE=shm bash scripts/manage_vllm_safe_tp2.sh start <model>
```

This is a correct fallback when a motherboard cannot route peer traffic, but it
is **not P2P**. Keep it only after an A/B benchmark proves it is the best path
for that host.

## Diagnostics

### Full validator

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  ~/venvs/vllm/bin/python scripts/p2p_doctor.py validate \
    --venv ~/venvs/vllm \
    --write-profile
```

### Check an existing profile

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  ~/venvs/vllm/bin/python scripts/p2p_doctor.py check-profile
```

### Integrity-gated bandwidth report

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  python3 scripts/p2p_bandwidth_bench.py --save p2p-report.txt
```

The benchmark first runs direct peer kernel reads/writes. It then reports the
throughput of the `cudaMemcpyPeer` API and a pinned-host bounce baseline.
`cudaMemcpyPeer` throughput is never labelled as proof of physical BAR1
transport.

### NCCL-only correctness test

```bash
source ~/venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_P2P_DISABLE=0 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM \
  python scripts/test_nccl_tp2.py
```

### Topology and ACS

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
lspci -tv
sudo lspci -vv | grep -E '^[0-9a-f]|LnkCap:|LnkSta:|ACSCtl:'
```

IOMMU passthrough is required by the upstream patch. ACS may force traffic
upstream through the root complex and can destroy P2P performance or make a
pair unusable. Prefer a BIOS/firmware ACS control. Kernel ACS override patches
change isolation semantics and should not be applied blindly; the validator,
not a topology label, decides whether a pair is accepted.

## Why the vLLM check must remain enabled

Current vLLM distinguishes between a driver's report and actual peer access:

- With `VLLM_SKIP_P2P_CHECK=1`, vLLM calls
  `torch.cuda.can_device_access_peer()` and **trusts the result**.
- With `VLLM_SKIP_P2P_CHECK=0`, vLLM opens CUDA IPC memory across two processes,
  mutates it from the peer GPU, verifies both views, and caches the result.

See the upstream implementation:

- [`vllm/envs.py`](https://github.com/vllm-project/vllm/blob/main/vllm/envs.py)
- [`custom_all_reduce.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py)
- [`all_reduce_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/all_reduce_utils.py)

A patched consumer driver is exactly the case where the real test is worth
keeping.

## File map

| Path | Purpose |
|---|---|
| `install.py` | Exact-version, DKMS, GRUB, optional package-lock, and vLLM installer |
| `scripts/p2p_probe.cu` | Direct peer kernel read/write data-integrity test |
| `scripts/p2p_doctor.py` | Driver, boot, Ampere, kernel, CUDA IPC, NCCL, and profile gate |
| `scripts/post-reboot-test.sh` | Reboot-time validation/profile wrapper |
| `scripts/manage_vllm_safe_tp2.sh` | Validated launcher; supports TP count from visible devices |
| `scripts/p2p_bandwidth_bench.py` | Integrity-gated benchmark/report wrapper |
| `scripts/test_nccl_tp2.py` | Standalone NCCL correctness test for all visible GPUs |
| `tests/` | Offline syntax and pure-function tests; no GitHub Actions required |

## Recovery

When validation fails, do not force the profile to `validated`. Work through the
failed gate:

1. Confirm every driver layer is exactly `595.58.03`.
2. Confirm the patched `Dual MIT/GPL` open module is the module selected by
   `modinfo nvidia`.
3. Confirm the running kernel was rebooted after DKMS installation.
4. Confirm the CPU-specific IOMMU argument and `iommu=pt` are in
   `/proc/cmdline`.
5. Inspect slot topology, link width/speed, Above 4G Decoding, Resizable BAR,
   and ACS controls in firmware.
6. Re-run the direct kernel and CUDA IPC gates for the exact pair.
7. Use `VLLM_P2P_MODE=shm` only as a clearly labelled fallback.

Detailed procedures are in [`docs/`](docs/).

## Security and maintenance

- `iommu=pt` weakens DMA isolation.
- Patched kernel modules are not an NVIDIA-supported configuration.
- Secure Boot requires a correctly enrolled signing key; the installer refuses
  to assume this has been handled.
- DKMS rebuild success must be checked after every kernel upgrade.
- `--lock-driver` pins NVIDIA packages but deliberately leaves NCCL upgradeable.
- A kernel or driver change invalidates the profile and requires revalidation.
- Do not run the validator while production workloads are using the selected
  GPUs; the tests allocate memory, create CUDA contexts, and execute peer writes.

## License

The guide and its original scripts are released under the repository license.
The NVIDIA userspace driver and upstream NVIDIA/open-kernel-module sources keep
their own licenses.
