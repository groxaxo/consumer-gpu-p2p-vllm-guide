# 2. Patched NVIDIA Driver

## Why Patch?

The stock NVIDIA open kernel modules disable P2P on consumer GPUs. The
[aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules)
fork re-enables the P2P codepaths, which allows:

- `cudaDeviceCanAccessPeer()` to return true
- `cudaMemcpyPeer()` to function via DMA staging through host RAM (~6 GB/s)
- NCCL to detect the P2P state and auto-select SHM transport

Without the patch, CUDA reports no P2P capability and NCCL may not even
attempt multi-GPU communication.

> **TL;DR for most users**: `python3 install.py` handles everything below
> automatically, including DKMS registration so the patched driver survives
> kernel upgrades. The manual steps here are for reference or if you want to
> do things by hand.

## Option A — Automated (Recommended)

```bash
git clone https://github.com/groxaxo/consumer-gpu-p2p-vllm-guide.git
cd consumer-gpu-p2p-vllm-guide
python3 install.py
```

The installer clones the driver, builds it, registers with DKMS (so it
rebuilds on every kernel update), writes `modprobe` config, and locks apt so
a future `apt upgrade` can't silently replace the patched driver.

## Option B — Manual Build + DKMS

If you prefer to do it by hand, DKMS is still strongly recommended over a
bare `make modules_install`. Without DKMS, a new kernel will boot without the
patched modules and leave you with no GPU access until you rebuild manually.

```bash
# Prerequisites
sudo apt install dkms linux-headers-$(uname -r) build-essential

# Clone the patched driver
git clone -b 595.58.03-p2p https://github.com/aikitoria/open-gpu-kernel-modules.git
sudo cp -r open-gpu-kernel-modules /usr/src/nvidia-p2p-595.58.03

# Register with DKMS — auto-rebuilds on kernel upgrade
sudo dkms add -m nvidia-p2p -v 595.58.03
sudo dkms build -m nvidia-p2p -v 595.58.03
sudo dkms install -m nvidia-p2p -v 595.58.03

# Verify DKMS registration
dkms status
# nvidia-p2p/595.58.03, <your-kernel>: installed
```

## Option C — Bare make (not recommended)

Use this only if you cannot use DKMS. You will need to rebuild manually after
every kernel upgrade.

```bash
git clone -b 595.58.03-p2p https://github.com/aikitoria/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules

make -j$(nproc) modules       # build as regular user
sudo make modules_install     # install as root
sudo depmod -a
```

## Load the New Driver

After installation (or after reboot — the recommended path):

```bash
# Unload existing driver (if loaded)
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null || true

# Load patched driver
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm

# Verify the right version loaded
modinfo nvidia | grep version
# version:        595.58.03
```

Or simply reboot — the modules will auto-load on boot.

## What the Patch Does

The patch modifies the RM (Resource Manager) force P2P type handling:

- `RMForceP2PType=0` (default after patch): Auto-detect. The driver reports P2P
  capability based on PCIe topology. On Intel consumer hardware the DMA staging
  path (`cudaMemcpyPeer`) is reported as available; direct BAR P2P is not.
- `RMForceP2PType=1`: Force PCIe P2P. Originally intended for NVLink systems.
  **Do not use on Intel consumer platforms** — causes silent NaN data corruption.

The aikitoria fork re-enables the P2P codepaths that the stock open modules
disable for non-datacenter GPUs. It does not change any behavior on platforms
that already support BAR P2P (Threadripper, EPYC, Xeon).

## Verify the Patched Driver is Active

```bash
modinfo nvidia | grep -E "version|license"
# version:        595.58.03
# license:        Dual MIT/GPL
```

The `Dual MIT/GPL` license string confirms this is the open-gpu-kernel-modules
(patched) driver, not the proprietary NVIDIA driver. The version must be
`595.58.03`.

## Compatibility

| Component       | Version                                 |
|---              |---                                      |
| Driver          | 595.58.03-p2p (aikitoria fork)          |
| CUDA Toolkit    | 12.8                                    |
| vLLM            | 0.21.0 (tested from 0.19.0+)            |
| Python          | 3.9+                                    |
| Ubuntu          | 22.04 LTS                               |
| Kernel          | Any (DKMS rebuilds on upgrade)          |
