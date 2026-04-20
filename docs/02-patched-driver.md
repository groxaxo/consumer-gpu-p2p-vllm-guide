# 2. Patched NVIDIA Driver

## Why Patch?

The stock NVIDIA open kernel modules disable P2P on consumer GPUs. The
[aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules)
fork re-enables P2P capability reporting, which allows:

- `cudaDeviceCanAccessPeer()` to return true
- `cudaMemcpyPeer()` to function (DMA staging through host RAM)
- NCCL to probe and use SHM transport

Without the patch, CUDA reports no P2P capability and NCCL may not even
attempt multi-GPU communication.

## Build & Install

```bash
# Clone the patched driver
git clone -b 595.58.03-p2p https://github.com/aikitoria/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules

# Build kernel modules
make -j$(nproc) modules

# Install
sudo make modules_install
sudo depmod -a

# Verify
modinfo nvidia | grep version
# version:        595.58.03
```

## Load the New Driver

```bash
# Unload existing driver
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia

# Load new driver
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
```

Or reboot.

## What the Patch Does

The patch modifies the RM (Resource Manager) force P2P type handling:

- `RMForceP2PType=0` (default): Auto-detect. The driver reports P2P capability
  based on PCIe topology.
- `RMForceP2PType=1`: Force PCIe P2P. This was originally intended for NVLink
  systems where you want to force PCIe over NVLink. **Do not use this on Intel
  consumer platforms** — it causes NaN data corruption.

The aikitoria fork enables the P2P codepaths that the stock open modules
disable for non-datacenter GPUs.

## Driver Version Pinning

To prevent Ubuntu from overwriting the patched driver with an update:

```bash
sudo apt-mark hold nvidia-driver-535
# Or whichever package provides the stock driver
```

## Compatibility

| Component | Version |
|---|---|
| Driver | 595.58.03-p2p (aikitoria fork) |
| CUDA Toolkit | 12.x |
| vLLM | 0.19.0+ |
| Python | 3.10+ |
| Ubuntu | 22.04 LTS |
