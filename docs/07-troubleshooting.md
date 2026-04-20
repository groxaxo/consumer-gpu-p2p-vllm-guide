# 7. Troubleshooting

## vLLM OOM during startup

**Symptom**: Process dies with `CUDA out of memory` during "Capturing CUDA
graphs" or model loading.

**Fix**: Add `--enforce-eager` to disable CUDA graph capture, which requires
extra memory for profiling. Optionally reduce utilization as well:

```bash
--enforce-eager --gpu-memory-utilization 0.85
```

See [doc 04](04-vllm-setup.md#why---enforce-eager-is-hardcoded) for why this
is needed on near-full VRAM configurations.

---

## NCCL timeout or hang after model loads

**Symptom**: vLLM loads the model successfully, then hangs indefinitely on the
first inference request. NCCL prints timeout messages.

**Debug**:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL \
  CUDA_VISIBLE_DEVICES=0,1 ~/venvs/vllm/bin/vllm serve <model> \
    --tensor-parallel-size 2 --enforce-eager
```

Look for the transport line:

- `via SHM/direct/direct` — correct, SHM transport is working
- `via P2P/...` — NCCL thinks P2P works; on Intel consumer hardware this leads
  to hangs or corruption

**Fix**: Set `NCCL_P2P_DISABLE=1`. This tells NCCL to skip the P2P probe
entirely and go straight to SHM transport. The production launcher already
sets this by default.

If the hang persists with `NCCL_P2P_DISABLE=1`, also verify:
- `NCCL_SHM_DISABLE=0` (SHM transport must be enabled)
- Sufficient `/dev/shm` space: `df -h /dev/shm` (should be several GiB free)
- No other processes holding GPU memory: `nvidia-smi`

---

## NaN outputs / garbled model responses

**Symptom**: Model returns nonsensical text, all-zero outputs, or responses
full of `nan`.

**Cause 1 (most common)**: `RMForceP2PType=1` in modprobe config. This forces
BAR-mapped P2P which silently corrupts tensors on Intel consumer platforms.

**Fix**: Ensure `/etc/modprobe.d/nvidia.conf` contains:

```
options nvidia NVreg_RegistryDwords="RMForceP2PType=0"
```

Then rebuild initramfs and reboot:

```bash
sudo update-initramfs -u
sudo reboot
```

Verify after reboot:

```bash
cat /proc/driver/nvidia/params | grep RMForceP2PType
# Expected: RMForceP2PType: 0
```

**Cause 2**: Both `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1` are set.
This disables all NCCL transports, preventing all-reduce from completing.

**Fix**: Remove `NCCL_SHM_DISABLE=1` (or set it to `0`). The correct
production config has `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=0`.

---

## `cudaDeviceCanAccessPeer` returns false

**Symptom**: P2P capability tests report "peer access is not supported between
GPU0 and GPU1". NCCL may not initialize multi-GPU at all.

**Cause**: Stock NVIDIA driver (not the patched open modules). The stock driver
disables P2P reporting for consumer GPUs.

**Fix**: Install the patched driver and verify:

```bash
modinfo nvidia | grep -E "version|license"
# version:        595.58.03
# license:        Dual MIT/GPL
```

If the license shows `NVIDIA` (proprietary) instead of `Dual MIT/GPL` (open
modules), the patched driver is not loaded. See [doc 02](02-patched-driver.md).

---

## "Custom allreduce is disabled" message in vLLM logs

This is **expected and correct** on Intel consumer platforms. It means vLLM
ran its `can_actually_p2p()` IPC test, caught the CUDA IPC failure, and
delegated all-reduce to NCCL. Do not try to force-enable custom all-reduce —
NCCL SHM transport is used instead and works correctly.

---

## PCIe errors in dmesg

**Symptom**: `AER: Corrected error received`, `BadTLP`, or `RxErr` messages
appear in `dmesg` or `/var/log/syslog`.

**Cause**: Signal integrity issues with riser cables, cards running at
reduced PCIe link speeds, or simply the high PCIe traffic from DMA copies.
These errors typically do not affect operation when using SHM transport.

**Fix**: Add `pci=noaer` to GRUB kernel args to suppress the log spam:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on iommu=pt pci=noaer pcie_aspm=off"
```

If errors are severe and reproducible, reseat the GPU cards and risers.

---

## Model loads but produces wrong answers

**Debug**:

```bash
# Verify NCCL all-reduce produces correct values
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/test_nccl_tp2.py
```

If the all-reduce test fails or produces wrong values, the SHM path is broken.
Check:

1. Sufficient system RAM: `free -h` — NCCL SHM needs a few GiB
2. `/dev/shm` is not full: `df -h /dev/shm`
3. No other processes consuming GPU memory: `nvidia-smi`
4. `RMForceP2PType=0` is set (see NaN outputs section above)

---

## GRUB changes not taking effect after reboot

**Fix**:

```bash
# Regenerate GRUB config
sudo update-grub

# Reboot
sudo reboot

# After reboot — verify the args are present
cat /proc/cmdline
# Should contain: intel_iommu=on iommu=pt
```

If you're on UEFI and the changes still don't take effect, verify GRUB is
the active bootloader:

```bash
sudo efibootmgr
# GRUB should appear as the first active entry
```

---

## Driver not loading after a kernel upgrade

If you installed the driver without DKMS, the modules won't exist for the new
kernel. With the installer's DKMS setup this shouldn't happen, but if it does:

```bash
# Check DKMS status
dkms status
# Should show: nvidia-p2p/595.58.03, <new-kernel>: installed

# If missing, rebuild manually
sudo dkms build -m nvidia-p2p -v 595.58.03 -k $(uname -r)
sudo dkms install -m nvidia-p2p -v 595.58.03 -k $(uname -r)

# Reload
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null || true
sudo modprobe nvidia
modinfo nvidia | grep version
# version:        595.58.03
```

See [doc 08](08-lockdown.md) for the full lockdown setup that prevents this.

---

## "Driver/library version mismatch" from nvidia-smi

**Symptom**: `nvidia-smi` prints "Failed to initialize NVML: Driver/library
version mismatch".

**Cause**: A package manager update replaced the userspace NVIDIA libraries
with a version that doesn't match the patched 595.58.03 kernel module.

**Fix**: Reinstall the matching userspace without touching the kernel modules:

```bash
sudo /opt/nvidia-p2p/NVIDIA-Linux-x86_64-595.58.03.run \
     --no-kernel-modules --ui=none --silent
```

No reboot needed. Verify with `nvidia-smi` and `sudo p2p-healthcheck`.

If the `.run` file isn't at that path, see [doc 08](08-lockdown.md) for how
the installer stashes it.
