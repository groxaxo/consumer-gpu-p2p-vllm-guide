# 7. Troubleshooting

## vLLM OOM during startup

**Symptom**: Process dies during "Capturing CUDA graphs" or model loading.

**Fix**: Add `--enforce-eager` or reduce `--gpu-memory-utilization`.

```
--gpu-memory-utilization 0.85 --enforce-eager
```

## NCCL timeout / hang

**Symptom**: vLLM hangs after loading model, NCCL prints timeout errors.

**Debug**:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL \
  vllm serve model --tensor-parallel-size 2
```

Check the transport selected:
- `via SHM/direct/direct` → correct
- `via P2P/...` → may hang if BAR P2P is broken

**Fix**: Ensure `NCCL_P2P_DISABLE=0` and `NCCL_SHM_DISABLE=0` are set.
NCCL should auto-detect and fall back to SHM.

## NaN outputs

**Symptom**: Model returns garbled text or empty responses.

**Cause 1**: `RMForceP2PType=1` in driver config. This forces BAR P2P which
produces NaN on Intel consumer platforms.

**Fix**: Set `RMForceP2PType=0`:

```
# /etc/modprobe.d/nvidia.conf
options nvidia NVreg_RegistryDwords="RMForceP2PType=0"
```

**Cause 2**: `NCCL_P2P_DISABLE=1` combined with `NCCL_SHM_DISABLE=1`. This
disables all NCCL transports.

**Fix**: Remove both, let NCCL auto-detect.

## `cudaDeviceCanAccessPeer` returns false

**Symptom**: `simpleP2P` reports "peer access is not supported".

**Cause**: Stock NVIDIA driver (not the patched open modules).

**Fix**: Install the aikitoria patched driver and verify with `modinfo nvidia`.

## vLLM "Custom allreduce is disabled" message

This is **expected and correct** on Intel consumer platforms. It means vLLM
detected that CUDA IPC doesn't work and fell back to NCCL. Do not try to
force-enable custom all-reduce.

## PCIe errors in dmesg

**Symptom**: `AER: Corrected error received`, `BadTLP`, `RxErr` messages.

**Cause**: Signal integrity issues with riser cables, many GPUs on limited
PCIe lanes, or cards running at Gen1 idle speed.

**Fix**: Add `pci=noaer` to kernel boot args to suppress warnings. Reseat
cards and risers if errors are severe. These errors typically don't affect
operation when using SHM transport.

## Model loads but produces wrong answers

**Debug**:

```bash
# Test NCCL directly
CUDA_VISIBLE_DEVICES=0,1 python test_nccl_tp2.py
```

If NCCL all-reduce produces wrong values, the SHM path is broken. Check:
1. Sufficient system RAM available
2. `/dev/shm` is not full (`df -h /dev/shm`)
3. No other processes consuming all GPU memory

## GRUB changes not taking effect

**Fix**:

```bash
sudo update-grub
sudo reboot
# After reboot, verify:
cat /proc/cmdline
```

If using UEFI, also check that GRUB is the default bootloader:

```bash
sudo efibootmgr
```

## Driver not loading after update

Ubuntu may have installed a stock NVIDIA driver that conflicts:

```bash
dpkg -l | grep nvidia-driver
sudo apt-mark hold nvidia-driver-*
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia
sudo modprobe nvidia
modinfo nvidia | grep version
```
