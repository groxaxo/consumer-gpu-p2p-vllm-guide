# 1. Prerequisites & Boot Configuration

## Why Boot Args Matter

The patched NVIDIA driver (`RMForceP2PType=0`) and CUDA P2P APIs require IOMMU
to be configured correctly. On Intel consumer platforms:

- `intel_iommu=on` enables the Intel IOMMU (VT-d), which the patched driver
  needs for DMA remapping of peer transactions.
- `iommu=pt` enables pass-through mode — devices use 1:1 DMA translations,
  avoiding performance overhead while keeping IOMMU enabled for the driver.
- `pci=noaer` disables PCIe Advanced Error Reporting printks (reduces log
  noise from the many PCIe errors this topology generates).
- `pcie_aspm=off` disables PCIe Active State Power Management — prevents
  link power-state transitions that can cause instability with many GPUs.

## Configure GRUB

Edit `/etc/default/grub`:

```bash
sudo nano /etc/default/grub
```

Change the `GRUB_CMDLINE_LINUX_DEFAULT` line:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on iommu=pt pci=noaer pcie_aspm=off"
```

Apply and reboot:

```bash
sudo update-grub
sudo reboot
```

## Verify After Reboot

```bash
cat /proc/cmdline
# Should contain: intel_iommu=on iommu=pt

cat /proc/driver/nvidia/params | grep RegistryDwords
# Should show: RMForceP2PType: 0
```

## Driver Registry Key

Create `/etc/modprobe.d/nvidia.conf`:

```
# RMForceP2PType=0 lets the driver auto-detect P2P capability.
# Setting to 1 forces PCIe P2P and can cause data corruption on Intel platforms
# where BAR-mapped P2P doesn't actually work.
options nvidia NVreg_RegistryDwords="RMForceP2PType=0"
```

Rebuild initramfs:

```bash
sudo update-initramfs -u
```

## Verify IOMMU

```bash
dmesg | grep -i iommu | head -5
# Should show: DMAR: IOMMU enabled, Intel-IOMMU enabled
```

## Next Step

Once GRUB and modprobe are configured and you've rebooted, continue to
[doc 02 — Patched NVIDIA Driver](02-patched-driver.md) to install the driver,
or run `python3 install.py` to have the installer handle everything from here.
