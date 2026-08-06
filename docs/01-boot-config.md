# 1. Boot configuration

The upstream P2P patch requires IOMMU passthrough. The required CPU-specific
switch is:

- Intel: `intel_iommu=on iommu=pt`
- AMD: `amd_iommu=on iommu=pt`

`iommu=pt` creates identity mappings for ordinary DMA devices. This is useful
for the peer BAR1 path, but it reduces DMA isolation. Do not use this setup on a
host that runs untrusted devices or workloads.

## Configure GRUB

Edit `/etc/default/grub` and append only the two required arguments to the
existing `GRUB_CMDLINE_LINUX_DEFAULT` value.

Intel example:

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on iommu=pt"
```

AMD example:

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=on iommu=pt"
```

Apply and reboot:

```bash
sudo update-grub
sudo reboot
```

The installer performs the same merge without discarding existing kernel
arguments:

```bash
python3 install.py --yes
```

## Arguments the guide no longer adds

The old revision added `pci=noaer` and `pcie_aspm=off` unconditionally. They are
not requirements of the upstream patch:

- `pci=noaer` hides Advanced Error Reporting. It can conceal the evidence needed
  to diagnose a bad riser, link, or peer transaction.
- `pcie_aspm=off` changes power management system-wide. Use it only after a
  reproducible link-state diagnosis, not as a P2P prerequisite.

## `RMForceP2PType`

Normal operation needs no registry override. The patched driver automatically
uses NVLink for a supported RTX 3090 pair when present and PCIe BAR1 otherwise.

`RMForceP2PType=1` is an upstream test mode that forces PCIe instead of NVLink.
Enable it only deliberately:

```bash
python3 install.py --force-pcie --yes
sudo reboot
```

The installer removes the old guide-managed `RMForceP2PType=0` file because
zero is already the driver default and the override adds no value.

## Verify after reboot

```bash
cat /proc/cmdline
find /sys/kernel/iommu_groups -mindepth 1 -maxdepth 1 -type d | wc -l
dmesg | grep -Ei 'DMAR|IOMMU' | head -30
```

Expected:

1. The correct CPU-specific IOMMU switch is present.
2. `iommu=pt` is present.
3. `/sys/kernel/iommu_groups` is populated.

Do not infer P2P success from those checks. Continue through the direct peer and
CUDA IPC tests in [03 — P2P diagnostics](03-p2p-diagnostics.md).

## Firmware checklist

Names differ by motherboard, but check:

- VT-d / AMD-Vi: enabled
- Above 4G Decoding: enabled
- Resizable BAR: test both firmware-supported states if peer mapping fails
- PCIe slot bifurcation: matches the physical card/riser layout
- ACS control: prefer firmware control; ACS can redirect peer traffic upstream
- PCIe generation: force Gen3 temporarily when diagnosing marginal Gen4 risers

A topology label such as `PHB` is not a correctness verdict. The validator's
exact peer load/store and CUDA IPC tests are the acceptance criteria.
