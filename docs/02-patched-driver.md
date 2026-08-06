# 2. Exact NVIDIA userspace and patched kernel modules

The patched open kernel module source does **not** contain NVIDIA's userspace
libraries. A working stack therefore has two independently installed layers
that must report the same version:

1. Official NVIDIA Linux userspace driver `595.58.03`
2. Patched NVIDIA open kernel modules built from the reviewed upstream revision

The previous installer built layer 2 without installing or requiring layer 1.
That was the central installation defect.

## Upstream source

- Repository: [`aikitoria/open-gpu-kernel-modules`](https://github.com/aikitoria/open-gpu-kernel-modules)
- Branch: `595.58.03-p2p`
- Revision pinned by this guide:
  `6dd6ba34a4abfb3761797b26102094b856b01edd`

The upstream README instructs operators to install the official 595.58.03
driver first, then replace its kernel modules with the patched build.

## Automated installation

When `nvidia-smi` already reports exactly `595.58.03`:

```bash
python3 install.py --lock-driver --yes
```

When another driver is active, supply the official runfile and explicitly opt
in to matching userspace installation:

```bash
python3 install.py \
  --driver-runfile ~/Downloads/NVIDIA-Linux-x86_64-595.58.03.run \
  --install-userspace \
  --lock-driver \
  --yes
```

The installer:

1. checks the runfile's own `--info` output for `595.58.03`;
2. installs userspace with `--no-kernel-modules`;
3. checks out the pinned patch revision;
4. registers the patched modules as `nvidia-p2p/595.58.03` in DKMS;
5. installs them into the current kernel and rebuilds initramfs;
6. requires a reboot before validation.

It refuses a mismatched driver instead of pinning a broken state.

## Manual outline

Install the official NVIDIA 595.58.03 userspace/kernel package using NVIDIA's
instructions. Then replace only the kernel modules:

```bash
git clone https://github.com/aikitoria/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules
git checkout --detach 6dd6ba34a4abfb3761797b26102094b856b01edd

make -j"$(nproc)" modules
sudo make modules_install
sudo depmod -a
sudo update-initramfs -u
sudo reboot
```

Bare `make modules_install` does not survive kernel upgrades. The automated
DKMS path is preferred.

## Verify all three version views

After reboot:

```bash
modinfo -F version nvidia
grep -E 'NVRM version|Kernel Module' /proc/driver/nvidia/version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

Every output must be exactly:

```text
595.58.03
```

Also confirm the open module license:

```bash
modinfo -F license nvidia
# Dual MIT/GPL
```

Then run the validator. Version alignment alone does not prove P2P correctness.

## Secure Boot

Patched modules must be signed by a key trusted by the running kernel. DKMS
signing behavior differs between distributions and local configurations. The
installer refuses to assume Secure Boot has been handled; it stops unless
Secure Boot is disabled or the operator passes `--allow-secure-boot` after
arranging signing and MOK enrollment.

Verify after reboot:

```bash
mokutil --sb-state
modinfo -F signer nvidia
journalctl -k -b | grep -Ei 'nvidia|verification|signature'
```

## Kernel upgrades

DKMS attempts to rebuild the pinned 595.58.03 source for each new kernel. A
future kernel may be incompatible with that source. Before rebooting into a new
kernel:

```bash
dkms status
```

The target kernel must show `nvidia-p2p/595.58.03 ... installed`. Even after a
successful rebuild, the machine-bound profile becomes stale because the kernel
changed; rerun `scripts/post-reboot-test.sh`.
