# Lockdown: Surviving `apt update && apt upgrade`

The patched 595.58.03 P2P driver lives outside the apt package system, so two
things can break it whenever you run `apt upgrade`:

1. **A new kernel** lands → the patched modules don't exist for the new
   kernel → after reboot, the box drops to nouveau or no GPU.
2. **An apt-installed nvidia/libnvidia package** lands (often as a dependency
   of something else, e.g. `nvidia-container-toolkit`) → it drops a userspace
   `libnvidia-ml.so` that doesn't match our kernel module → `nvidia-smi` fails
   with "Driver/library version mismatch".

`install.py` solves **(1)** with DKMS and **(2)** with apt holds + an apt
preferences pin. After the installer runs you can `apt update && apt upgrade`
freely.

## What the installer does

| Layer | Path | Effect |
|---|---|---|
| **DKMS registration** | `/usr/src/nvidia-p2p-595.58.03/`, `dkms status` | When apt installs any new kernel, the kernel postinst hook runs `dkms autoinstall` which rebuilds the patched `nvidia.ko`/`nvidia-uvm.ko`/`nvidia-modeset.ko`/`nvidia-drm.ko`/`nvidia-peermem.ko` against it before the next reboot. Modules are signed with the system MOK key so they load under Secure Boot. |
| `apt-mark hold` on `libnccl2`, `libnccl-dev` | (dpkg state) | Newer NCCL releases (which can quietly disable P2P codepaths) can't land. |
| `apt-mark hold` on every currently-installed `nvidia-*` / `libnvidia-*` package | (dpkg state) | Existing userspace packages are frozen at their current version. |
| Apt preferences pin | `/etc/apt/preferences.d/00-nvidia-p2p-pin` | Any **new** `nvidia-driver-*`, `libnvidia-compute-*`, `linux-modules-nvidia-*`, etc. is given `Pin-Priority: -1` → apt refuses to install it even as a dependency. |
| `.run` installer stash | `/opt/nvidia-p2p/NVIDIA-Linux-x86_64-595.58.03.run` | Survives `/home` cleanup. Re-run with `--no-kernel-modules` if userspace ever drifts. |
| Healthcheck binary | `/usr/local/sbin/p2p-healthcheck` | One-shot verifier: matches `modinfo nvidia` ↔ `/proc/driver/nvidia/version` ↔ `nvidia-smi`, then prints the `topo -p2p r` matrix. |

The kernel is **not** held — DKMS handles kernel upgrades automatically.

## Verifying

```bash
sudo p2p-healthcheck
```

Expected:

```
kernel module version (modinfo): 595.58.03
loaded kernel module (NVRM):     595.58.03
nvidia-smi userspace:            595.58.03
...
OK: P2P driver healthy (595.58.03)
```

## Inspecting the locks

```bash
dkms status                                # should show nvidia-p2p/595.58.03, <kernel>: installed
apt-mark showhold                          # all held packages
cat /etc/apt/preferences.d/00-nvidia-p2p-pin
ls -la /opt/nvidia-p2p/
```

## What happens on `apt upgrade`

1. apt downloads a new `linux-image-X.Y.Z-generic` and `linux-headers-X.Y.Z-generic`.
2. The kernel `postinst` hook fires `dkms autoinstall`.
3. DKMS rebuilds `nvidia-p2p-595.58.03` against the new kernel, signs the
   modules with your MOK key, and installs them to
   `/lib/modules/X.Y.Z-generic/updates/dkms/`.
4. `update-initramfs -u` is run automatically.
5. After reboot the new kernel boots with the patched P2P driver loaded.

If the DKMS build ever fails (e.g. NVIDIA source incompatible with a brand-new
kernel), the build error appears in the apt output and `dkms status` shows the
kernel without an "installed" entry. The OLD kernel still has its patched
modules, so you can keep using it until you address the build failure.

## Recovering from a broken userspace

If `nvidia-smi` ever prints "Driver/library version mismatch" again (e.g. a
co-located CUDA installer overwrote the libs), the kernel module is almost
certainly still fine — just reinstall the matching userspace:

```bash
sudo /opt/nvidia-p2p/NVIDIA-Linux-x86_64-595.58.03.run \
     --no-kernel-modules --ui=none --silent
```

No reboot needed. Re-run `sudo p2p-healthcheck` to confirm.

## Intentionally upgrading the *NVIDIA driver* later

You **don't** need to do this for routine `apt upgrade` — those are safe.
You only need to do this if you want to move to a different patched driver
version:

```bash
# Free the userspace locks
sudo rm /etc/apt/preferences.d/00-nvidia-p2p-pin
sudo apt-mark unhold $(apt-mark showhold | grep -E 'nvidia|libnccl')

# Remove the DKMS module for the old version
sudo dkms remove -m nvidia-p2p -v 595.58.03 --all

# Then install the new driver and re-run install.py to re-lock.
```

## Skipping lockdown

If you have your own kernel/driver lifecycle management (DKMS, salt, ansible),
disable the lockdown step:

```bash
python3 install.py --skip-lockdown
```
