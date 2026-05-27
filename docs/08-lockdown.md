# Lockdown: Surviving `apt upgrade`

The patched 595.58.03 P2P driver is fragile because it lives outside the apt
package system. Three things can break it without warning:

1. **A new kernel** lands via `apt upgrade` → the patched modules don't exist
   for the new kernel → after reboot, the box drops to nouveau or no GPU.
2. **An apt-installed nvidia/libnvidia package** lands (often as a dependency
   of something else, e.g. `nvidia-container-toolkit`) → it drops a userspace
   `libnvidia-ml.so` that doesn't match our kernel module → `nvidia-smi` fails
   with "Driver/library version mismatch".
3. **A different `.run` file** is installed by a confused operator → ditto.

`install.py` now applies a four-layer lockdown to prevent all of these.

## What the installer does

| Layer | Path | Effect |
|---|---|---|
| `apt-mark hold` on the running kernel + headers + `linux-generic` meta-packages | (dpkg state) | `apt upgrade` cannot install a new kernel image. |
| `apt-mark hold` on `libnccl2`, `libnccl-dev` | (dpkg state) | Newer NCCL releases (which can quietly disable P2P codepaths) can't land. |
| `apt-mark hold` on every currently-installed `nvidia-*` / `libnvidia-*` package | (dpkg state) | Existing userspace packages are frozen at their current version. |
| Apt preferences pin | `/etc/apt/preferences.d/00-nvidia-p2p-pin` | Any **new** `nvidia-driver-*`, `libnvidia-compute-*`, `linux-modules-nvidia-*`, etc. is given `Pin-Priority: -1` → apt refuses to install it even as a dependency. |
| `.run` installer stash | `/opt/nvidia-p2p/NVIDIA-Linux-x86_64-595.58.03.run` | Survives `/home` cleanup. Re-run with `--no-kernel-modules` if userspace ever drifts. |
| Healthcheck binary | `/usr/local/sbin/p2p-healthcheck` | One-shot verifier: matches `modinfo nvidia` ↔ `/proc/driver/nvidia/version` ↔ `nvidia-smi`, then prints the `topo -p2p r` matrix. |

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
apt-mark showhold                          # all held packages
cat /etc/apt/preferences.d/00-nvidia-p2p-pin
ls -la /opt/nvidia-p2p/
```

## Recovering from a broken userspace

If `nvidia-smi` ever prints "Driver/library version mismatch" again (e.g. a
co-located CUDA installer overwrote the libs), the kernel module is almost
certainly still fine — just reinstall the matching userspace:

```bash
sudo /opt/nvidia-p2p/NVIDIA-Linux-x86_64-595.58.03.run \
     --no-kernel-modules --ui=none --silent
```

No reboot needed. Re-run `sudo p2p-healthcheck` to confirm.

## Intentionally upgrading later

```bash
# Free the locks
sudo rm /etc/apt/preferences.d/00-nvidia-p2p-pin
sudo apt-mark unhold $(apt-mark showhold | grep -E 'linux-|nvidia|libnccl')

# Then upgrade as normal
sudo apt update && sudo apt upgrade
```

After any kernel upgrade, you MUST rebuild the patched modules against the new
kernel before rebooting (see [02-patched-driver.md](02-patched-driver.md)),
otherwise the next boot loses the GPU driver.

## Skipping lockdown

If you have your own kernel/driver lifecycle management (DKMS, salt, ansible),
disable the lockdown step:

```bash
python3 install.py --skip-lockdown
```
