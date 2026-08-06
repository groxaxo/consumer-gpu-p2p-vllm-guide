# 8. Driver lifecycle and optional package lock

The patched kernel modules and NVIDIA userspace must remain version-aligned.
DKMS and apt package policy solve different parts of that problem.

## DKMS

The installer registers:

```text
nvidia-p2p/595.58.03
```

under:

```text
/usr/src/nvidia-p2p-595.58.03
```

Before rebooting after any kernel update:

```bash
dkms status
```

The new kernel must show the module as `installed`. A future kernel can break
the pinned source build; DKMS is automation, not a compatibility guarantee.

Every running-kernel change invalidates the machine-bound P2P profile even when
DKMS succeeds. Reboot and run:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/post-reboot-test.sh
```

## Optional NVIDIA apt lock

The installer no longer freezes NVIDIA and NCCL packages automatically. Global
holds concealed mismatches and prevented unrelated NCCL fixes.

After the exact driver stack is verified, opt in:

```bash
python3 install.py --lock-driver --yes
```

This:

1. writes `/etc/apt/preferences.d/00-nvidia-p2p-pin` for NVIDIA driver/module
   packages;
2. holds currently installed `nvidia-*`, `libnvidia-*`, and Xorg NVIDIA
   packages; and
3. deliberately leaves `libnccl*` upgradeable.

Inspect:

```bash
cat /etc/apt/preferences.d/00-nvidia-p2p-pin
apt-mark showhold | grep -E 'nvidia|libnvidia'
```

## Why NCCL is not pinned

The acceptance contract is exact-value collective correctness with P2P enabled,
not a permanent NCCL version. After any NCCL/PyTorch/vLLM upgrade:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/manage_vllm_safe_tp2.sh revalidate
```

If the transport or correctness changes, validation exposes it. Freezing NCCL
without evidence is not a substitute for testing.

## Runfile stash

When an official runfile is supplied, the installer copies it to:

```text
/opt/nvidia-p2p/NVIDIA-Linux-x86_64-595.58.03.run
```

Recover matching userspace:

```bash
sudo sh /opt/nvidia-p2p/NVIDIA-Linux-x86_64-595.58.03.run \
  --silent --ui=none --no-questions --accept-license --no-kernel-modules
sudo reboot
```

Do not run that command with a different patched kernel-module version.

## Intentionally changing driver versions

The patch is version-specific. To move away from 595.58.03:

```bash
# Remove apt policy/holds
sudo rm -f /etc/apt/preferences.d/00-nvidia-p2p-pin
held="$(apt-mark showhold | grep -E '^(nvidia|libnvidia|xserver-xorg-video-nvidia)' || true)"
if [[ -n "$held" ]]; then
  # shellcheck disable=SC2086
  sudo apt-mark unhold $held
fi

# Remove patched DKMS modules
sudo dkms remove -m nvidia-p2p -v 595.58.03 --all

# Remove old validation profiles and vLLM P2P caches only after stopping vLLM
rm -f ~/.config/vllm/consumer-p2p.env
rm -f ~/.cache/vllm/gpu_p2p_access_cache_for_*.json
```

Then install a fully matched new userspace/kernel stack. Do not reuse this
profile or assume the old patch applies to the new driver.

## Secure Boot

DKMS may create a signing key, but the kernel accepts it only when the key is
enrolled and trusted. Verify rather than assuming:

```bash
mokutil --sb-state
modinfo -F signer nvidia
journalctl -k -b | grep -Ei 'nvidia|signature|verification'
```

The installer requires an explicit `--allow-secure-boot` opt-in after the
operator has arranged signing/MOK enrollment.
