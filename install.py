#!/usr/bin/env python3
"""
Consumer GPU P2P Driver Installer

Animated psychedelic installer for the patched NVIDIA P2P driver + vLLM setup.
Uses asciimatics for the full-screen animation; bootstraps it automatically if
it is not already installed.

Usage:
    python3 install.py [--yes] [--dry-run]
                       [--skip-driver] [--skip-grub] [--skip-vllm]
                       [--skip-lockdown]
                       [--driver-dir PATH] [--venv-dir PATH]
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


# ─── constants ────────────────────────────────────────────────────────────────

INSTALL_NAME   = "GPU P2P"
DRIVER_REPO    = "https://github.com/aikitoria/open-gpu-kernel-modules.git"
DRIVER_BRANCH  = "595.58.03-p2p"
DEFAULT_DRIVER_DIR = Path("~/src/open-gpu-kernel-modules").expanduser()
DEFAULT_VENV_DIR   = Path("~/venvs/vllm").expanduser()
VLLM_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
VLLM_TORCH_PACKAGES = [
    "torch==2.11.0+cu128",
    "torchvision==0.26.0+cu128",
    "torchaudio==2.11.0+cu128",
]
VLLM_PACKAGE = "vllm==0.21.0"

REQUIRED_APT_PACKAGES = [
    "build-essential",
    "curl",
    "dkms",
    "git",
    "lsof",
    "python3",
    "python3-pip",
    "python3-venv",
]
REQUIRED_GRUB_ARGS = [
    "intel_iommu=on",
    "iommu=pt",
    "pci=noaer",
    "pcie_aspm=off",
]
NVIDIA_MODPROBE_CONF = 'options nvidia NVreg_RegistryDwords="RMForceP2PType=0"\n'

# Lockdown: prevent apt from upgrading kernels (would orphan our patched modules)
# or replacing the .run-installed userspace driver.
PATCHED_DRIVER_VERSION = "595.58.03"
NVIDIA_RUN_STASH_DIR = Path("/opt/nvidia-p2p")
APT_PIN_PATH = Path("/etc/apt/preferences.d/00-nvidia-p2p-pin")
APT_PIN_CONTENT = """# Block apt-installed NVIDIA drivers from overwriting the patched
# 595.58.03 .run install. Managed by install.py — delete this file
# if you intentionally want to upgrade the driver.

Package: nvidia-driver-* nvidia-dkms-* libnvidia-compute-* libnvidia-decode-* libnvidia-encode-* libnvidia-extra-* libnvidia-fbc1-* libnvidia-gl-* nvidia-kernel-source-* nvidia-kernel-common-* nvidia-utils-* xserver-xorg-video-nvidia-* linux-modules-nvidia-*
Pin: release *
Pin-Priority: -1
"""
HEALTHCHECK_PATH = Path("/usr/local/sbin/p2p-healthcheck")
HEALTHCHECK_SCRIPT = """#!/usr/bin/env bash
# Verifies the patched P2P driver kernel module and userspace match.
# Exit 0 = healthy, 1 = problem. Installed by install.py.
set -u
EXPECT="{version}"

KMOD_VER=$(modinfo nvidia 2>/dev/null | awk '/^version:/ {{print $2}}')
NVRM_VER=$(awk '/NVRM version/ {{for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\\.[0-9]+\\.[0-9]+$/) print $i}}' /proc/driver/nvidia/version 2>/dev/null | head -1)
SMI_OUT=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)

echo "kernel module version (modinfo): $KMOD_VER"
echo "loaded kernel module (NVRM):     $NVRM_VER"
echo "nvidia-smi userspace:            $SMI_OUT"

fail=0
[ "$KMOD_VER" = "$EXPECT" ] || {{ echo "FAIL: modinfo nvidia is not $EXPECT"; fail=1; }}
[ "$NVRM_VER" = "$EXPECT" ] || {{ echo "FAIL: loaded NVRM is not $EXPECT (reboot may be needed)"; fail=1; }}
echo "$SMI_OUT" | grep -q "$EXPECT" || {{ echo "FAIL: nvidia-smi mismatch (got: $SMI_OUT)"; fail=1; }}

if [ $fail -eq 0 ]; then
  echo ""
  echo "P2P matrix:"
  nvidia-smi topo -p2p r 2>&1 | sed -E 's/\\x1b\\[[0-9;]*m//g' | head -20
  echo ""
  echo "OK: P2P driver healthy ($EXPECT)"
fi
exit $fail
""".format(version=PATCHED_DRIVER_VERSION)


# ─── shared install state (written by worker thread, read by animation) ───────

_state_lock     = threading.Lock()
INSTALL_SUCCESS = False
INSTALL_FAILED  = False
INSTALL_MESSAGE = "Initializing..."
INSTALL_LOG: list[str] = []


def _set_msg(msg: str) -> None:
    global INSTALL_MESSAGE
    with _state_lock:
        INSTALL_MESSAGE = msg


def _log(line: str) -> None:
    with _state_lock:
        INSTALL_LOG.append(line)


def _fail(msg: str) -> None:
    global INSTALL_FAILED, INSTALL_MESSAGE
    with _state_lock:
        INSTALL_FAILED  = True
        INSTALL_MESSAGE = f"ERROR: {msg}"
        INSTALL_LOG.append(f"[ERROR] {msg}")


# ─── bootstrap asciimatics ────────────────────────────────────────────────────

def _ensure_asciimatics() -> None:
    try:
        import asciimatics  # noqa: F401
    except ImportError:
        print("[*] asciimatics not found — installing it now...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "asciimatics"],
            check=True,
        )
        # Re-exec so the fresh import succeeds
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _ensure_uv() -> list[str]:
    uv_cmd = [sys.executable, "-m", "uv"]
    try:
        subprocess.run(
            [*uv_cmd, "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[*] uv not found — installing it now...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "uv"],
            check=True,
        )
    return uv_cmd


# ─── error type ───────────────────────────────────────────────────────────────

class InstallerError(RuntimeError):
    pass


# ─── command runner (captures output into INSTALL_LOG) ───────────────────────

def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
) -> None:
    label = shlex.join(cmd) + (f"  [cwd: {cwd}]" if cwd else "")
    _log(f"$ {label}")
    if dry_run:
        return
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in (result.stdout or "").splitlines():
        _log(line)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout
        )


def _run_privileged(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
) -> None:
    if os.geteuid() == 0:
        _run(cmd, cwd=cwd, dry_run=dry_run)
    else:
        _run(["sudo", *cmd], cwd=cwd, dry_run=dry_run)


# ─── pre-flight checks ────────────────────────────────────────────────────────

def _require_linux() -> None:
    if platform.system() != "Linux":
        raise InstallerError("This installer only supports Linux.")


def _require_tools(tools: list[str]) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise InstallerError(
            "Missing required tools: " + ", ".join(missing)
        )


# ─── installation steps ───────────────────────────────────────────────────────

def _install_apt_packages(dry_run: bool) -> None:
    _set_msg("Installing OS prerequisites...")
    _run_privileged(["apt-get", "update", "-qq"], dry_run=dry_run)
    kernel_release = subprocess.check_output(["uname", "-r"], text=True).strip()
    packages = REQUIRED_APT_PACKAGES + [f"linux-headers-{kernel_release}"]
    _run_privileged(
        ["apt-get", "install", "-y", "-qq", *packages],
        dry_run=dry_run,
    )


def _ensure_driver_repo(driver_dir: Path, dry_run: bool) -> None:
    _set_msg("Cloning P2P driver source (aikitoria fork)...")
    if driver_dir.exists():
        if not (driver_dir / ".git").exists():
            raise InstallerError(
                f"{driver_dir} exists but is not a git checkout."
            )
        _run(
            ["git", "-C", str(driver_dir), "fetch", "--depth", "1",
             "origin", DRIVER_BRANCH],
            dry_run=dry_run,
        )
        _run(
            ["git", "-C", str(driver_dir), "checkout",
             "-B", DRIVER_BRANCH, f"origin/{DRIVER_BRANCH}"],
            dry_run=dry_run,
        )
    else:
        driver_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(
            ["git", "clone", "--depth", "1",
             "--branch", DRIVER_BRANCH,
             DRIVER_REPO, str(driver_dir)],
            dry_run=dry_run,
        )


DKMS_PACKAGE = "nvidia-p2p"
DKMS_SRC_DIR = Path(f"/usr/src/{DKMS_PACKAGE}-{PATCHED_DRIVER_VERSION}")
DKMS_CONF = """PACKAGE_NAME="{pkg}"
PACKAGE_VERSION="{ver}"
AUTOINSTALL="yes"

MAKE[0]="'make' -j$(nproc) NV_EXCLUDE_BUILD_MODULES='' KERNEL_UNAME=${{kernelver}} modules"

BUILT_MODULE_NAME[0]="nvidia"
BUILT_MODULE_LOCATION[0]="kernel-open"
DEST_MODULE_LOCATION[0]="/kernel/drivers/video"

BUILT_MODULE_NAME[1]="nvidia-uvm"
BUILT_MODULE_LOCATION[1]="kernel-open"
DEST_MODULE_LOCATION[1]="/kernel/drivers/video"

BUILT_MODULE_NAME[2]="nvidia-modeset"
BUILT_MODULE_LOCATION[2]="kernel-open"
DEST_MODULE_LOCATION[2]="/kernel/drivers/video"

BUILT_MODULE_NAME[3]="nvidia-drm"
BUILT_MODULE_LOCATION[3]="kernel-open"
DEST_MODULE_LOCATION[3]="/kernel/drivers/video"

BUILT_MODULE_NAME[4]="nvidia-peermem"
BUILT_MODULE_LOCATION[4]="kernel-open"
DEST_MODULE_LOCATION[4]="/kernel/drivers/video"
""".format(pkg=DKMS_PACKAGE, ver=PATCHED_DRIVER_VERSION)


def _dkms_registered() -> bool:
    """True if `dkms status` already shows our nvidia-p2p package."""
    result = subprocess.run(
        ["dkms", "status", "-m", DKMS_PACKAGE, "-v", PATCHED_DRIVER_VERSION],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return bool(result.stdout.strip())


def _build_and_install_driver(driver_dir: Path, dry_run: bool) -> None:
    """Register the patched driver source with DKMS so apt-installed kernel
    upgrades automatically rebuild it. DKMS is the only mechanism that makes
    `apt upgrade` safe for an out-of-tree NVIDIA driver."""
    _set_msg(f"Staging driver source at {DKMS_SRC_DIR}...")
    # Refresh the staged source from the checked-out git tree so we always
    # build the current branch contents.
    _run_privileged(["rm", "-rf", str(DKMS_SRC_DIR)], dry_run=dry_run)
    _run_privileged(["cp", "-a", str(driver_dir), str(DKMS_SRC_DIR)],
                    dry_run=dry_run)
    # Clean any prior in-tree build artifacts that would confuse DKMS.
    _run_privileged(["make", "-C", str(DKMS_SRC_DIR), "clean"], dry_run=dry_run)
    if not dry_run:
        tmp = Path("/tmp/nvidia-p2p-dkms.conf")
        tmp.write_text(DKMS_CONF)
        _run_privileged(
            ["install", "-o", "root", "-g", "root", "-m", "0644",
             str(tmp), str(DKMS_SRC_DIR / "dkms.conf")],
            dry_run=False,
        )
        tmp.unlink(missing_ok=True)

    if _dkms_registered():
        _set_msg("DKMS module already registered; removing for clean rebuild...")
        _run_privileged(
            ["dkms", "remove", "-m", DKMS_PACKAGE,
             "-v", PATCHED_DRIVER_VERSION, "--all"],
            dry_run=dry_run,
        )
    _set_msg(f"Registering {DKMS_PACKAGE} with DKMS...")
    _run_privileged(["dkms", "add", "-m", DKMS_PACKAGE,
                     "-v", PATCHED_DRIVER_VERSION], dry_run=dry_run)

    _set_msg("Building patched kernel modules via DKMS... (5-10 min)")
    _run_privileged(["dkms", "build", "-m", DKMS_PACKAGE,
                     "-v", PATCHED_DRIVER_VERSION], dry_run=dry_run)

    _set_msg("Installing patched kernel modules via DKMS...")
    _run_privileged(["dkms", "install", "-m", DKMS_PACKAGE,
                     "-v", PATCHED_DRIVER_VERSION, "--force"], dry_run=dry_run)
    _set_msg("Running depmod...")
    _run_privileged(["depmod", "-a"], dry_run=dry_run)


def _merge_args(existing: list[str], required: list[str]) -> list[str]:
    merged = list(existing)
    for arg in required:
        if arg not in merged:
            merged.append(arg)
    return merged


def _update_grub_config(grub_path: Path, dry_run: bool) -> None:
    if not grub_path.exists():
        raise InstallerError(f"Missing GRUB config: {grub_path}")
    original = grub_path.read_text()
    pattern = re.compile(
        r'^(GRUB_CMDLINE_LINUX_DEFAULT=)(["\'])(.*?)(\2)$', re.M
    )
    match = pattern.search(original)
    if not match:
        raise InstallerError(
            f"Could not find GRUB_CMDLINE_LINUX_DEFAULT in {grub_path}"
        )
    tokens = shlex.split(match.group(3).strip()) if match.group(3).strip() else []
    updated_tokens = _merge_args(tokens, REQUIRED_GRUB_ARGS)
    updated = pattern.sub(
        lambda m: f'{m.group(1)}"{" ".join(updated_tokens)}"',
        original,
        count=1,
    )
    if updated == original:
        _log(f"{grub_path} already contains the required boot args.")
        return
    # backup
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup = grub_path.with_name(f"{grub_path.name}.bak-{stamp}")
    _log(f"Backing up {grub_path} -> {backup}")
    if not dry_run:
        shutil.copy2(grub_path, backup)
        grub_path.write_text(updated)


def _update_nvidia_modprobe(modprobe_path: Path, dry_run: bool) -> None:
    if modprobe_path.exists():
        if modprobe_path.read_text() == NVIDIA_MODPROBE_CONF:
            _log(f"{modprobe_path} already contains the required config.")
            return
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup = modprobe_path.with_name(f"{modprobe_path.name}.bak-{stamp}")
        _log(f"Backing up {modprobe_path} -> {backup}")
        if not dry_run:
            shutil.copy2(modprobe_path, backup)
    modprobe_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Writing {modprobe_path}")
    if not dry_run:
        modprobe_path.write_text(NVIDIA_MODPROBE_CONF)


def _update_boot_config(dry_run: bool) -> None:
    _set_msg("Updating GRUB boot arguments...")
    _update_grub_config(Path("/etc/default/grub"), dry_run=dry_run)
    _set_msg("Writing /etc/modprobe.d/nvidia.conf...")
    _update_nvidia_modprobe(Path("/etc/modprobe.d/nvidia.conf"), dry_run=dry_run)
    _set_msg("Rebuilding GRUB config...")
    _run_privileged(["update-grub"], dry_run=dry_run)
    _set_msg("Rebuilding initramfs...")
    _run_privileged(["update-initramfs", "-u"], dry_run=dry_run)


def _pkg_installed(pkg: str) -> bool:
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${Status}", pkg],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return "install ok installed" in result.stdout


def _installed_nvidia_pkgs() -> list[str]:
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${Package}\\t${Status}\\n"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    out: list[str] = []
    for line in (result.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or "install ok installed" not in parts[1]:
            continue
        name = parts[0].split(":")[0]
        if re.match(r"^(nvidia|libnvidia|xserver-xorg-video-nvidia)", name):
            out.append(name)
    return out


def _apply_apt_holds(dry_run: bool) -> None:
    """Lock down apt's *userspace* nvidia/NCCL packages.

    We do NOT hold the kernel: DKMS rebuilds the patched modules on every
    kernel upgrade, so kernel upgrades are safe. We DO hold:
    - libnccl2/libnccl-dev — newer NCCL releases can disable P2P codepaths.
    - All currently-installed nvidia-*/libnvidia-* packages — apt-installed
      userspace would replace the .run-installed libs and mismatch our DKMS
      kernel module ("Driver/library version mismatch").
    """
    _set_msg("Applying apt holds (NCCL + nvidia userspace)...")
    for pkg in ("libnccl2", "libnccl-dev"):
        if _pkg_installed(pkg):
            _run_privileged(["apt-mark", "hold", pkg], dry_run=dry_run)
        else:
            _log(f"{pkg} is not installed; skipping hold.")

    nvidia_pkgs = _installed_nvidia_pkgs()
    if nvidia_pkgs:
        _run_privileged(["apt-mark", "hold", *nvidia_pkgs], dry_run=dry_run)
    else:
        _log("No nvidia-* apt packages installed; skipping nvidia holds.")


def _needs_root_write(path: Path, content: str) -> bool:
    try:
        return path.read_text() != content
    except (FileNotFoundError, PermissionError):
        return True


def _install_apt_pin(dry_run: bool) -> None:
    """Drop an apt preferences pin so apt-installed nvidia/kernel-module
    packages can never replace the .run-installed userspace driver."""
    _set_msg("Installing apt pin to block nvidia package installs...")
    if not _needs_root_write(APT_PIN_PATH, APT_PIN_CONTENT):
        _log(f"{APT_PIN_PATH} already up to date.")
        return
    if dry_run:
        _log(f"[dry-run] would write {APT_PIN_PATH}")
        return
    # Stage in /tmp then move via sudo (current user may not have write perms).
    tmp = Path("/tmp/00-nvidia-p2p-pin")
    tmp.write_text(APT_PIN_CONTENT)
    _run_privileged(["install", "-o", "root", "-g", "root", "-m", "0644",
                     str(tmp), str(APT_PIN_PATH)], dry_run=False)
    tmp.unlink(missing_ok=True)


def _install_healthcheck(dry_run: bool) -> None:
    """Install /usr/local/sbin/p2p-healthcheck so the user can verify the
    driver state at any time."""
    _set_msg("Installing p2p-healthcheck script...")
    if not _needs_root_write(HEALTHCHECK_PATH, HEALTHCHECK_SCRIPT):
        _log(f"{HEALTHCHECK_PATH} already up to date.")
        return
    if dry_run:
        _log(f"[dry-run] would write {HEALTHCHECK_PATH}")
        return
    tmp = Path("/tmp/p2p-healthcheck")
    tmp.write_text(HEALTHCHECK_SCRIPT)
    _run_privileged(["install", "-o", "root", "-g", "root", "-m", "0755",
                     str(tmp), str(HEALTHCHECK_PATH)], dry_run=False)
    tmp.unlink(missing_ok=True)


def _stash_run_installer(dry_run: bool) -> None:
    """Copy the .run installer to /opt/nvidia-p2p so it survives /home
    cleanups and can be re-applied after any accidental userspace overwrite."""
    candidates = [
        Path.home() / f"NVIDIA-Linux-x86_64-{PATCHED_DRIVER_VERSION}.run",
        Path(f"/tmp/NVIDIA-Linux-x86_64-{PATCHED_DRIVER_VERSION}.run"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    dest = NVIDIA_RUN_STASH_DIR / f"NVIDIA-Linux-x86_64-{PATCHED_DRIVER_VERSION}.run"
    if dest.exists():
        _log(f"{dest} already present; not re-stashing.")
        return
    if src is None:
        _log(f"No NVIDIA-Linux-x86_64-{PATCHED_DRIVER_VERSION}.run found in "
             f"~ or /tmp; skipping stash to {NVIDIA_RUN_STASH_DIR}.")
        return
    _set_msg(f"Stashing {src.name} to {NVIDIA_RUN_STASH_DIR}...")
    if dry_run:
        _log(f"[dry-run] would copy {src} -> {dest}")
        return
    _run_privileged(["mkdir", "-p", str(NVIDIA_RUN_STASH_DIR)], dry_run=False)
    _run_privileged(["cp", str(src), str(dest)], dry_run=False)
    _run_privileged(["chmod", "755", str(dest)], dry_run=False)
    _run_privileged(["chown", "root:root", str(dest)], dry_run=False)


def _apply_lockdown(dry_run: bool) -> None:
    """Lock the system so `apt upgrade`/`apt update` cannot break P2P."""
    _apply_apt_holds(dry_run=dry_run)
    _install_apt_pin(dry_run=dry_run)
    _install_healthcheck(dry_run=dry_run)
    _stash_run_installer(dry_run=dry_run)


def _verify_vllm_cuda12(venv_python: Path) -> None:
    result = subprocess.run(
        [
            str(venv_python),
            "-c",
            (
                "import torch, vllm; "
                "print(torch.__version__); "
                "print(torch.version.cuda or ''); "
                "print(vllm.__version__)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in (result.stdout or "").splitlines():
        _log(line)
    if result.returncode != 0:
        raise InstallerError("vLLM environment verification failed.")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        raise InstallerError("Could not determine the installed vLLM CUDA version.")
    cuda_version = lines[-2]
    try:
        cuda_major = int(cuda_version.split(".", 1)[0])
    except (TypeError, ValueError):
        raise InstallerError(f"Unexpected CUDA version string: {cuda_version!r}") from None
    if cuda_major >= 13:
        raise InstallerError(
            f"vLLM resolved to CUDA {cuda_version}. Expected CUDA 12.x; aborting."
        )


def _install_vllm(venv_dir: Path, dry_run: bool) -> None:
    uv_cmd = _ensure_uv()
    venv_python = venv_dir / "bin" / "python"
    if venv_python.exists():
        _set_msg(f"Reusing existing virtual environment at {venv_dir}...")
    else:
        _set_msg("Creating Python virtual environment...")
        _run([*uv_cmd, "venv", str(venv_dir)], dry_run=dry_run)
    _set_msg("Installing PyTorch CUDA 12.8...")
    _run(
        [
            *uv_cmd,
            "pip",
            "install",
            "--python",
            str(venv_python),
            "--index-url",
            VLLM_TORCH_INDEX_URL,
            *VLLM_TORCH_PACKAGES,
        ],
        dry_run=dry_run,
    )
    _set_msg("Installing vLLM (large download, please wait)...")
    _run(
        [*uv_cmd, "pip", "install", "--python", str(venv_python), VLLM_PACKAGE],
        dry_run=dry_run,
    )
    if not dry_run:
        _set_msg("Verifying vLLM resolved to CUDA 12.x...")
        _verify_vllm_cuda12(venv_python)


def _make_scripts_executable(repo_root: Path, dry_run: bool) -> None:
    for script in repo_root.joinpath("scripts").glob("*.sh"):
        mode = script.stat().st_mode
        if not (mode & 0o111):
            _log(f"chmod +x {script}")
            if not dry_run:
                script.chmod(mode | 0o111)


# ─── worker thread ────────────────────────────────────────────────────────────

def _run_installation(args: argparse.Namespace, repo_root: Path) -> None:
    global INSTALL_SUCCESS
    try:
        _install_apt_packages(dry_run=args.dry_run)

        if not args.skip_driver:
            _ensure_driver_repo(args.driver_dir.expanduser(), dry_run=args.dry_run)
            _build_and_install_driver(args.driver_dir.expanduser(), dry_run=args.dry_run)

        if not args.skip_grub:
            _update_boot_config(dry_run=args.dry_run)

        if not args.skip_lockdown:
            _apply_lockdown(dry_run=args.dry_run)

        if not args.skip_vllm:
            _install_vllm(args.venv_dir.expanduser(), dry_run=args.dry_run)

        _make_scripts_executable(repo_root, dry_run=args.dry_run)

        _set_msg("Done! Reboot to activate the driver.")
        INSTALL_SUCCESS = True

    except (InstallerError, subprocess.CalledProcessError, OSError) as exc:
        _fail(str(exc))
        time.sleep(6)   # keep the error message visible before animation exits
        INSTALL_SUCCESS = True  # signal the screen to stop


# ─── asciimatics animation ────────────────────────────────────────────────────

from asciimatics.effects import Print, Effect
from asciimatics.renderers import Plasma, FigletText, Rainbow
from asciimatics.scene import Scene
from asciimatics.screen import Screen
from asciimatics.exceptions import ResizeScreenError, StopApplication
from asciimatics.event import KeyboardEvent


class _CheckDone(Effect):
    """Stops the animation once the install thread has finished."""

    def __init__(self, screen: Screen, **kwargs):
        super().__init__(screen, **kwargs)

    def _update(self, frame_no: int) -> None:
        if INSTALL_SUCCESS:
            raise StopApplication("Install complete")

    @property
    def stop_frame(self) -> int:
        return 0

    def reset(self) -> None:
        pass

    def process_event(self, event):
        return event   # absorb all — no quit key during driver install


class _StatusText(Effect):
    """Shows INSTALL_MESSAGE + a last-log-line + ping-pong progress bar."""

    def __init__(self, screen: Screen, **kwargs):
        super().__init__(screen, **kwargs)

    def _update(self, frame_no: int) -> None:
        mid_y = self._screen.height // 2

        with _state_lock:
            msg     = INSTALL_MESSAGE
            failed  = INSTALL_FAILED
            last_log = INSTALL_LOG[-1] if INSTALL_LOG else ""

        colour = Screen.COLOUR_RED if failed else Screen.COLOUR_CYAN

        # ── status message ──────────────────────────────────────────────────
        status_str = f" {msg} "
        x = max(0, (self._screen.width - len(status_str)) // 2)
        y = mid_y + 5
        self._screen.print_at(
            " " * self._screen.width, 0, y, bg=Screen.COLOUR_BLACK
        )
        self._screen.print_at(
            status_str, x, y, colour=colour, bg=Screen.COLOUR_BLACK
        )

        # ── last log line (truncated) ────────────────────────────────────────
        max_log_w = self._screen.width - 4
        log_str = last_log[:max_log_w] if last_log else ""
        lx = max(0, (self._screen.width - len(log_str)) // 2)
        ly = y + 1
        self._screen.print_at(
            " " * self._screen.width, 0, ly, bg=Screen.COLOUR_BLACK
        )
        self._screen.print_at(
            log_str, lx, ly,
            colour=Screen.COLOUR_WHITE, attr=Screen.A_BOLD,
            bg=Screen.COLOUR_BLACK,
        )

        # ── ping-pong progress bar ───────────────────────────────────────────
        bar_w = min(50, self._screen.width - 4)
        if bar_w > 0:
            cycle = frame_no % (bar_w * 2)
            filled = cycle if cycle <= bar_w else bar_w * 2 - cycle
            bar = "[" + "=" * filled + " " * (bar_w - filled) + "]"
            bx = max(0, (self._screen.width - len(bar)) // 2)
            by = y + 3
            self._screen.print_at(
                " " * self._screen.width, 0, by, bg=Screen.COLOUR_BLACK
            )
            bar_colour = Screen.COLOUR_RED if failed else Screen.COLOUR_GREEN
            self._screen.print_at(
                bar, bx, by, colour=bar_colour, bg=Screen.COLOUR_BLACK
            )

    @property
    def stop_frame(self) -> int:
        return 0

    def reset(self) -> None:
        pass


def _screen_demo(screen: Screen) -> None:
    title_renderer = FigletText(INSTALL_NAME, font="big")
    title_x = max(0, (screen.width - title_renderer.max_width) // 2)

    effects = [
        # background: shifting plasma
        Print(
            screen,
            Plasma(screen.height, screen.width, screen.colours),
            0,
            speed=1,
            transparent=False,
        ),
        # title: rainbow figlet
        Print(
            screen,
            Rainbow(screen, title_renderer),
            y=(screen.height // 2) - 7,
            x=title_x,
            speed=1,
            transparent=True,
        ),
        # status text + log + progress bar
        _StatusText(screen),
        # completion watcher
        _CheckDone(screen),
    ]
    screen.play([Scene(effects, -1)], stop_on_resize=True, repeat=False)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Animated P2P driver + vLLM installer."
    )
    p.add_argument("--yes",          action="store_true",
                   help="Skip confirmation prompt.")
    p.add_argument("--dry-run",      action="store_true",
                   help="Print actions without making any changes.")
    p.add_argument("--skip-driver",  action="store_true",
                   help="Skip cloning, building, and installing the patched driver.")
    p.add_argument("--skip-grub",    action="store_true",
                   help="Skip GRUB and modprobe config changes.")
    p.add_argument("--skip-vllm",    action="store_true",
                   help="Skip venv creation and vLLM install.")
    p.add_argument("--skip-lockdown", action="store_true",
                   help="Skip apt holds, apt pin, healthcheck, and .run stashing.")
    p.add_argument("--driver-dir",   type=Path, default=DEFAULT_DRIVER_DIR,
                   help=f"Driver checkout path (default: {DEFAULT_DRIVER_DIR}).")
    p.add_argument("--venv-dir",     type=Path, default=DEFAULT_VENV_DIR,
                   help=f"venv path (default: {DEFAULT_VENV_DIR}).")
    return p.parse_args()


# ─── entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    _ensure_asciimatics()

    args     = _parse_args()
    repo_root = Path(__file__).resolve().parent

    # ── pre-flight (before the screen swallows stdout) ──────────────────────
    try:
        _require_linux()
        _require_tools(["git", "python3"])
        if not args.dry_run and os.geteuid() != 0:
            _require_tools(["sudo"])
    except InstallerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # ── plan summary + confirmation ─────────────────────────────────────────
    planned = ["Install OS prerequisites (apt)"]
    if not args.skip_driver:
        planned += [
            f"Clone {DRIVER_BRANCH} from {DRIVER_REPO}",
            f"Register driver source with DKMS ({DKMS_PACKAGE}-{PATCHED_DRIVER_VERSION})",
            "Build and install patched kernel modules via DKMS",
        ]
    if not args.skip_grub:
        planned += [
            "Add IOMMU/ASPM args to GRUB_CMDLINE_LINUX_DEFAULT",
            "Write /etc/modprobe.d/nvidia.conf (RMForceP2PType=0)",
            "Run update-grub + update-initramfs",
        ]
    if not args.skip_lockdown:
        planned += [
            "Hold nvidia userspace + libnccl2 packages (apt-mark hold)",
            f"Install apt pin at {APT_PIN_PATH}",
            f"Install healthcheck at {HEALTHCHECK_PATH}",
            f"Stash NVIDIA-Linux-x86_64-{PATCHED_DRIVER_VERSION}.run to "
            f"{NVIDIA_RUN_STASH_DIR} (if present)",
        ]
    if not args.skip_vllm:
        planned.append(
            f"Create venv at {args.venv_dir} and install CUDA 12.8 + vLLM via uv"
        )

    print()
    print("  Consumer GPU P2P Driver Installer")
    print("  ──────────────────────────────────")
    print("  This installer will:")
    for item in planned:
        print(f"    • {item}")
    print()

    if args.dry_run:
        print("  [DRY RUN — no changes will be made]")
        print()

    if not (args.yes or args.dry_run):
        if not sys.stdin.isatty():
            print(
                "ERROR: interactive confirmation required; use --yes for non-interactive.",
                file=sys.stderr,
            )
            return 1
        answer = input("  Continue? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("  Cancelled.")
            return 0
        print()

    # ── cache sudo credentials before taking over the screen ───────────────
    if not args.dry_run and os.geteuid() != 0:
        try:
            subprocess.run(["sudo", "-v"], check=True)
        except subprocess.CalledProcessError:
            print("ERROR: sudo authentication failed.", file=sys.stderr)
            return 1

    # ── start installation in background thread ─────────────────────────────
    worker = threading.Thread(
        target=_run_installation,
        args=(args, repo_root),
        daemon=True,
    )
    worker.start()

    # ── run the psychedelic animation in the main thread ───────────────────
    try:
        Screen.wrapper(_screen_demo)
    except ResizeScreenError:
        # user resized — re-enter the animation loop
        while not INSTALL_SUCCESS:
            try:
                Screen.wrapper(_screen_demo)
            except ResizeScreenError:
                pass
    except Exception:
        pass

    # ── wait for worker if animation exited early ──────────────────────────
    worker.join()

    # ── clear screen and print final report ────────────────────────────────
    print("\033[H\033[J", end="")   # clear

    if INSTALL_FAILED:
        print("=" * 60)
        print("  INSTALLATION FAILED")
        print("=" * 60)
        print()
        print("  Last 20 log lines:")
        for line in INSTALL_LOG[-20:]:
            print(f"    {line}")
        print()
        print("  Fix the error above and re-run install.py")
        return 1

    print("=" * 60)
    print("  GPU P2P DRIVER SETUP COMPLETE")
    print("=" * 60)
    print()
    if not args.skip_driver and not args.dry_run:
        print("  IMPORTANT: You must reboot for the driver to take effect.")
        print()
        print("  After reboot, validate with:")
        print("    bash scripts/post-reboot-test.sh")
        print()
        print("  Then launch vLLM:")
        print("    bash scripts/manage_vllm_safe_tp2.sh start")
    elif args.dry_run:
        print("  Dry run complete — no changes were made.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
