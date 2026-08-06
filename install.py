#!/usr/bin/env python3
"""Install the exact driver stack required by the consumer-GPU P2P patch.

This installer intentionally refuses to mix a patched 595.58.03 kernel module
with a different NVIDIA userspace driver.  The previous guide did exactly that,
which is the most common cause of an apparently successful installation that
fails after reboot.

The official NVIDIA 595.58.03 runfile is not mirrored or downloaded here.  Pass
an operator-supplied runfile with --driver-runfile and opt in to userspace
installation with --install-userspace, or preinstall the exact driver before
running this script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence

EXPECTED_DRIVER_VERSION = "595.58.03"
DRIVER_REPO = "https://github.com/aikitoria/open-gpu-kernel-modules.git"
DRIVER_BRANCH = "595.58.03-p2p"
# Pin the reviewed upstream revision instead of silently building a moving branch.
DRIVER_COMMIT = "6dd6ba34a4abfb3761797b26102094b856b01edd"
DKMS_PACKAGE = "nvidia-p2p"
DEFAULT_DRIVER_DIR = pathlib.Path.home() / "src" / "open-gpu-kernel-modules-p2p"
DEFAULT_VENV_DIR = pathlib.Path.home() / "venvs" / "vllm"
DEFAULT_RUNFILE_STASH = pathlib.Path("/opt/nvidia-p2p")
APT_PIN_PATH = pathlib.Path("/etc/apt/preferences.d/00-nvidia-p2p-pin")
FORCE_PCIE_CONFIG = pathlib.Path("/etc/modprobe.d/99-consumer-p2p-force-pcie.conf")
LEGACY_MODPROBE_CONFIG = pathlib.Path("/etc/modprobe.d/nvidia.conf")
LEGACY_MODPROBE_CONTENT = 'options nvidia NVreg_RegistryDwords="RMForceP2PType=0"\n'

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
PYTHON_PACKAGES = (
    "torch==2.11.0+cu128",
    "torchvision==0.26.0+cu128",
    "torchaudio==2.11.0+cu128",
    "vllm==0.21.0",
)

APT_PACKAGES = (
    "build-essential",
    "curl",
    "dkms",
    "git",
    "lsof",
    "mokutil",
    "pciutils",
    "python3",
    "python3-pip",
    "python3-venv",
)

APT_PIN_CONTENT = """# Managed by consumer-gpu-p2p-vllm-guide.
# The patched kernel modules require the matching 595.58.03 userspace stack.
# Remove this file before intentionally changing NVIDIA driver versions.

Package: nvidia-driver-* nvidia-dkms-* libnvidia-compute-* libnvidia-decode-* libnvidia-encode-* libnvidia-extra-* libnvidia-fbc1-* libnvidia-gl-* nvidia-kernel-source-* nvidia-kernel-common-* nvidia-utils-* xserver-xorg-video-nvidia-* linux-modules-nvidia-*
Pin: version *
Pin-Priority: -1
"""


class InstallerError(RuntimeError):
    """Actionable installation failure."""


class Runner:
    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run

    def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: pathlib.Path | None = None,
        privileged: bool = False,
        check: bool = True,
        capture: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = [os.fspath(item) for item in args]
        if privileged and os.geteuid() != 0:
            command = ["sudo", *command]
        suffix = f"  # cwd={cwd}" if cwd else ""
        print(f"$ {shlex.join(command)}{suffix}")
        if self.dry_run:
            return subprocess.CompletedProcess(command, 0, "", "")
        try:
            completed = subprocess.run(
                command,
                cwd=os.fspath(cwd) if cwd else None,
                text=True,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.STDOUT if capture else None,
                check=False,
            )
        except FileNotFoundError as exc:
            completed = subprocess.CompletedProcess(command, 127, str(exc) if capture else None, None)
        if check and completed.returncode != 0:
            output = completed.stdout or ""
            raise InstallerError(
                f"Command failed ({completed.returncode}): {shlex.join(command)}\n{output}"
            )
        return completed

    def write_root_file(
        self,
        path: pathlib.Path,
        content: str,
        *,
        mode: str = "0644",
        backup: bool = True,
    ) -> None:
        current = ""
        try:
            current = path.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError):
            pass
        if current == content:
            print(f"= {path} already up to date")
            return
        if backup and path.exists():
            stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = path.with_name(f"{path.name}.bak-{stamp}")
            self.run(["cp", "-a", path, backup_path], privileged=True)
        if self.dry_run:
            print(f"[dry-run] write {path}")
            return
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(content)
            temporary = pathlib.Path(handle.name)
        try:
            self.run(
                ["install", "-o", "root", "-g", "root", "-m", mode, temporary, path],
                privileged=True,
            )
        finally:
            temporary.unlink(missing_ok=True)


def require_linux() -> None:
    if platform.system() != "Linux":
        raise InstallerError("This installer supports Linux only.")


def require_commands(commands: Sequence[str]) -> None:
    missing = [command for command in commands if shutil.which(command) is None]
    if missing:
        raise InstallerError("Missing commands: " + ", ".join(missing))


def cpu_vendor() -> str:
    try:
        content = pathlib.Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError as exc:
        raise InstallerError(f"Cannot read /proc/cpuinfo: {exc}") from exc
    match = re.search(r"^vendor_id\s*:\s*(\S+)", content, re.MULTILINE)
    return match.group(1) if match else "unknown"


def iommu_argument() -> str:
    vendor = cpu_vendor()
    if vendor == "GenuineIntel":
        return "intel_iommu=on"
    if vendor == "AuthenticAMD":
        return "amd_iommu=on"
    raise InstallerError(f"Unsupported CPU vendor for automatic IOMMU setup: {vendor}")


def current_driver_version(runner: Runner) -> str:
    result = runner.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        return ""
    versions = {line.strip() for line in (result.stdout or "").splitlines() if line.strip()}
    return versions.pop() if len(versions) == 1 else "mixed"


def inspect_runfile(runner: Runner, runfile: pathlib.Path) -> None:
    if not runfile.is_file():
        raise InstallerError(f"Driver runfile does not exist: {runfile}")
    result = runner.run(["sh", runfile, "--info"], check=False, capture=True)
    output = result.stdout or ""
    if result.returncode != 0 or EXPECTED_DRIVER_VERSION not in output:
        raise InstallerError(
            f"Runfile did not identify itself as NVIDIA {EXPECTED_DRIVER_VERSION}:\n{output}"
        )


def install_matching_userspace(
    runner: Runner,
    *,
    runfile: pathlib.Path | None,
    install_userspace: bool,
) -> pathlib.Path | None:
    if runner.dry_run:
        print(f"[dry-run] require or install NVIDIA userspace {EXPECTED_DRIVER_VERSION}")
        return runfile
    observed = current_driver_version(runner)
    if observed == EXPECTED_DRIVER_VERSION:
        print(f"= NVIDIA userspace already matches {EXPECTED_DRIVER_VERSION}")
        if runfile:
            inspect_runfile(runner, runfile)
        return runfile

    if not install_userspace:
        raise InstallerError(
            f"NVIDIA userspace is {observed or 'not available'}, but the patch is "
            f"for {EXPECTED_DRIVER_VERSION}. Install the exact official driver first, "
            "or rerun with --driver-runfile PATH --install-userspace. The installer "
            "will not create a driver/library mismatch."
        )
    if runfile is None:
        raise InstallerError("--install-userspace requires --driver-runfile PATH")
    inspect_runfile(runner, runfile)
    print(
        "Installing matching NVIDIA userspace only. The patched kernel modules "
        "are installed separately through DKMS."
    )
    runner.run(
        [
            "sh",
            runfile,
            "--silent",
            "--ui=none",
            "--no-questions",
            "--accept-license",
            "--no-kernel-modules",
        ],
        privileged=True,
    )
    return runfile


def secure_boot_enabled(runner: Runner) -> bool:
    result = runner.run(["mokutil", "--sb-state"], check=False, capture=True)
    return result.returncode == 0 and "enabled" in (result.stdout or "").lower()


def install_prerequisites(runner: Runner) -> None:
    kernel = os.uname().release
    runner.run(["apt-get", "update"], privileged=True)
    runner.run(
        ["apt-get", "install", "-y", *APT_PACKAGES, f"linux-headers-{kernel}"],
        privileged=True,
    )


def ensure_driver_checkout(runner: Runner, driver_dir: pathlib.Path) -> None:
    driver_dir = driver_dir.expanduser()
    if driver_dir.exists() and not (driver_dir / ".git").is_dir():
        raise InstallerError(f"{driver_dir} exists but is not a Git checkout")
    if driver_dir.exists():
        runner.run(["git", "fetch", "origin", DRIVER_BRANCH], cwd=driver_dir)
    else:
        if not runner.dry_run:
            driver_dir.parent.mkdir(parents=True, exist_ok=True)
        runner.run(["git", "clone", DRIVER_REPO, driver_dir])
    runner.run(["git", "checkout", "--detach", DRIVER_COMMIT], cwd=driver_dir)
    result = runner.run(["git", "rev-parse", "HEAD"], cwd=driver_dir, capture=True)
    if not runner.dry_run and (result.stdout or "").strip() != DRIVER_COMMIT:
        raise InstallerError("Driver checkout did not resolve to the reviewed commit")


def dkms_config() -> str:
    return f'''PACKAGE_NAME="{DKMS_PACKAGE}"
PACKAGE_VERSION="{EXPECTED_DRIVER_VERSION}"
AUTOINSTALL="yes"

MAKE[0]="make -j$(nproc) NV_EXCLUDE_BUILD_MODULES='' KERNEL_UNAME=${{kernelver}} modules"
CLEAN="make clean"

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
'''


def dkms_is_registered(runner: Runner) -> bool:
    result = runner.run(
        ["dkms", "status", "-m", DKMS_PACKAGE, "-v", EXPECTED_DRIVER_VERSION],
        check=False,
        capture=True,
    )
    return bool((result.stdout or "").strip())


def install_patched_modules(runner: Runner, driver_dir: pathlib.Path) -> None:
    source_dir = pathlib.Path(f"/usr/src/{DKMS_PACKAGE}-{EXPECTED_DRIVER_VERSION}")
    runner.run(["rm", "-rf", source_dir], privileged=True)
    runner.run(["cp", "-a", driver_dir, source_dir], privileged=True)
    runner.write_root_file(source_dir / "dkms.conf", dkms_config(), backup=False)

    if dkms_is_registered(runner):
        runner.run(
            ["dkms", "remove", "-m", DKMS_PACKAGE, "-v", EXPECTED_DRIVER_VERSION, "--all"],
            privileged=True,
        )
    runner.run(
        ["dkms", "add", "-m", DKMS_PACKAGE, "-v", EXPECTED_DRIVER_VERSION],
        privileged=True,
    )
    runner.run(
        ["dkms", "build", "-m", DKMS_PACKAGE, "-v", EXPECTED_DRIVER_VERSION],
        privileged=True,
    )
    runner.run(
        [
            "dkms",
            "install",
            "-m",
            DKMS_PACKAGE,
            "-v",
            EXPECTED_DRIVER_VERSION,
            "--force",
        ],
        privileged=True,
    )
    runner.run(["depmod", "-a"], privileged=True)

    if not runner.dry_run:
        result = runner.run(["modinfo", "-F", "version", "nvidia"], capture=True)
        observed = (result.stdout or "").strip().splitlines()[0]
        if observed != EXPECTED_DRIVER_VERSION:
            raise InstallerError(
                f"Installed nvidia.ko reports {observed}, expected {EXPECTED_DRIVER_VERSION}"
            )


def merge_grub_arguments(original: str, required: Sequence[str]) -> str:
    pattern = re.compile(r'^(GRUB_CMDLINE_LINUX_DEFAULT=)(["\'])(.*?)(\2)$', re.MULTILINE)
    match = pattern.search(original)
    if not match:
        raise InstallerError("Could not find GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub")
    tokens = shlex.split(match.group(3)) if match.group(3).strip() else []
    for argument in required:
        if argument not in tokens:
            tokens.append(argument)
    replacement = f'{match.group(1)}"{" ".join(tokens)}"'
    return original[: match.start()] + replacement + original[match.end() :]


def remove_legacy_modprobe_override(runner: Runner) -> None:
    try:
        content = LEGACY_MODPROBE_CONFIG.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return
    if content == LEGACY_MODPROBE_CONTENT:
        print(f"Removing obsolete guide-managed override: {LEGACY_MODPROBE_CONFIG}")
        runner.run(["rm", "-f", LEGACY_MODPROBE_CONFIG], privileged=True)
    elif "RMForceP2PType" in content:
        raise InstallerError(
            f"{LEGACY_MODPROBE_CONFIG} contains a custom RMForceP2PType setting. "
            "Remove it manually or use --force-pcie intentionally."
        )


def configure_boot(runner: Runner, *, force_pcie: bool) -> None:
    grub_path = pathlib.Path("/etc/default/grub")
    try:
        original = grub_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InstallerError(f"Cannot read {grub_path}: {exc}") from exc
    updated = merge_grub_arguments(original, [iommu_argument(), "iommu=pt"])
    runner.write_root_file(grub_path, updated)

    remove_legacy_modprobe_override(runner)
    if force_pcie:
        runner.write_root_file(
            FORCE_PCIE_CONFIG,
            '# Force RTX 3090 pairs to use PCIe instead of NVLink for testing.\n'
            'options nvidia NVreg_RegistryDwords="RMForceP2PType=1"\n',
        )
    elif FORCE_PCIE_CONFIG.exists():
        runner.run(["rm", "-f", FORCE_PCIE_CONFIG], privileged=True)

    runner.run(["update-grub"], privileged=True)
    runner.run(["update-initramfs", "-u"], privileged=True)


def installed_nvidia_packages(runner: Runner) -> list[str]:
    result = runner.run(
        ["dpkg-query", "-W", "-f=${Package}\\t${Status}\\n"],
        check=False,
        capture=True,
    )
    packages: list[str] = []
    for line in (result.stdout or "").splitlines():
        name, separator, status = line.partition("\t")
        name = name.split(":", 1)[0]
        if separator and "install ok installed" in status and re.match(
            r"^(nvidia|libnvidia|xserver-xorg-video-nvidia)", name
        ):
            packages.append(name)
    return sorted(set(packages))


def lock_driver_packages(runner: Runner) -> None:
    runner.write_root_file(APT_PIN_PATH, APT_PIN_CONTENT)
    packages = installed_nvidia_packages(runner)
    if packages:
        runner.run(["apt-mark", "hold", *packages], privileged=True)
    print("NCCL packages were deliberately not pinned; transport validation, not version freezing, is the gate.")


def stash_runfile(runner: Runner, runfile: pathlib.Path | None) -> None:
    if runfile is None:
        return
    destination = DEFAULT_RUNFILE_STASH / runfile.name
    runner.run(["mkdir", "-p", DEFAULT_RUNFILE_STASH], privileged=True)
    runner.run(["cp", "-f", runfile, destination], privileged=True)
    runner.run(["chmod", "0755", destination], privileged=True)


def install_vllm(runner: Runner, venv_dir: pathlib.Path) -> None:
    venv_dir = venv_dir.expanduser()
    python = venv_dir / "bin" / "python"
    if not python.exists():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        runner.run([sys.executable, "-m", "venv", venv_dir])
    pip = [python, "-m", "pip"]
    runner.run([*pip, "install", "--upgrade", "pip", "setuptools", "wheel"])
    runner.run(
        [
            *pip,
            "install",
            "--extra-index-url",
            TORCH_INDEX_URL,
            *PYTHON_PACKAGES,
        ]
    )
    runner.run([*pip, "check"])
    if runner.dry_run:
        return
    result = runner.run(
        [
            python,
            "-c",
            (
                "import json, torch, vllm; "
                "print(json.dumps({'torch': torch.__version__, "
                "'cuda': torch.version.cuda, 'vllm': vllm.__version__}))"
            ),
        ],
        capture=True,
    )
    output = (result.stdout or "").strip()
    print(f"vLLM runtime: {output}")
    if '"cuda": "12.' not in output:
        raise InstallerError(f"vLLM did not resolve to a CUDA 12.x PyTorch wheel: {output}")


def make_scripts_executable(runner: Runner, repo_root: pathlib.Path) -> None:
    for path in (repo_root / "scripts").glob("*"):
        if path.suffix in {".sh", ".py"}:
            runner.run(["chmod", "+x", path])


def print_plan(args: argparse.Namespace) -> None:
    print("Consumer GPU P2P installer")
    print("==========================")
    print(f"Reviewed patch: {DRIVER_REPO}@{DRIVER_COMMIT}")
    print(f"Required NVIDIA userspace: {EXPECTED_DRIVER_VERSION}")
    print(f"CPU IOMMU argument: {iommu_argument()}")
    print("\nActions:")
    print("  - install build, DKMS, diagnostic, and Python prerequisites")
    if args.install_userspace:
        print(f"  - install userspace from {args.driver_runfile} without kernel modules")
    else:
        print("  - require an already-matching NVIDIA userspace stack")
    if not args.skip_driver_patch:
        print("  - build and install reviewed patched open kernel modules through DKMS")
    if not args.skip_boot_config:
        print("  - add only the CPU-specific IOMMU switch and iommu=pt to GRUB")
        print(f"  - {'force PCIe instead of NVLink' if args.force_pcie else 'use upstream P2P auto-selection'}")
    if args.lock_driver:
        print("  - pin NVIDIA packages (NCCL remains upgradeable)")
    if not args.skip_vllm:
        print(f"  - install vLLM in {args.venv_dir}")
    print("  - require a reboot, then destructive data-integrity validation")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Skip confirmation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--driver-runfile",
        type=pathlib.Path,
        help=f"Official NVIDIA-Linux-x86_64-{EXPECTED_DRIVER_VERSION}.run path.",
    )
    parser.add_argument(
        "--install-userspace",
        action="store_true",
        help="Install matching userspace from --driver-runfile (explicit opt-in).",
    )
    parser.add_argument(
        "--driver-dir", type=pathlib.Path, default=DEFAULT_DRIVER_DIR
    )
    parser.add_argument("--venv-dir", type=pathlib.Path, default=DEFAULT_VENV_DIR)
    parser.add_argument("--skip-driver-patch", action="store_true")
    parser.add_argument("--skip-boot-config", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument(
        "--force-pcie",
        action="store_true",
        help="Set RMForceP2PType=1 to force RTX 3090 PCIe instead of NVLink. Not needed for PCIe-only pairs.",
    )
    parser.add_argument(
        "--lock-driver",
        action="store_true",
        help="Pin NVIDIA apt packages after installation. NCCL is not pinned.",
    )
    parser.add_argument(
        "--allow-secure-boot",
        action="store_true",
        help="Proceed with Secure Boot enabled after you have arranged DKMS module signing/MOK enrollment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = Runner(dry_run=args.dry_run)
    repo_root = pathlib.Path(__file__).resolve().parent
    try:
        require_linux()
        require_commands(["git", "python3"] + ([] if os.geteuid() == 0 else ["sudo"]))
        print_plan(args)
        if not (args.yes or args.dry_run):
            if not sys.stdin.isatty():
                raise InstallerError("Interactive confirmation required; pass --yes.")
            answer = input("Continue? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Cancelled.")
                return 0
        if not args.dry_run and os.geteuid() != 0:
            runner.run(["sudo", "-v"])

        install_prerequisites(runner)
        if secure_boot_enabled(runner) and not args.allow_secure_boot:
            raise InstallerError(
                "Secure Boot is enabled. Disable it, or enroll a DKMS signing key and "
                "rerun with --allow-secure-boot. Unsigned patched modules will not load."
            )

        runfile = args.driver_runfile.expanduser().resolve() if args.driver_runfile else None
        validated_runfile = install_matching_userspace(
            runner,
            runfile=runfile,
            install_userspace=args.install_userspace,
        )

        if not args.skip_driver_patch:
            ensure_driver_checkout(runner, args.driver_dir)
            install_patched_modules(runner, args.driver_dir.expanduser())
        if not args.skip_boot_config:
            configure_boot(runner, force_pcie=args.force_pcie)
        if args.lock_driver:
            lock_driver_packages(runner)
        stash_runfile(runner, validated_runfile)
        if not args.skip_vllm:
            install_vllm(runner, args.venv_dir)
        make_scripts_executable(runner, repo_root)

    except (InstallerError, OSError) as exc:
        print(f"\nINSTALLATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("\nInstallation staged successfully.")
    if args.dry_run:
        print("Dry run only; no changes were made.")
        return 0
    print("\nReboot before validation:")
    print("  sudo reboot")
    print("\nAfter reboot, validate the exact GPU set and write its profile:")
    print("  CUDA_VISIBLE_DEVICES=0,1 bash scripts/post-reboot-test.sh")
    print("\nOnly after RESULT=PASS should you launch vLLM:")
    print("  CUDA_VISIBLE_DEVICES=0,1 bash scripts/manage_vllm_safe_tp2.sh start <model>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
