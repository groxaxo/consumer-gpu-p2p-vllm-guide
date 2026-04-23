#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_NAME = "Consumer GPU P2P & vLLM Guide"
DRIVER_REPO = "https://github.com/aikitoria/open-gpu-kernel-modules.git"
DRIVER_BRANCH = "595.58.03-p2p"
DEFAULT_DRIVER_DIR = Path("~/src/open-gpu-kernel-modules").expanduser()
DEFAULT_VENV_DIR = Path("~/venvs/vllm").expanduser()
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
NVIDIA_MODPROBE_CONF = (
    'options nvidia NVreg_RegistryDwords="RMForceP2PType=0"\n'
)


class InstallerError(RuntimeError):
    pass


def banner() -> None:
    print(
        """
=========================================
  Consumer GPU P2P & vLLM Installer
=========================================
"""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the consumer GPU P2P + vLLM setup from this repo."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run without prompting for confirmation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions that would be taken without making changes.",
    )
    parser.add_argument(
        "--skip-driver",
        action="store_true",
        help="Skip cloning, building, and installing the patched NVIDIA driver.",
    )
    parser.add_argument(
        "--skip-grub",
        action="store_true",
        help="Skip GRUB and modprobe configuration changes.",
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Skip creating the venv and installing vLLM.",
    )
    parser.add_argument(
        "--driver-dir",
        type=Path,
        default=DEFAULT_DRIVER_DIR,
        help=f"Driver checkout path (default: {DEFAULT_DRIVER_DIR}).",
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=DEFAULT_VENV_DIR,
        help=f"Python virtualenv path (default: {DEFAULT_VENV_DIR}).",
    )
    return parser.parse_args()


def require_linux() -> None:
    if platform.system() != "Linux":
        raise InstallerError("This installer only supports Linux.")


def require_tools(tools: list[str]) -> None:
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        raise InstallerError(
            "Missing required tools: " + ", ".join(missing) + "."
        )


def quote(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    location = f" (cwd: {cwd})" if cwd else ""
    print(f"$ {quote(cmd)}{location}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def run_privileged(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    if os.geteuid() == 0:
        run(cmd, cwd=cwd, env=env, dry_run=dry_run)
        return
    run(["sudo", *cmd], cwd=cwd, env=env, dry_run=dry_run)


def prompt_continue(summary: list[str], assume_yes: bool) -> None:
    print("This installer will:")
    for item in summary:
        print(f"  - {item}")
    print()

    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise InstallerError(
            "Interactive confirmation required; rerun with --yes for non-interactive use."
        )

    answer = input("Continue? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise InstallerError("Installation cancelled by user.")


def backup_file(path: Path, dry_run: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    print(f"Back up {path} -> {backup}")
    if not dry_run:
        shutil.copy2(path, backup)
    return backup


def install_apt_packages(dry_run: bool) -> None:
    run_privileged(["apt-get", "update"], dry_run=dry_run)
    kernel_release = subprocess.check_output(["uname", "-r"], text=True).strip()
    packages = REQUIRED_APT_PACKAGES + [f"linux-headers-{kernel_release}"]
    run_privileged(["apt-get", "install", "-y", *packages], dry_run=dry_run)


def ensure_driver_repo(driver_dir: Path, dry_run: bool) -> None:
    if driver_dir.exists():
        if not (driver_dir / ".git").exists():
            raise InstallerError(
                f"{driver_dir} exists but is not a git checkout."
            )
        run(
            ["git", "-C", str(driver_dir), "fetch", "--depth", "1", "origin", DRIVER_BRANCH],
            dry_run=dry_run,
        )
        run(
            [
                "git",
                "-C",
                str(driver_dir),
                "checkout",
                "-B",
                DRIVER_BRANCH,
                f"origin/{DRIVER_BRANCH}",
            ],
            dry_run=dry_run,
        )
        return

    driver_dir.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            DRIVER_BRANCH,
            DRIVER_REPO,
            str(driver_dir),
        ],
        dry_run=dry_run,
    )


def build_and_install_driver(driver_dir: Path, dry_run: bool) -> None:
    run(["make", f"-j{os.cpu_count() or 1}", "modules"], cwd=driver_dir, dry_run=dry_run)
    run_privileged(["make", "modules_install"], cwd=driver_dir, dry_run=dry_run)
    run_privileged(["depmod", "-a"], dry_run=dry_run)


def merge_args(existing: list[str], required: list[str]) -> list[str]:
    merged = list(existing)
    for arg in required:
        if arg not in merged:
            merged.append(arg)
    return merged


def update_grub_config(grub_path: Path, dry_run: bool) -> None:
    if not grub_path.exists():
        raise InstallerError(f"Missing GRUB config: {grub_path}")

    original = grub_path.read_text()
    pattern = re.compile(r'^(GRUB_CMDLINE_LINUX_DEFAULT=)(["\'])(.*?)(\2)$', re.M)
    match = pattern.search(original)
    if not match:
        raise InstallerError(
            f"Could not find GRUB_CMDLINE_LINUX_DEFAULT in {grub_path}"
        )

    current = match.group(3).strip()
    tokens = shlex.split(current) if current else []
    updated_tokens = merge_args(tokens, REQUIRED_GRUB_ARGS)
    updated_value = " ".join(updated_tokens)
    updated = pattern.sub(
        lambda m: f'{m.group(1)}"{updated_value}"',
        original,
        count=1,
    )

    if updated == original:
        print(f"{grub_path} already contains the required boot args.")
        return

    backup_file(grub_path, dry_run=dry_run)
    print(f"Updating {grub_path} with required boot args.")
    if not dry_run:
        grub_path.write_text(updated)


def update_nvidia_modprobe(modprobe_path: Path, dry_run: bool) -> None:
    if modprobe_path.exists():
        current = modprobe_path.read_text()
        if current == NVIDIA_MODPROBE_CONF:
            print(f"{modprobe_path} already contains the required config.")
            return
        backup_file(modprobe_path, dry_run=dry_run)

    modprobe_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {modprobe_path}.")
    if not dry_run:
        modprobe_path.write_text(NVIDIA_MODPROBE_CONF)


def update_boot_config(dry_run: bool) -> None:
    update_grub_config(Path("/etc/default/grub"), dry_run=dry_run)
    update_nvidia_modprobe(Path("/etc/modprobe.d/nvidia.conf"), dry_run=dry_run)
    run_privileged(["update-grub"], dry_run=dry_run)
    run_privileged(["update-initramfs", "-u"], dry_run=dry_run)


def install_vllm(venv_dir: Path, dry_run: bool) -> None:
    run(["python3", "-m", "venv", str(venv_dir)], dry_run=dry_run)
    venv_python = venv_dir / "bin" / "python"
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], dry_run=dry_run)
    run([str(venv_python), "-m", "pip", "install", "vllm"], dry_run=dry_run)


def make_scripts_executable(repo_root: Path, dry_run: bool) -> None:
    for script in repo_root.joinpath("scripts").glob("*.sh"):
        mode = script.stat().st_mode
        if mode & 0o111:
            continue
        print(f"Marking {script} executable.")
        if not dry_run:
            script.chmod(mode | 0o111)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    try:
        require_linux()
        require_tools(["git", "python3"])
        if not args.dry_run and os.geteuid() != 0:
            require_tools(["sudo"])

        banner()
        planned = ["install OS prerequisites"]
        if not args.skip_driver:
            planned.extend(
                [
                    "build and install the patched NVIDIA driver",
                    "write the required GRUB and NVIDIA module settings",
                ]
            )
        if not args.skip_vllm:
            planned.append("create a Python virtual environment and install vLLM")
        prompt_continue(
            planned,
            assume_yes=args.yes or args.dry_run,
        )

        if not args.dry_run and os.geteuid() != 0:
            run(["sudo", "-v"], dry_run=False)

        install_apt_packages(dry_run=args.dry_run)

        if not args.skip_driver:
            ensure_driver_repo(args.driver_dir.expanduser(), dry_run=args.dry_run)
            build_and_install_driver(args.driver_dir.expanduser(), dry_run=args.dry_run)

        if not args.skip_grub:
            update_boot_config(dry_run=args.dry_run)

        if not args.skip_vllm:
            install_vllm(args.venv_dir.expanduser(), dry_run=args.dry_run)

        make_scripts_executable(repo_root, dry_run=args.dry_run)

        print()
        print("Setup complete.")
        print("Reboot before launching vLLM.")
        print(
            "After reboot, run: scripts/post-reboot-test.sh or scripts/manage_vllm_safe_tp2.sh start"
        )
        return 0
    except (InstallerError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
