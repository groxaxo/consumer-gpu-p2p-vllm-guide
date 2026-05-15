#!/usr/bin/env python3
"""
Consumer GPU P2P Driver Installer

Animated psychedelic installer for the patched NVIDIA P2P driver + vLLM setup.
Uses asciimatics for the full-screen animation; bootstraps it automatically if
it is not already installed.

Usage:
    python3 install.py [--yes] [--dry-run]
                       [--skip-driver] [--skip-grub] [--skip-vllm]
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


def _build_and_install_driver(driver_dir: Path, dry_run: bool) -> None:
    _set_msg("Compiling NVIDIA kernel module... (5-10 min)")
    _run(
        ["make", f"-j{os.cpu_count() or 1}", "modules"],
        cwd=driver_dir,
        dry_run=dry_run,
    )
    _set_msg("Installing NVIDIA kernel module...")
    _run_privileged(["make", "modules_install"], cwd=driver_dir, dry_run=dry_run)
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


def _install_vllm(venv_dir: Path, dry_run: bool) -> None:
    _set_msg("Creating Python virtual environment...")
    _run(["python3", "-m", "venv", str(venv_dir)], dry_run=dry_run)
    venv_python = venv_dir / "bin" / "python"
    _set_msg("Upgrading pip...")
    _run(
        [str(venv_python), "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel"],
        dry_run=dry_run,
    )
    _set_msg("Installing vLLM (large download, please wait)...")
    _run(
        [str(venv_python), "-m", "pip", "install", "vllm"],
        dry_run=dry_run,
    )


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
            "Compile and install the patched NVIDIA kernel module",
        ]
    if not args.skip_grub:
        planned += [
            "Add IOMMU/ASPM args to GRUB_CMDLINE_LINUX_DEFAULT",
            "Write /etc/modprobe.d/nvidia.conf (RMForceP2PType=0)",
            "Run update-grub + update-initramfs",
        ]
    if not args.skip_vllm:
        planned.append(f"Create venv at {args.venv_dir} and install vLLM")

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
