#!/usr/bin/env python3
"""Exact-runtime front-end for the consumer-GPU P2P installer.

The installer core owns the driver, DKMS, GRUB, Secure Boot, and package-lock
workflow.  This small boundary pins the Python runtime to the CUDA variant that
vLLM 0.21.0 actually publishes for Linux: CUDA 12.9 throughout PyTorch and the
precompiled vLLM wheel.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import install_core as _core

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu129"
PYTHON_PACKAGES = (
    "torch==2.11.0+cu129",
    "torchvision==0.26.0+cu129",
    "torchaudio==2.11.0+cu129",
    "vllm==0.21.0",
)

# The core resolves these names at execution time. Override them before exposing
# its public API or invoking main().
_core.TORCH_INDEX_URL = TORCH_INDEX_URL
_core.PYTHON_PACKAGES = PYTHON_PACKAGES

# Preserve the original module-level API for tests and operators importing the
# installer helpers.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_core, _name))


def install_vllm(runner: Any, venv_dir: pathlib.Path) -> None:
    """Install and verify one ABI-compatible CUDA 12.9 runtime transaction."""

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
                "import json, torch, torchvision, torchaudio, vllm; "
                "print(json.dumps({"
                "'torch': torch.__version__, "
                "'torch_cuda': torch.version.cuda, "
                "'torchvision': torchvision.__version__, "
                "'torchaudio': torchaudio.__version__, "
                "'vllm': vllm.__version__"
                "}, sort_keys=True))"
            ),
        ],
        capture=True,
    )
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if not lines:
        raise _core.InstallerError("vLLM runtime verification returned no output")
    try:
        observed = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise _core.InstallerError(
            f"Could not parse vLLM runtime verification output: {lines[-1]!r}"
        ) from exc

    expected = {
        "torch": "2.11.0+cu129",
        "torch_cuda": "12.9",
        "torchvision": "0.26.0+cu129",
        "torchaudio": "2.11.0+cu129",
        "vllm": "0.21.0",
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise _core.InstallerError(
            "vLLM/PyTorch CUDA ABI mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    print("vLLM runtime: " + json.dumps(observed, sort_keys=True))


_core.install_vllm = install_vllm


def main() -> int:
    return _core.main()


if __name__ == "__main__":
    raise SystemExit(main())
