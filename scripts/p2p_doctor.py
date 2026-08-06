#!/usr/bin/env python3
"""Fail-closed front-end for the reviewed consumer-GPU P2P validator core.

The core contains the CUDA, vLLM IPC, and NCCL worker implementations.  This
front-end adds the policy boundary that must remain small and auditable:

* numeric physical GPU ordering for vLLM 0.21 custom all-reduce;
* strict NCCL transport classification (no mixed SHM/NET profile);
* machine-bound, owner-only profiles with an allowlisted grammar; and
* profile generation only after confirmed all-channel P2P.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import pathlib
import re
import shlex
import stat
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

CORE_PATH = pathlib.Path(__file__).with_name("p2p_doctor_core.py")
PROFILE_KEYS = frozenset(
    {
        "P2P_PROFILE_VERSION",
        "P2P_PROFILE_STATUS",
        "P2P_PROFILE_DEVICES",
        "P2P_PROFILE_DRIVER_VERSION",
        "P2P_PROFILE_KERNEL",
        "P2P_PROFILE_GPU_FINGERPRINT",
        "P2P_PROFILE_GPU_UUIDS",
        "P2P_PROFILE_NCCL_TRANSPORT_OBSERVATION",
        "P2P_PROFILE_CREATED_UTC",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "VLLM_SKIP_P2P_CHECK",
    }
)
SAFE_VALUE = re.compile(r"[A-Za-z0-9_.,:+/@=-]+")


def _load_core() -> Any:
    if not CORE_PATH.is_file():
        raise RuntimeError(f"P2P validator core is missing: {CORE_PATH}")
    spec = importlib.util.spec_from_file_location("consumer_p2p_doctor_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load P2P validator core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = _load_core()


def normalize_devices(value: str) -> str:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        raise ValueError("At least two comma-separated physical GPU indices are required.")
    if any(re.fullmatch(r"[0-9]+", item) is None for item in devices):
        raise ValueError(
            "Use numeric physical GPU indices in CUDA_VISIBLE_DEVICES (for "
            "example 0,1 or 0,1,2). UUID tokens are rejected because pinned "
            "vLLM 0.21.0 custom all-reduce parses this value as integers."
        )
    normalized = [str(int(item, 10)) for item in devices]
    if len(set(normalized)) != len(normalized):
        raise ValueError("CUDA device indices must be unique.")
    return ",".join(normalized)


def strict_nccl_check(
    python: pathlib.Path,
    env: dict[str, str],
    *,
    timeout: int,
    script_path: pathlib.Path,
):
    child_env = env.copy()
    child_env.update(
        {
            "NCCL_P2P_DISABLE": "0",
            "NCCL_SHM_DISABLE": "0",
            "NCCL_IB_DISABLE": child_env.get("NCCL_IB_DISABLE", "1"),
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,P2P,SHM",
            "PYTHONUNBUFFERED": "1",
        }
    )
    result = core.run_command(
        [python, script_path, "_nccl-test"], env=child_env, timeout=timeout
    )
    marker = next(
        (line for line in result.stdout.splitlines() if line.startswith("P2P_NCCL_JSON=")),
        "",
    )
    combined = result.combined
    if result.returncode != 0 or not marker:
        return (
            core.CheckResult("NCCL all-reduce with P2P enabled", False, combined[-8000:]),
            "unknown",
        )

    payload = json.loads(marker.split("=", 1)[1])
    passed = payload.get("passed") is True
    upper = combined.upper()
    p2p_lines = [
        line
        for line in upper.splitlines()
        if re.search(r"\bVIA\s+P2P(?:/|\b)", line)
        or "P2P/IPC" in line
        or "P2P/CUMEM" in line
    ]
    fallback_lines = [
        line
        for line in upper.splitlines()
        if re.search(r"\bVIA\s+(?:SHM|NET)(?:/|\b)", line)
    ]
    if p2p_lines and fallback_lines:
        transport = "mixed-observed"
    elif p2p_lines:
        transport = "p2p-confirmed"
    elif fallback_lines:
        transport = "fallback-observed"
    else:
        transport = "not-reported"
    detail = (
        json.dumps(payload, separators=(",", ":"))
        + f"; transport={transport}; p2p_channels={len(p2p_lines)}"
        + f"; fallback_channels={len(fallback_lines)}"
    )
    return core.CheckResult("NCCL all-reduce with P2P enabled", passed, detail), transport


def profile_values(
    *,
    devices: str,
    inventory: list[dict[str, str]],
    fingerprint: str,
    transport: str,
) -> dict[str, str]:
    if transport != "p2p-confirmed":
        raise ValueError(
            f"Refusing to create a validated P2P profile from NCCL transport {transport}."
        )
    return {
        "P2P_PROFILE_VERSION": core.PROFILE_VERSION,
        "P2P_PROFILE_STATUS": "validated",
        "P2P_PROFILE_DEVICES": devices,
        "P2P_PROFILE_DRIVER_VERSION": core.EXPECTED_DRIVER_VERSION,
        "P2P_PROFILE_KERNEL": os.uname().release,
        "P2P_PROFILE_GPU_FINGERPRINT": fingerprint,
        "P2P_PROFILE_GPU_UUIDS": ",".join(gpu["uuid"] for gpu in inventory),
        "P2P_PROFILE_NCCL_TRANSPORT_OBSERVATION": "p2p-confirmed",
        "P2P_PROFILE_CREATED_UTC": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "NCCL_P2P_DISABLE": "0",
        "NCCL_SHM_DISABLE": "0",
        "VLLM_SKIP_P2P_CHECK": "0",
    }


def write_profile(path: pathlib.Path, values: dict[str, str]) -> None:
    extra = sorted(set(values) - PROFILE_KEYS)
    if extra:
        raise RuntimeError(f"Invalid profile keys: {extra}")
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lines = [
        "# Generated only after destructive consumer-GPU P2P integrity checks.",
        "# Machine-bound: do not copy to another kernel, driver, GPU order, or host.",
    ]
    for key in sorted(values):
        value = values[key]
        if SAFE_VALUE.fullmatch(value) is None:
            raise RuntimeError(f"Unsafe profile value for {key}: {value!r}")
        lines.append(f"export {key}={shlex.quote(value)}")
    payload = "\n".join(lines) + "\n"

    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = pathlib.Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"\nWrote validated profile: {path}")


def validate_profile_file_security(path: pathlib.Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Profile must not be a symbolic link: {path}")
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Validated profile not found: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"Profile is not a regular file: {path}")
    if metadata.st_uid != os.getuid():
        raise RuntimeError(
            f"Profile owner UID {metadata.st_uid} does not match current UID {os.getuid()}."
        )
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != 0o600:
        raise RuntimeError(f"Profile mode must be exactly 0600, observed {mode:04o}.")


def parse_profile(path: pathlib.Path) -> dict[str, str]:
    path = path.expanduser()
    validate_profile_file_security(path)
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, separator, raw_value = line.partition("=")
        if not separator or key not in PROFILE_KEYS:
            raise RuntimeError(f"Unknown or invalid profile line: {raw_line!r}")
        if key in values:
            raise RuntimeError(f"Duplicate profile key: {key}")
        if any(token in raw_value for token in ("$", "`", "\\", ";", "&", "|", "<", ">")):
            raise RuntimeError(f"Unsafe shell syntax in profile key {key}")
        tokens = shlex.split(raw_value, posix=True)
        if len(tokens) != 1 or SAFE_VALUE.fullmatch(tokens[0]) is None:
            raise RuntimeError(f"Unsafe profile value for {key}")
        values[key] = tokens[0]
    return values


def check_profile(path: pathlib.Path, devices: str) -> int:
    try:
        profile = parse_profile(path)
        inventory, fingerprint = core.query_gpu_inventory(devices)
    except (RuntimeError, ValueError) as exc:
        print(f"PROFILE_INVALID: {exc}", file=sys.stderr)
        return 1

    expected = {
        "P2P_PROFILE_VERSION": core.PROFILE_VERSION,
        "P2P_PROFILE_STATUS": "validated",
        "P2P_PROFILE_DEVICES": devices,
        "P2P_PROFILE_DRIVER_VERSION": core.EXPECTED_DRIVER_VERSION,
        "P2P_PROFILE_KERNEL": os.uname().release,
        "P2P_PROFILE_GPU_FINGERPRINT": fingerprint,
        "P2P_PROFILE_GPU_UUIDS": ",".join(gpu["uuid"] for gpu in inventory),
        "P2P_PROFILE_NCCL_TRANSPORT_OBSERVATION": "p2p-confirmed",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_SHM_DISABLE": "0",
        "VLLM_SKIP_P2P_CHECK": "0",
    }
    mismatches = {
        key: {"profile": profile.get(key), "current": value}
        for key, value in expected.items()
        if profile.get(key) != value
    }
    if set(profile) != PROFILE_KEYS:
        mismatches["profile_keys"] = {
            "missing": sorted(PROFILE_KEYS - set(profile)),
            "extra": sorted(set(profile) - PROFILE_KEYS),
        }
    driver_result = core.check_driver_stack(inventory, core.EXPECTED_DRIVER_VERSION)
    if not driver_result.passed:
        mismatches["driver_stack"] = {
            "profile": core.EXPECTED_DRIVER_VERSION,
            "current": driver_result.detail,
        }
    boot_result = core.check_boot_configuration()
    if not boot_result.passed:
        mismatches["boot_configuration"] = {
            "profile": "IOMMU passthrough enabled",
            "current": boot_result.detail,
        }
    if mismatches:
        print("PROFILE_STALE=" + json.dumps(mismatches, separators=(",", ":")))
        return 1
    print(
        "PROFILE_VALID="
        + json.dumps(
            {
                "path": str(path),
                "devices": devices,
                "fingerprint": fingerprint,
                "transport": "p2p-confirmed",
            },
            separators=(",", ":"),
        )
    )
    return 0


def run_validation(args: argparse.Namespace) -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    script_path = pathlib.Path(__file__).resolve()
    devices = normalize_devices(args.devices)
    venv = pathlib.Path(args.venv).expanduser().resolve()
    python = core.vllm_python(venv)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = devices

    print("Consumer GPU P2P Doctor")
    print("=======================")
    print(f"Devices: {devices}")
    print(f"vLLM environment: {venv}")
    print(f"Kernel: {os.uname().release}")

    try:
        inventory, fingerprint = core.query_gpu_inventory(devices)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    core.print_inventory(inventory)

    results = [
        core.check_driver_stack(inventory, core.EXPECTED_DRIVER_VERSION),
        core.check_boot_configuration(),
        core.check_ampere(inventory, python, env),
    ]
    kernel_result = core.check_kernel_peer_access(
        repo_root,
        env,
        required=not args.allow_missing_nvcc,
        timeout=args.timeout,
    )
    if args.allow_missing_nvcc and "nvcc not found" in kernel_result.detail:
        kernel_result.passed = False
        kernel_result.required = False
    results.append(kernel_result)
    results.append(
        core.check_vllm_ipc(python, env, timeout=args.timeout, script_path=script_path)
    )
    nccl_result, transport = strict_nccl_check(
        python, env, timeout=args.timeout, script_path=script_path
    )
    results.append(nccl_result)
    core.print_results(results)

    if any(item.required and not item.passed for item in results):
        print("\nRESULT=FAIL")
        print("No validated P2P profile was written.")
        return 1

    if transport != "p2p-confirmed":
        if not args.allow_non_p2p_nccl:
            print("\nRESULT=FAIL")
            print(
                f"NCCL collective values were correct, but transport was {transport}. "
                "Every observed local channel must use P2P and no SHM/NET fallback."
            )
            return 1
        if args.write_profile:
            print("\nRESULT=FAIL")
            print("Diagnostic transport overrides can never generate a validated profile.")
            return 1
        print(f"\nRESULT=PASS_DIAGNOSTIC (transport={transport}; no profile written)")
        return 0

    if args.write_profile:
        write_profile(
            pathlib.Path(args.profile),
            profile_values(
                devices=devices,
                inventory=inventory,
                fingerprint=fingerprint,
                transport=transport,
            ),
        )
    print("\nRESULT=PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed P2P validator for consumer Ampere GPUs and vLLM."
    )
    subparsers = parser.add_subparsers(dest="command")
    validate = subparsers.add_parser("validate", help="Run all P2P validation gates.")
    validate.add_argument(
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", core.DEFAULT_DEVICES),
        help="Numeric physical GPU indices, comma-separated (default: 0,1).",
    )
    validate.add_argument(
        "--venv",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("VLLM_VENV_PATH", core.DEFAULT_VENV)),
    )
    validate.add_argument(
        "--profile",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("P2P_PROFILE_PATH", core.DEFAULT_PROFILE)),
    )
    validate.add_argument("--write-profile", action="store_true")
    validate.add_argument("--allow-missing-nvcc", action="store_true")
    validate.add_argument(
        "--allow-non-p2p-nccl",
        "--allow-nccl-shm",
        dest="allow_non_p2p_nccl",
        action="store_true",
        help="Diagnostic only; never permits profile generation.",
    )
    validate.add_argument("--timeout", type=int, default=core.DEFAULT_TIMEOUT_SECONDS)

    profile = subparsers.add_parser(
        "check-profile", help="Verify that a saved profile still matches this host."
    )
    profile.add_argument(
        "--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", core.DEFAULT_DEVICES)
    )
    profile.add_argument(
        "--profile",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("P2P_PROFILE_PATH", core.DEFAULT_PROFILE)),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args == ["_vllm-ipc"]:
        return core.hidden_vllm_ipc()
    if raw_args == ["_nccl-test"]:
        return core.hidden_nccl_test()
    if not raw_args:
        raw_args = ["validate"]
    parser = build_parser()
    args = parser.parse_args(raw_args)
    if args.command == "validate":
        try:
            return run_validation(args)
        except (RuntimeError, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
    if args.command == "check-profile":
        try:
            devices = normalize_devices(args.devices)
        except ValueError as exc:
            print(f"PROFILE_INVALID: {exc}", file=sys.stderr)
            return 1
        return check_profile(pathlib.Path(args.profile), devices)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
