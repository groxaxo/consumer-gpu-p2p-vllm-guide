#!/usr/bin/env python3
"""Validate consumer-GPU P2P before allowing vLLM to use it.

The validator is deliberately integrity-first.  A positive
cudaDeviceCanAccessPeer() result is not accepted on its own.  A validated
profile requires:

1. matching 595.58.03 NVIDIA userspace and kernel modules;
2. IOMMU passthrough boot configuration;
3. a real cross-process CUDA IPC mutation test using vLLM's own checker;
4. a correct NCCL all-reduce with NCCL P2P enabled; and
5. when nvcc is available, direct kernel peer reads and writes with exact
   uint64 data verification.

A successful run can write a machine-bound shell profile.  The launcher checks
its driver, kernel, selected devices, and GPU UUIDs before every start.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Iterable, Sequence
from typing import Any

EXPECTED_DRIVER_VERSION = "595.58.03"
PROFILE_VERSION = "2"
DEFAULT_DEVICES = "0,1"
DEFAULT_VENV = pathlib.Path.home() / "venvs" / "vllm"
DEFAULT_PROFILE = pathlib.Path.home() / ".config" / "vllm" / "consumer-p2p.env"
DEFAULT_TIMEOUT_SECONDS = 180


@dataclasses.dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    required: bool = True


@dataclasses.dataclass(slots=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        if self.stderr:
            return f"{self.stdout}\n{self.stderr}".strip()
        return self.stdout.strip()


def run_command(
    args: Sequence[str | os.PathLike[str]],
    *,
    env: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    check: bool = False,
) -> CommandResult:
    command = [os.fspath(arg) for arg in args]
    try:
        completed = subprocess.run(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        result = CommandResult(command, 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        result = CommandResult(
            command,
            124,
            stdout,
            f"{stderr}\nTimed out after {timeout} seconds".strip(),
        )
    else:
        result = CommandResult(
            command,
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {shlex.join(result.args)}\n"
            f"{result.combined}"
        )
    return result


def read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def cpu_vendor() -> str:
    content = read_text(pathlib.Path("/proc/cpuinfo"))
    match = re.search(r"^vendor_id\s*:\s*(\S+)", content, re.MULTILINE)
    return match.group(1) if match else "unknown"


def required_iommu_argument(vendor: str) -> str | None:
    if vendor == "GenuineIntel":
        return "intel_iommu=on"
    if vendor == "AuthenticAMD":
        return "amd_iommu=on"
    return None


def normalize_devices(value: str) -> str:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        raise ValueError("At least two comma-separated GPU identifiers are required.")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA device identifiers must be unique.")
    return ",".join(devices)


def selected_device_count(devices: str) -> int:
    return len(devices.split(","))


def query_gpu_inventory(devices: str) -> tuple[list[dict[str, str]], str]:
    result = run_command(
        [
            "nvidia-smi",
            "-i",
            devices,
            "--query-gpu=index,uuid,name,pci.bus_id,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi inventory query failed: {result.combined}")

    inventory: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 4)]
        if len(parts) != 5:
            raise RuntimeError(f"Unexpected nvidia-smi row: {line!r}")
        index, uuid, name, pci_bus_id, driver = parts
        inventory.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "pci_bus_id": pci_bus_id,
                "driver": driver,
            }
        )
    if len(inventory) != selected_device_count(devices):
        raise RuntimeError(
            f"Requested {selected_device_count(devices)} GPUs but nvidia-smi "
            f"returned {len(inventory)}."
        )
    fingerprint_payload = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    return inventory, fingerprint


def module_version() -> str:
    result = run_command(["modinfo", "-F", "version", "nvidia"])
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 else ""


def loaded_driver_version() -> str:
    content = read_text(pathlib.Path("/proc/driver/nvidia/version"))
    match = re.search(r"Kernel Module\s+(\d+\.\d+(?:\.\d+)?)", content)
    if not match:
        match = re.search(r"NVRM version:.*?\s(\d+\.\d+(?:\.\d+)?)\s", content)
    return match.group(1) if match else ""


def check_driver_stack(
    inventory: list[dict[str, str]], expected: str
) -> CheckResult:
    kernel_module = module_version()
    loaded = loaded_driver_version()
    userspace_versions = sorted({gpu["driver"] for gpu in inventory})
    observed = {
        "modinfo": kernel_module or "missing",
        "loaded": loaded or "missing",
        "nvidia-smi": userspace_versions,
    }
    passed = (
        kernel_module == expected
        and loaded == expected
        and userspace_versions == [expected]
    )
    detail = json.dumps(observed, sort_keys=True)
    if not passed:
        detail += (
            f"; expected every layer to be {expected}. Install the exact NVIDIA "
            "userspace driver before the patched kernel modules."
        )
    return CheckResult("driver/userspace version match", passed, detail)


def check_boot_configuration() -> CheckResult:
    vendor = cpu_vendor()
    cmdline = read_text(pathlib.Path("/proc/cmdline")).strip()
    vendor_arg = required_iommu_argument(vendor)
    missing: list[str] = []
    if vendor_arg and vendor_arg not in cmdline.split():
        missing.append(vendor_arg)
    if "iommu=pt" not in cmdline.split():
        missing.append("iommu=pt")
    iommu_groups = pathlib.Path("/sys/kernel/iommu_groups")
    group_count = len(list(iommu_groups.iterdir())) if iommu_groups.is_dir() else 0
    passed = not missing and group_count > 0
    detail = (
        f"vendor={vendor}; required={vendor_arg or 'vendor-specific IOMMU arg unknown'},"
        f" iommu_groups={group_count}"
    )
    if missing:
        detail += f"; missing boot args: {', '.join(missing)}"
    if group_count == 0:
        detail += "; no active IOMMU groups found"
    return CheckResult("IOMMU passthrough boot configuration", passed, detail)


def check_ampere(inventory: list[dict[str, str]], python: pathlib.Path, env: dict[str, str]) -> CheckResult:
    code = textwrap.dedent(
        """
        import json, torch
        rows=[]
        for i in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(i)
            rows.append({"index": i, "name": p.name, "capability": list(torch.cuda.get_device_capability(i))})
        print("P2P_CAPABILITIES=" + json.dumps(rows, separators=(",", ":")))
        """
    )
    result = run_command([python, "-c", code], env=env)
    marker = next(
        (line for line in result.stdout.splitlines() if line.startswith("P2P_CAPABILITIES=")),
        "",
    )
    if result.returncode != 0 or not marker:
        return CheckResult("Ampere device check", False, result.combined)
    rows = json.loads(marker.split("=", 1)[1])
    passed = len(rows) == len(inventory) and all(row["capability"][0] == 8 for row in rows)
    detail = json.dumps(rows, separators=(",", ":"))
    if not passed:
        detail += "; this profile is intended for Ampere compute capability 8.x"
    return CheckResult("Ampere device check", passed, detail)


def find_nvcc() -> pathlib.Path | None:
    candidates = [
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
    ]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).is_file():
            return pathlib.Path(candidate)
    return None


def build_kernel_probe(repo_root: pathlib.Path, nvcc: pathlib.Path) -> pathlib.Path:
    source = repo_root / "scripts" / "p2p_probe.cu"
    if not source.is_file():
        raise RuntimeError(f"Missing probe source: {source}")
    cache_dir = pathlib.Path.home() / ".cache" / "consumer-gpu-p2p"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    output = cache_dir / f"p2p_probe-{digest}"
    if output.is_file() and os.access(output, os.X_OK):
        return output
    command = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-gencode",
        "arch=compute_80,code=sm_80",
        "-gencode",
        "arch=compute_86,code=sm_86",
        "-gencode",
        "arch=compute_86,code=compute_86",
        source,
        "-o",
        output,
    ]
    result = run_command(command, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed: {result.combined}")
    output.chmod(0o755)
    return output


def check_kernel_peer_access(
    repo_root: pathlib.Path,
    env: dict[str, str],
    *,
    required: bool,
    timeout: int,
) -> CheckResult:
    nvcc = find_nvcc()
    if nvcc is None:
        return CheckResult(
            "direct peer kernel read/write integrity",
            not required,
            "nvcc not found; install a CUDA toolkit matching the host runtime to run the defense-in-depth probe",
            required=required,
        )
    try:
        binary = build_kernel_probe(repo_root, nvcc)
    except RuntimeError as exc:
        return CheckResult(
            "direct peer kernel read/write integrity", False, str(exc), required=required
        )
    result = run_command(
        [binary, "--size-mib", "8", "--iterations", "3", "--require-ampere"],
        env=env,
        timeout=timeout,
    )
    result_marker = next(
        (line for line in result.stdout.splitlines() if line.startswith("RESULT=")),
        "",
    )
    passed = result.returncode == 0 and result_marker == "RESULT=PASS"
    pair_lines = [line for line in result.stdout.splitlines() if line.startswith("PAIR ")]
    detail = "; ".join(pair_lines) if pair_lines else result.combined[-4000:]
    return CheckResult(
        "direct peer kernel read/write integrity", passed, detail, required=required
    )


def vllm_python(venv: pathlib.Path) -> pathlib.Path:
    python = venv / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"vLLM Python not found: {python}")
    return python


def check_vllm_ipc(
    python: pathlib.Path,
    env: dict[str, str],
    *,
    timeout: int,
    script_path: pathlib.Path,
) -> CheckResult:
    child_env = env.copy()
    child_env["VLLM_SKIP_P2P_CHECK"] = "0"
    result = run_command(
        [python, script_path, "_vllm-ipc"], env=child_env, timeout=timeout
    )
    marker = next(
        (line for line in result.stdout.splitlines() if line.startswith("P2P_IPC_JSON=")),
        "",
    )
    if result.returncode != 0 or not marker:
        return CheckResult("vLLM CUDA IPC integrity", False, result.combined[-6000:])
    payload = json.loads(marker.split("=", 1)[1])
    pairs = payload.get("pairs", [])
    passed = bool(pairs) and all(item.get("passed") is True for item in pairs)
    detail = json.dumps(payload, separators=(",", ":"))
    return CheckResult("vLLM CUDA IPC integrity", passed, detail)


def check_nccl(
    python: pathlib.Path,
    env: dict[str, str],
    *,
    timeout: int,
    script_path: pathlib.Path,
) -> tuple[CheckResult, str]:
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
    result = run_command(
        [python, script_path, "_nccl-test"], env=child_env, timeout=timeout
    )
    marker = next(
        (line for line in result.stdout.splitlines() if line.startswith("P2P_NCCL_JSON=")),
        "",
    )
    combined = result.combined
    if result.returncode != 0 or not marker:
        return (
            CheckResult("NCCL all-reduce with P2P enabled", False, combined[-8000:]),
            "unknown",
        )
    payload = json.loads(marker.split("=", 1)[1])
    passed = payload.get("passed") is True
    upper = combined.upper()
    if re.search(r"\bVIA\s+P2P(?:/|\b)", upper) or "P2P/IPC" in upper:
        transport = "p2p-confirmed"
    elif re.search(r"\bVIA\s+SHM(?:/|\b)", upper):
        transport = "shm-observed"
    else:
        transport = "not-reported"
    detail = json.dumps(payload, separators=(",", ":")) + f"; transport={transport}"
    return CheckResult("NCCL all-reduce with P2P enabled", passed, detail), transport


def print_inventory(inventory: list[dict[str, str]]) -> None:
    print("\nSelected GPUs")
    print("--------------")
    for gpu in inventory:
        print(
            f"GPU {gpu['index']}: {gpu['name']} | {gpu['uuid']} | "
            f"PCI {gpu['pci_bus_id']} | driver {gpu['driver']}"
        )


def print_results(results: Iterable[CheckResult]) -> None:
    print("\nValidation")
    print("----------")
    for item in results:
        status = "PASS" if item.passed else ("WARN" if not item.required else "FAIL")
        print(f"[{status}] {item.name}")
        for line in item.detail.splitlines() or [""]:
            print(f"       {line}")


def quote_env(value: str) -> str:
    return shlex.quote(value)


def profile_values(
    *,
    devices: str,
    inventory: list[dict[str, str]],
    fingerprint: str,
    transport: str,
) -> dict[str, str]:
    return {
        "P2P_PROFILE_VERSION": PROFILE_VERSION,
        "P2P_PROFILE_STATUS": "validated",
        "P2P_PROFILE_DEVICES": devices,
        "P2P_PROFILE_DRIVER_VERSION": EXPECTED_DRIVER_VERSION,
        "P2P_PROFILE_KERNEL": os.uname().release,
        "P2P_PROFILE_GPU_FINGERPRINT": fingerprint,
        "P2P_PROFILE_GPU_UUIDS": ",".join(gpu["uuid"] for gpu in inventory),
        "P2P_PROFILE_NCCL_TRANSPORT_OBSERVATION": transport,
        "P2P_PROFILE_CREATED_UTC": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "NCCL_P2P_DISABLE": "0",
        "NCCL_SHM_DISABLE": "0",
        # Keep the real vLLM checker enabled.  Once generated, vLLM reuses its
        # fingerprinted cache instead of repeating the subprocess probe.
        "VLLM_SKIP_P2P_CHECK": "0",
    }


def write_profile(path: pathlib.Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by p2p_doctor.py after destructive integrity checks.",
        "# Do not copy this file to another machine or GPU ordering.",
    ]
    lines.extend(f"export {key}={quote_env(value)}" for key, value in values.items())
    data = "\n".join(lines) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    print(f"\nWrote validated profile: {path}")


def parse_profile(path: pathlib.Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"Validated profile not found: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, separator, raw_value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Z0-9_]+", key):
            raise RuntimeError(f"Invalid profile line: {raw_line!r}")
        tokens = shlex.split(raw_value, posix=True)
        if len(tokens) != 1:
            raise RuntimeError(f"Invalid profile value for {key}")
        values[key] = tokens[0]
    return values


def check_profile(path: pathlib.Path, devices: str) -> int:
    try:
        profile = parse_profile(path)
        inventory, fingerprint = query_gpu_inventory(devices)
    except (RuntimeError, ValueError) as exc:
        print(f"PROFILE_INVALID: {exc}", file=sys.stderr)
        return 1
    expected = {
        "P2P_PROFILE_VERSION": PROFILE_VERSION,
        "P2P_PROFILE_STATUS": "validated",
        "P2P_PROFILE_DEVICES": devices,
        "P2P_PROFILE_DRIVER_VERSION": EXPECTED_DRIVER_VERSION,
        "P2P_PROFILE_KERNEL": os.uname().release,
        "P2P_PROFILE_GPU_FINGERPRINT": fingerprint,
        "NCCL_P2P_DISABLE": "0",
        "NCCL_SHM_DISABLE": "0",
        "VLLM_SKIP_P2P_CHECK": "0",
    }
    mismatches = {
        key: {"profile": profile.get(key), "current": value}
        for key, value in expected.items()
        if profile.get(key) != value
    }
    driver_result = check_driver_stack(inventory, EXPECTED_DRIVER_VERSION)
    if not driver_result.passed:
        mismatches["driver_stack"] = {
            "profile": EXPECTED_DRIVER_VERSION,
            "current": driver_result.detail,
        }
    boot_result = check_boot_configuration()
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
                "transport": profile.get(
                    "P2P_PROFILE_NCCL_TRANSPORT_OBSERVATION", "unknown"
                ),
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
    python = vllm_python(venv)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = devices

    print("Consumer GPU P2P Doctor")
    print("=======================")
    print(f"Devices: {devices}")
    print(f"vLLM environment: {venv}")
    print(f"Kernel: {os.uname().release}")

    try:
        inventory, fingerprint = query_gpu_inventory(devices)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print_inventory(inventory)

    results: list[CheckResult] = [
        check_driver_stack(inventory, EXPECTED_DRIVER_VERSION),
        check_boot_configuration(),
        check_ampere(inventory, python, env),
    ]
    results.append(
        check_kernel_peer_access(
            repo_root,
            env,
            required=not args.allow_missing_nvcc,
            timeout=args.timeout,
        )
    )
    results.append(
        check_vllm_ipc(
            python, env, timeout=args.timeout, script_path=script_path
        )
    )
    nccl_result, transport = check_nccl(
        python, env, timeout=args.timeout, script_path=script_path
    )
    results.append(nccl_result)
    print_results(results)

    required_failures = [item for item in results if item.required and not item.passed]
    if required_failures:
        print("\nRESULT=FAIL")
        print(
            "No P2P profile was written. Use VLLM_P2P_MODE=shm only as an "
            "explicit fallback while resolving the failed gate(s)."
        )
        return 1

    if transport == "shm-observed" and not args.allow_nccl_shm:
        print("\nRESULT=FAIL")
        print(
            "NCCL completed correctly but logged SHM transport while P2P was "
            "enabled. Re-run with --allow-nccl-shm only when this is an "
            "intentional, benchmarked fallback; it is not direct P2P."
        )
        return 1

    if args.write_profile:
        write_profile(
            pathlib.Path(args.profile).expanduser(),
            profile_values(
                devices=devices,
                inventory=inventory,
                fingerprint=fingerprint,
                transport=transport,
            ),
        )
    print("\nRESULT=PASS")
    return 0


def _import_vllm_p2p_checker() -> Any:
    candidates = (
        "vllm.distributed.device_communicators.all_reduce_utils",
        "vllm.distributed.device_communicators.custom_all_reduce_utils",
    )
    errors: list[str] = []
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["can_actually_p2p"])
            return getattr(module, "can_actually_p2p")
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise RuntimeError("Could not import vLLM P2P checker: " + "; ".join(errors))


def hidden_vllm_ipc() -> int:
    try:
        import torch

        checker = _import_vllm_p2p_checker()
        count = torch.cuda.device_count()
        if count < 2:
            raise RuntimeError("At least two visible CUDA devices are required")
        sources: list[int] = []
        targets: list[int] = []
        for source in range(count):
            for target in range(count):
                if source != target:
                    sources.append(source)
                    targets.append(target)
        outcomes = list(checker(sources, targets))
        payload = {
            "device_count": count,
            "pairs": [
                {"source": source, "target": target, "passed": bool(passed)}
                for source, target, passed in zip(sources, targets, outcomes)
            ],
        }
        print("P2P_IPC_JSON=" + json.dumps(payload, separators=(",", ":")))
        return 0 if all(outcomes) else 1
    except Exception as exc:  # noqa: BLE001 - child must serialize all failures
        print(f"P2P_IPC_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _nccl_worker(rank: int, world_size: int, port: int, result_file: str) -> None:
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    sizes = (1024, 1 << 20, 8 << 20)
    checks: list[dict[str, Any]] = []
    for elements in sizes:
        tensor = torch.full(
            (elements,), float(rank + 1), dtype=torch.float32, device=rank
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(rank)
        expected = float(world_size * (world_size + 1) // 2)
        finite = bool(torch.isfinite(tensor).all().item())
        correct = bool(torch.all(tensor == expected).item())
        checks.append(
            {
                "elements": elements,
                "expected": expected,
                "finite": finite,
                "correct": correct,
            }
        )
    gathered: list[list[dict[str, Any]] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, checks)
    if rank == 0:
        payload = {
            "world_size": world_size,
            "ranks": gathered,
            "passed": all(
                item["finite"] and item["correct"]
                for rank_checks in gathered
                if rank_checks is not None
                for item in rank_checks
            ),
        }
        pathlib.Path(result_file).write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
    dist.barrier()
    dist.destroy_process_group()


def hidden_nccl_test() -> int:
    try:
        import socket

        import torch
        import torch.multiprocessing as mp

        world_size = torch.cuda.device_count()
        if world_size < 2:
            raise RuntimeError("At least two visible CUDA devices are required")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.NamedTemporaryFile(prefix="p2p-nccl-", delete=False) as handle:
            result_path = handle.name
        try:
            mp.spawn(
                _nccl_worker,
                args=(world_size, port, result_path),
                nprocs=world_size,
                join=True,
            )
            payload = pathlib.Path(result_path).read_text(encoding="utf-8")
            print("P2P_NCCL_JSON=" + payload)
            return 0 if json.loads(payload).get("passed") is True else 1
        finally:
            pathlib.Path(result_path).unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"P2P_NCCL_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrity-first P2P validator for consumer Ampere GPUs and vLLM."
    )
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate", help="Run all P2P validation gates.")
    validate.add_argument(
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", DEFAULT_DEVICES),
        help="Physical GPU indices/UUIDs to expose, comma-separated (default: 0,1).",
    )
    validate.add_argument(
        "--venv",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("VLLM_VENV_PATH", DEFAULT_VENV)),
        help=f"vLLM virtual environment (default: {DEFAULT_VENV}).",
    )
    validate.add_argument(
        "--profile",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("P2P_PROFILE_PATH", DEFAULT_PROFILE)),
        help=f"Profile output path (default: {DEFAULT_PROFILE}).",
    )
    validate.add_argument(
        "--write-profile",
        action="store_true",
        help="Write the machine-bound launcher profile after every required gate passes.",
    )
    validate.add_argument(
        "--allow-missing-nvcc",
        action="store_true",
        help="Treat the direct CUDA-kernel probe as advisory when nvcc is unavailable.",
    )
    validate.add_argument(
        "--allow-nccl-shm",
        action="store_true",
        help="Allow a correct NCCL run that explicitly logs SHM instead of P2P.",
    )
    validate.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-probe timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )

    profile = subparsers.add_parser(
        "check-profile", help="Verify that a saved profile still matches this host."
    )
    profile.add_argument(
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", DEFAULT_DEVICES),
    )
    profile.add_argument(
        "--profile",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("P2P_PROFILE_PATH", DEFAULT_PROFILE)),
    )

    # Internal subprocess entry points. They remain visible in --help only as a
    # debugging aid and are not part of the public workflow.
    subparsers.add_parser("_vllm-ipc", help=argparse.SUPPRESS)
    subparsers.add_parser("_nccl-test", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["validate", *(argv or [])])
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
        return check_profile(pathlib.Path(args.profile).expanduser(), devices)
    if args.command == "_vllm-ipc":
        return hidden_vllm_ipc()
    if args.command == "_nccl-test":
        return hidden_nccl_test()
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
