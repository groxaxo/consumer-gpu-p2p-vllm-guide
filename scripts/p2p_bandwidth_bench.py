#!/usr/bin/env python3
"""Run integrity-gated CUDA peer-copy bandwidth measurements."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROBE_SOURCE = SCRIPT_DIR / "p2p_probe.cu"
BENCH_SOURCE = SCRIPT_DIR / "p2p_bandwidth_bench.cu"


def run(args: Sequence[str | os.PathLike[str]], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [os.fspath(item) for item in args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(map(os.fspath, args))}\n"
            f"{completed.stdout}"
        )
    return completed


def compile_cuda(nvcc: pathlib.Path, source: pathlib.Path, output: pathlib.Path) -> None:
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
    result = run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed for {source}:\n{result.stdout}")


def command_output(args: Sequence[str]) -> str:
    result = run(args, check=False)
    return result.stdout.strip()


def system_report() -> str:
    lines = [
        "SYSTEM_REPORT_VERSION=2",
        f"timestamp_utc={dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}",
        f"host={platform.node()}",
        f"os={platform.platform()}",
        f"kernel={platform.release()}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}",
    ]
    driver = command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    lines.append("driver_versions=" + "|".join(sorted(set(driver.splitlines()))))
    inventory = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id,pci.link.gen.current,pci.link.width.current",
            "--format=csv,noheader,nounits",
        ]
    )
    lines.append("\n[GPU inventory]\n" + inventory)
    topology = command_output(["nvidia-smi", "topo", "-m"])
    lines.append("\n[Topology]\n" + topology)
    if shutil.which("lspci"):
        bus_ids = re.findall(r"(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", inventory)
        pcie_rows: list[str] = []
        for bus_id in bus_ids:
            short_id = bus_id[-7:]
            output = command_output(["lspci", "-s", short_id, "-vv"])
            link_lines = [line.strip() for line in output.splitlines() if "LnkCap:" in line or "LnkSta:" in line or "ACSCtl:" in line]
            pcie_rows.append(f"{short_id}: " + " | ".join(link_lines))
        lines.append("\n[PCIe link and ACS]\n" + "\n".join(pcie_rows))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=pathlib.Path, help="Save complete output.")
    parser.add_argument(
        "--skip-integrity-probe",
        action="store_true",
        help="Benchmark without direct peer read/write validation (not recommended).",
    )
    args = parser.parse_args()

    nvcc_path = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    nvcc = pathlib.Path(nvcc_path)
    if not nvcc.is_file():
        print("nvcc is required for the CUDA probes.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="consumer-p2p-bench-") as temporary:
        temporary_path = pathlib.Path(temporary)
        probe_binary = temporary_path / "p2p_probe"
        bench_binary = temporary_path / "p2p_bandwidth_bench"
        try:
            compile_cuda(nvcc, PROBE_SOURCE, probe_binary)
            compile_cuda(nvcc, BENCH_SOURCE, bench_binary)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

        output_parts = [system_report()]
        if not args.skip_integrity_probe:
            probe = run(
                [probe_binary, "--require-ampere", "--size-mib", "8", "--iterations", "3"],
                check=False,
            )
            output_parts.append("\n[Direct peer integrity]\n" + probe.stdout)
            if probe.returncode != 0:
                complete = "\n".join(output_parts)
                print(complete)
                if args.save:
                    args.save.write_text(complete + "\n", encoding="utf-8")
                print(
                    "Integrity probe failed; bandwidth numbers would be misleading, so the benchmark was not run.",
                    file=sys.stderr,
                )
                return 1

        benchmark = run([bench_binary], check=False)
        output_parts.append("\n[CUDA peer-copy API benchmark]\n" + benchmark.stdout)
        complete = "\n".join(output_parts)
        print(complete)
        if args.save:
            args.save.write_text(complete + "\n", encoding="utf-8")
            print(f"Saved: {args.save}")
        return benchmark.returncode


if __name__ == "__main__":
    raise SystemExit(main())
