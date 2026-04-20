#!/usr/bin/env python3
"""
p2p_bandwidth_bench.py

Compile and run the CUDA P2P bandwidth benchmark, prefixed with a full
system context report (driver, GPU inventory, PCIe link state).

Usage:
    python3 scripts/p2p_bandwidth_bench.py [--save <output.txt>]

Requirements:
    - nvcc (CUDA Toolkit)
    - lspci  (pciutils)
    - nvidia-smi
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CUDA_SRC   = SCRIPT_DIR / "p2p_bandwidth_bench.cu"
CUDA_BIN   = Path(tempfile.mkdtemp()) / "p2p_bandwidth_bench"


# ─── helpers ──────────────────────────────────────────────────────────────────

def run(cmd, check=True, capture=True):
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None)
    if check and r.returncode != 0:
        print(f"[ERROR] command failed: {cmd}", file=sys.stderr)
        if r.stderr:
            print(r.stderr, file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip() if capture else ""


def banner(title):
    width = 80
    pad = (width - 2 - len(title)) // 2
    print("=" * width)
    print(" " * pad + title)
    print("=" * width)


# ─── system info ──────────────────────────────────────────────────────────────

def collect_system_info():
    lines = []

    # date / hostname
    lines.append(f"  Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Host      : {platform.node()}")
    lines.append(f"  OS        : {platform.platform()}")
    lines.append(f"  Kernel    : {platform.release()}")

    # driver version from /proc
    proc_ver = Path("/proc/driver/nvidia/version")
    if proc_ver.exists():
        m = re.search(r"NVRM version:.*?(\d+\.\d+\.\d+)", proc_ver.read_text())
        lines.append(f"  Driver    : {m.group(1) if m else 'unknown'}")

    # CUDA / nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        v = run(f"{nvcc} --version 2>&1 | tail -1", check=False)
        m = re.search(r"release (\S+),", v)
        lines.append(f"  CUDA      : {m.group(1) if m else v}")
        lines.append(f"  nvcc      : {nvcc}")
    else:
        lines.append("  CUDA      : nvcc not found in PATH")

    return "\n".join(lines)


def collect_gpu_info():
    """nvidia-smi GPU table."""
    fmt = "--query-gpu=index,name,memory.total,pci.bus_id,pci.link.gen.current,pci.link.width.current"
    raw = run(f"nvidia-smi {fmt} --format=csv,noheader 2>/dev/null", check=False)
    if not raw:
        return "  (nvidia-smi not available)"
    lines = []
    for row in raw.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 6:
            continue
        idx, name, mem, bus, gen, width = parts
        lines.append(f"  GPU{idx}  {name:<32} {mem:>8}  bus {bus}  PCIe Gen{gen} x{width}")
    return "\n".join(lines)


def collect_pcie_info():
    """Parse lspci LnkSta for each NVIDIA GPU."""
    if not shutil.which("lspci"):
        return "  (lspci not available — install pciutils)"

    # get bus IDs from nvidia-smi
    bus_ids_raw = run(
        "nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null",
        check=False
    )
    if not bus_ids_raw:
        return "  (nvidia-smi not available)"

    lines = []
    for row in bus_ids_raw.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 2:
            continue
        idx, full_bus = parts
        # strip domain prefix (0000:) lspci uses short form
        bus = full_bus.replace("00000000:", "").lower()

        lspci_out = run(f"lspci -vv -s {bus} 2>/dev/null", check=False)
        cap_m  = re.search(r"LnkCap:.*?Speed\s+(\S+),\s+Width\s+x(\d+)", lspci_out)
        sta_m  = re.search(r"LnkSta:.*?Speed\s+(\S+),\s+Width\s+x(\d+)", lspci_out)

        cap_str = f"Gen? x?" if not cap_m else f"cap: {cap_m.group(1)} x{cap_m.group(2)}"
        if sta_m:
            actual_speed = sta_m.group(1)
            actual_width = int(sta_m.group(2))
            downgrade = " ⚠ DOWNGRADED" if ("downgrad" in lspci_out.lower()) else ""
            sta_str = f"actual: {actual_speed} x{actual_width}{downgrade}"
        else:
            sta_str = "actual: unknown"

        lines.append(f"  GPU{idx}  bus {bus}   {cap_str}   {sta_str}")

    return "\n".join(lines)


def collect_topo():
    topo = run("nvidia-smi topo -m 2>/dev/null", check=False)
    if not topo:
        return "  (not available)"
    return "\n".join("  " + l for l in topo.splitlines())


# ─── compile ──────────────────────────────────────────────────────────────────

def compile_bench():
    nvcc = shutil.which("nvcc")
    if not nvcc:
        print("[ERROR] nvcc not found. Install CUDA Toolkit.", file=sys.stderr)
        sys.exit(1)
    if not CUDA_SRC.exists():
        print(f"[ERROR] source not found: {CUDA_SRC}", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Compiling {CUDA_SRC.name} ...", flush=True)
    r = subprocess.run(
        [nvcc, "-O2", "-arch=native", "-o", str(CUDA_BIN), str(CUDA_SRC)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        # fallback: sm_86 (Ampere)
        r = subprocess.run(
            [nvcc, "-O2", "-arch=sm_86", "-o", str(CUDA_BIN), str(CUDA_SRC)],
            capture_output=True, text=True
        )
    if r.returncode != 0:
        print("[ERROR] Compilation failed:", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    print(f"[+] Binary: {CUDA_BIN}", flush=True)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="P2P bandwidth benchmark wrapper")
    ap.add_argument("--save", metavar="FILE", help="save full output to file")
    args = ap.parse_args()

    compile_bench()

    out_lines = []

    def emit(line=""):
        print(line)
        out_lines.append(line)

    emit()
    banner("CONSUMER GPU P2P BANDWIDTH REPORT")
    out_lines.append("=" * 80)
    out_lines.append(" " * 26 + "CONSUMER GPU P2P BANDWIDTH REPORT")
    out_lines.append("=" * 80)

    emit("\n[System]")
    info = collect_system_info()
    emit(info)
    out_lines.append("[System]")
    out_lines.append(info)

    emit("\n[GPU Inventory (nvidia-smi)]")
    gpu_info = collect_gpu_info()
    emit(gpu_info)
    out_lines.append("\n[GPU Inventory (nvidia-smi)]")
    out_lines.append(gpu_info)

    emit("\n[PCIe Link State (lspci)]")
    pcie_info = collect_pcie_info()
    emit(pcie_info)
    out_lines.append("\n[PCIe Link State (lspci)]")
    out_lines.append(pcie_info)

    emit("\n[NV-SMI Topology]")
    topo = collect_topo()
    emit(topo)
    out_lines.append("\n[NV-SMI Topology]")
    out_lines.append(topo)

    emit("\n" + "─" * 80)
    emit("[CUDA Benchmark]")
    emit("─" * 80 + "\n")
    out_lines.append("\n" + "─" * 80)
    out_lines.append("[CUDA Benchmark]")
    out_lines.append("─" * 80)

    # run the CUDA binary and tee output
    proc = subprocess.Popen(
        [str(CUDA_BIN)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        out_lines.append(line.rstrip())
    proc.wait()

    if args.save:
        Path(args.save).write_text("\n".join(out_lines) + "\n")
        emit(f"\n[+] Output saved to: {args.save}")

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
