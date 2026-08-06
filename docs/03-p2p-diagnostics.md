# 3. P2P diagnostics: capability is not correctness

Consumer-GPU P2P must be validated at multiple layers. Do not collapse the
following into one “P2P works” statement.

## Layer 1 — driver capability report

```bash
python3 - <<'PY'
import torch
for source in range(torch.cuda.device_count()):
    for target in range(torch.cuda.device_count()):
        if source != target:
            print(source, target, torch.cuda.can_device_access_peer(source, target))
PY
```

A `True` result means the driver reports peer capability. vLLM explicitly warns
that a driver can report `True` even when actual peer access is broken.

## Layer 2 — `cudaMemcpyPeer`

`cudaMemcpyPeer` is a copy API. Its success and throughput do not, by
themselves, prove that a kernel on one GPU can dereference another GPU's mapped
memory. Drivers and runtimes can select different internal paths.

The revised benchmark labels the measurement accurately:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  python3 scripts/p2p_bandwidth_bench.py --save p2p-report.txt
```

It compares effective payload throughput for:

- the CUDA peer-copy API; and
- an explicit pinned-host D2H + H2D bounce.

It runs the direct integrity probe first unless the operator uses the strongly
discouraged `--skip-integrity-probe` flag.

## Layer 3 — direct peer kernel loads and stores

`scripts/p2p_probe.cu` enables every directed peer mapping and launches kernels
on GPU A that:

1. read an exact `uint64_t` pattern from memory owned by GPU B;
2. transform it and verify the complete local result;
3. write an exact pattern directly into memory owned by GPU B; and
4. verify the complete remote allocation from GPU B.

This catches silent corruption, partial mappings, and one-way failures.

Run it through the doctor:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  ~/venvs/vllm/bin/python scripts/p2p_doctor.py validate \
    --venv ~/venvs/vllm
```

Or compile it manually:

```bash
nvcc -O3 -std=c++17 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_86,code=compute_86 \
  scripts/p2p_probe.cu -o /tmp/p2p_probe

CUDA_VISIBLE_DEVICES=0,1 /tmp/p2p_probe --require-ampere
```

Every directed pair must report `read=PASS write=PASS`.

## Layer 4 — CUDA IPC across processes

vLLM custom all-reduce shares GPU allocations between worker processes. The
relevant test must therefore cross a process boundary.

With `VLLM_SKIP_P2P_CHECK=0`, vLLM's checker:

1. allocates memory on a source GPU in one process;
2. exports a CUDA IPC handle;
3. opens it from a process bound to the target GPU;
4. modifies the allocation from the target context; and
5. verifies both processes observe the exact mutation.

The doctor invokes this implementation directly for all directed pairs. A
failure blocks profile generation.

`VLLM_SKIP_P2P_CHECK=1` skips this operation and trusts the driver's capability
query. Do not use that setting with this patch.

## Layer 5 — NCCL correctness and transport

The doctor starts one NCCL rank per visible GPU with:

```bash
NCCL_P2P_DISABLE=0
NCCL_SHM_DISABLE=0
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM
```

It checks exact all-reduce results at multiple tensor sizes. It records a P2P
or SHM transport observation when the installed NCCL version emits a parsable
transport line. NCCL log wording varies, so absence of a transport string is
not treated as corruption; incorrect collective results always fail.

A log that explicitly says `via SHM` causes strict validation to fail because
SHM is not direct P2P. `--allow-nccl-shm` exists only for an intentional,
clearly labelled fallback profile.

## Full post-reboot gate

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/post-reboot-test.sh
```

The profile is written only after all required checks pass. `check-profile`
then validates the driver, kernel, boot state, device order, GPU UUIDs, and PCI
bus IDs before every vLLM start.

## Topology interpretation

Useful evidence:

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
lspci -tv
sudo lspci -vv -s <GPU-BDF>
```

Terms such as `PIX`, `PXB`, `PHB`, and `SYS` describe the route. They do not
prove data integrity. ACS may redirect peer traffic toward the root complex;
link width may also be lower than the physical slot suggests. Use topology to
explain a measured result, never to replace it.
