# Correction and v2 release: consumer NVIDIA P2P for vLLM

The original version of this guide made an important category error: it called
a host-memory SHM configuration “P2P” while exporting
`NCCL_P2P_DISABLE=1`. It also skipped vLLM's actual CUDA IPC check and did not
install or require the 595.58.03 userspace driver that must match the patched
kernel modules.

Version 2 corrects the contract:

- exact NVIDIA 595.58.03 userspace/kernel-module alignment;
- reviewed and pinned open-kernel-module patch revision;
- CPU-specific IOMMU passthrough without hiding AER globally;
- direct GPU-kernel reads and writes across every directed pair;
- vLLM's own cross-process CUDA IPC mutation test;
- exact-value NCCL all-reduce with P2P enabled;
- a machine-bound profile invalidated by kernel, driver, GPU order/UUID, PCI
  bus, or boot-state changes;
- `NCCL_P2P_DISABLE=0` and `VLLM_SKIP_P2P_CHECK=0` in validated mode;
- an explicit `VLLM_P2P_MODE=shm` recovery mode that is correctly labelled as
  host-memory fallback.

The guide targets RTX 3090/Ampere Linux workstations and supports validating two
or three visible GPUs. TP=3 can use NCCL P2P, but vLLM custom all-reduce does not
support world size 3; the launcher now makes that distinction explicit.

I am not publishing version-independent claims about Ollama's internal transfer
path. The validator proves the CUDA peer path on the host. Each Ollama or
llama.cpp build must then be inspected and benchmarked on its own terms.

Repository: `groxaxo/consumer-gpu-p2p-vllm-guide`
