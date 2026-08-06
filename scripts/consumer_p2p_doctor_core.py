"""Spawn-import shim for the dynamically loaded P2P validator core.

`torch.multiprocessing.spawn` serializes worker functions by module and name.
The front-end loads the reviewed core as `consumer_p2p_doctor_core`; this shim
makes the NCCL worker importable in spawned child interpreters.
"""

from p2p_doctor_core import _nccl_worker

__all__ = ["_nccl_worker"]
