from __future__ import annotations

import base64
import importlib.util
import pathlib
import pickle
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class SpawnImportTests(unittest.TestCase):
    def test_dynamic_nccl_worker_is_importable_by_spawn_name(self) -> None:
        wrapper_path = ROOT / "scripts" / "p2p_doctor.py"
        spec = importlib.util.spec_from_file_location("spawn_wrapper_test", wrapper_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        payload = base64.b64encode(pickle.dumps(module.core._nccl_worker)).decode()
        code = (
            "import base64,pickle,sys;"
            f"sys.path.insert(0,{str(ROOT / 'scripts')!r});"
            "fn=pickle.loads(base64.b64decode(sys.argv[1]));"
            "assert fn.__name__ == '_nccl_worker'"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code, payload],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
