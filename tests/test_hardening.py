from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


doctor = load_module("consumer_p2p_hardening", ROOT / "scripts" / "p2p_doctor.py")


class HardeningTests(unittest.TestCase):
    def test_numeric_device_order_is_canonical(self) -> None:
        self.assertEqual(doctor.normalize_devices(" 02,0,1 "), "2,0,1")
        with self.assertRaises(ValueError):
            doctor.normalize_devices("GPU-abcd,1")
        with self.assertRaises(ValueError):
            doctor.normalize_devices("0,00")

    def test_profile_round_trip_is_owner_only(self) -> None:
        inventory = [{"uuid": "GPU-a"}, {"uuid": "GPU-b"}]
        values = doctor.profile_values(
            devices="0,1",
            inventory=inventory,
            fingerprint="abc123",
            transport="p2p-confirmed",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "profile.env"
            doctor.write_profile(path, values)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(doctor.parse_profile(path), values)

    def test_profile_rejects_shell_and_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "profile.env"
            path.write_text(
                "export P2P_PROFILE_VERSION=$(id)\nexport UNKNOWN=ok\n",
                encoding="utf-8",
            )
            path.chmod(0o600)
            with self.assertRaises(RuntimeError):
                doctor.parse_profile(path)

    def test_profile_rejects_non_p2p_transport(self) -> None:
        with self.assertRaises(ValueError):
            doctor.profile_values(
                devices="0,1",
                inventory=[{"uuid": "GPU-a"}, {"uuid": "GPU-b"}],
                fingerprint="abc123",
                transport="mixed-observed",
            )

    def test_nccl_mixed_transport_is_not_confirmed(self) -> None:
        original = doctor.core.run_command
        try:
            doctor.core.run_command = lambda *args, **kwargs: doctor.core.CommandResult(
                ["python"],
                0,
                'P2P_NCCL_JSON={"passed":true}\nChannel 00 via P2P/IPC\nChannel 01 via SHM/direct/direct',
                "",
            )
            result, transport = doctor.strict_nccl_check(
                pathlib.Path(sys.executable),
                os.environ.copy(),
                timeout=10,
                script_path=ROOT / "scripts" / "p2p_doctor.py",
            )
            self.assertTrue(result.passed)
            self.assertEqual(transport, "mixed-observed")
        finally:
            doctor.core.run_command = original


if __name__ == "__main__":
    unittest.main()
