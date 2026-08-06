from __future__ import annotations

import importlib.util
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


installer = load_module("consumer_p2p_installer", ROOT / "install.py")
doctor = load_module("consumer_p2p_doctor", ROOT / "scripts" / "p2p_doctor.py")


class InstallerTests(unittest.TestCase):
    def test_merge_grub_arguments_preserves_existing(self) -> None:
        original = (
            '# comment\n'
            'GRUB_CMDLINE_LINUX_DEFAULT="quiet splash mitigations=auto"\n'
            'GRUB_CMDLINE_LINUX=""\n'
        )
        updated = installer.merge_grub_arguments(
            original, ["intel_iommu=on", "iommu=pt"]
        )
        self.assertIn(
            'GRUB_CMDLINE_LINUX_DEFAULT="quiet splash mitigations=auto '
            'intel_iommu=on iommu=pt"',
            updated,
        )
        self.assertIn('GRUB_CMDLINE_LINUX=""', updated)

    def test_merge_grub_arguments_is_idempotent(self) -> None:
        original = (
            'GRUB_CMDLINE_LINUX_DEFAULT="quiet intel_iommu=on iommu=pt"\n'
        )
        once = installer.merge_grub_arguments(
            original, ["intel_iommu=on", "iommu=pt"]
        )
        twice = installer.merge_grub_arguments(
            once, ["intel_iommu=on", "iommu=pt"]
        )
        self.assertEqual(once, twice)
        self.assertEqual(once.count("iommu=pt"), 1)

    def test_merge_grub_arguments_requires_assignment(self) -> None:
        with self.assertRaises(installer.InstallerError):
            installer.merge_grub_arguments("GRUB_TIMEOUT=5\n", ["iommu=pt"])

    def test_dkms_config_contains_all_modules(self) -> None:
        config = installer.dkms_config()
        for module in (
            "nvidia",
            "nvidia-uvm",
            "nvidia-modeset",
            "nvidia-drm",
            "nvidia-peermem",
        ):
            self.assertIn(f'="{module}"', config)
        self.assertIn('PACKAGE_VERSION="595.58.03"', config)


class DoctorTests(unittest.TestCase):
    def test_normalize_devices(self) -> None:
        self.assertEqual(doctor.normalize_devices(" 2, 0,1 "), "2,0,1")
        with self.assertRaises(ValueError):
            doctor.normalize_devices("0")
        with self.assertRaises(ValueError):
            doctor.normalize_devices("0,0")

    def test_profile_round_trip(self) -> None:
        values = {
            "P2P_PROFILE_VERSION": "2",
            "P2P_PROFILE_STATUS": "validated",
            "P2P_PROFILE_DEVICES": "2,0",
            "P2P_PROFILE_KERNEL": "kernel with spaces",
            "NCCL_P2P_DISABLE": "0",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "profile.env"
            doctor.write_profile(path, values)
            parsed = doctor.parse_profile(path)
            self.assertEqual(parsed, values)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_parse_profile_rejects_shell_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "profile.env"
            path.write_text("export SAFE=ok\nrm -rf /\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                doctor.parse_profile(path)

    def test_profile_values_keep_real_vllm_check_enabled(self) -> None:
        values = doctor.profile_values(
            devices="0,1",
            inventory=[
                {"uuid": "GPU-a"},
                {"uuid": "GPU-b"},
            ],
            fingerprint="abc",
            transport="p2p-confirmed",
        )
        self.assertEqual(values["NCCL_P2P_DISABLE"], "0")
        self.assertEqual(values["VLLM_SKIP_P2P_CHECK"], "0")


if __name__ == "__main__":
    unittest.main()
