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

    def test_python_runtime_pin_matches_vllm_cuda_variant(self) -> None:
        self.assertEqual(
            installer.TORCH_INDEX_URL,
            "https://download.pytorch.org/whl/cu129",
        )
        self.assertEqual(
            installer.PYTHON_PACKAGES,
            (
                "torch==2.11.0+cu129",
                "torchvision==0.26.0+cu129",
                "torchaudio==2.11.0+cu129",
                "vllm==0.21.0",
            ),
        )
        self.assertIs(installer._core.install_vllm, installer.install_vllm)


class DoctorTests(unittest.TestCase):
    def test_normalize_devices(self) -> None:
        self.assertEqual(doctor.normalize_devices(" 2, 0,1 "), "2,0,1")
        with self.assertRaises(ValueError):
            doctor.normalize_devices("0")
        with self.assertRaises(ValueError):
            doctor.normalize_devices("0,0")
        with self.assertRaises(ValueError):
            doctor.normalize_devices("GPU-a,GPU-b")
        self.assertEqual(doctor.normalize_devices("00, 02"), "0,2")

    def test_profile_round_trip(self) -> None:
        values = {
            "P2P_PROFILE_VERSION": "2",
            "P2P_PROFILE_STATUS": "validated",
            "P2P_PROFILE_DEVICES": "2,0",
            "P2P_PROFILE_KERNEL": "kernel-no-spaces",
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

    def test_parse_profile_rejects_unknown_and_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "profile.env"
            path.write_text("export LD_PRELOAD=/tmp/evil.so\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                doctor.parse_profile(path)
            path.write_text(
                "export NCCL_P2P_DISABLE=0\nexport NCCL_P2P_DISABLE=0\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                doctor.parse_profile(path)
            path.write_text(
                "export P2P_PROFILE_CREATED_UTC=$(id)\n", encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                doctor.parse_profile(path)

    def test_profile_security_requires_mode_0600(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "profile.env"
            path.write_text("export NCCL_P2P_DISABLE=0\n", encoding="utf-8")
            path.chmod(0o644)
            with self.assertRaises(RuntimeError):
                doctor.validate_profile_file_security(path)
            path.chmod(0o600)
            doctor.validate_profile_file_security(path)

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
        with self.assertRaises(ValueError):
            doctor.profile_values(
                devices="0,1",
                inventory=[{"uuid": "GPU-a"}, {"uuid": "GPU-b"}],
                fingerprint="abc",
                transport="fallback-observed",
            )


if __name__ == "__main__":
    unittest.main()
