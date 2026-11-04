"""Post-training recovery regression tests; CPU fixtures are software-only."""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
from pathlib import Path
import math
import tempfile
import unittest
from unittest.mock import patch

import torch
from aces import engine, utils
from aces.models import build_model
from aces.synthetic import make_synthetic


@contextmanager
def cuda_probe_using_cpu():
    """Exercise the CUDA-check branch with real CPU autograd, not GPU emulation results."""
    randn = torch.randn
    tensors, contexts = [], []
    def make_tensor(*args, **kwargs):
        if kwargs.get("device") != torch.device("cuda"):
            raise AssertionError("Probe did not request CUDA")
        contexts.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        kwargs["device"] = "cpu"
        tensor = randn(*args, **kwargs)
        tensors.append(tensor)
        return tensor
    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch, "randn", side_effect=make_tensor), \
         patch.object(torch.cuda, "synchronize") as synchronize:
        yield tensors, contexts, synchronize


class PostTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_cuda_probe(self, context):
        with context:
            before = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
            with cuda_probe_using_cpu() as (tensors, contexts, synchronize):
                result = utils.device_check(True)
            self.assertEqual(result, torch.device("cuda"))
            self.assertEqual((torch.is_grad_enabled(), torch.is_inference_mode_enabled()), before)
        self.assertEqual(contexts, [(True, False)])
        self.assertEqual(synchronize.call_count, 1)
        self.assertIsNotNone(tensors[0].grad)
        self.assertTrue(torch.isfinite(tensors[0].grad).all())
        self.assertGreater(tensors[0].grad.abs().sum().item(), 0)

    def test_cuda_probe_inside_no_grad_has_real_backward_and_restores_context(self):
        self.assert_cuda_probe(torch.no_grad())

    def test_cuda_probe_inside_inference_mode_has_real_backward_and_restores_context(self):
        self.assert_cuda_probe(torch.inference_mode())

    def test_cuda_probe_normal_context_keeps_grad_enabled(self):
        self.assert_cuda_probe(nullcontext())

    def test_cuda_probe_failure_restores_context_and_retains_underlying_error(self):
        with torch.no_grad():
            with patch.object(torch.cuda, "is_available", return_value=True), \
                 patch.object(torch, "randn", side_effect=RuntimeError("kernel fixture failed")):
                with self.assertRaisesRegex(RuntimeError, "kernel fixture failed") as caught:
                    utils.device_check(True)
            self.assertFalse(torch.is_grad_enabled())
            self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_cuda_unavailable_still_requires_explicit_cpu_permission(self):
        with patch.object(torch.cuda, "is_available", return_value=False), torch.no_grad():
            self.assertEqual(utils.device_check(False), torch.device("cpu"))
            with self.assertRaises(RuntimeError):
                utils.device_check(True)
            self.assertFalse(torch.is_grad_enabled())

    def fixture(self, root):
        manifest = make_synthetic(root/"data")
        settings = utils.read_json(Path(__file__).parents[1]/"settings.json")
        cfg = settings["training"]
        cfg.update(image_size=64, base_channels=8, graph_grid=4, graph_dim=32,
                   graph_heads=4, amp=False, cpu_threads=2)
        model = build_model("aces", cfg).eval()
        contract = {"config": cfg, "method": "aces", "seed": 17, "label_fraction": 0.25,
                    "manifest_hash": utils.digest(utils.read_json(manifest))}
        run = root/"runs"/"aces_labels025_seed17"
        checkpoint = run/"best.pt"
        utils.atomic_torch_save({"format_version": 1, "model": model.state_dict(),
                                "contract": contract, "epoch": 200,
                                "evaluation_weights": "software_fixture_only"}, checkpoint)
        utils.write_json(root/"settings.json", settings)
        return manifest, checkpoint, settings

    def checked_device_on_cpu(self, observed):
        def check(require_cuda=True):
            if not require_cuda:
                raise AssertionError("Post-training entry did not require the GPU check")
            observed.append(torch.is_grad_enabled())
            with cuda_probe_using_cpu() as (tensors, contexts, _):
                result = utils.device_check(True)
            self.assertEqual(result, torch.device("cuda"))
            self.assertEqual(contexts, [(True, False)])
            self.assertIsNotNone(tensors[0].grad)
            # Only inference on this test fixture runs on CPU. This does not
            # constitute a CUDA/Windows runtime or performance verification.
            return torch.device("cpu")
        return check

    def test_benchmark_uses_existing_checkpoint_under_no_grad(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, checkpoint, _ = self.fixture(root)
            original = utils.sha256(checkpoint)
            observed = []
            with patch.object(engine, "device_check", side_effect=self.checked_device_on_cpu(observed)):
                engine.benchmark(checkpoint, manifest, root/"latency.json", warmup=1, repetitions=2)
            self.assertEqual(observed, [False])
            self.assertGreater(utils.read_json(root/"latency.json")["median_ms"], 0)
            self.assertEqual(original, utils.sha256(checkpoint))

    def test_temperature_fitting_keeps_validation_scope_and_checkpoint_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, checkpoint, _ = self.fixture(root)
            original = utils.sha256(checkpoint)
            observed = []
            with patch.object(engine, "device_check", side_effect=self.checked_device_on_cpu(observed)):
                temperature = engine.fit_temperature(checkpoint, manifest, root/"temperature.json",
                                                     pixels_per_patient=128)
            self.assertEqual(observed, [True])
            self.assertTrue(math.isfinite(temperature) and temperature > 0)
            metadata = utils.read_json(root/"temperature.json")
            self.assertEqual(metadata["source_split"], "CAMUS_validation_only")
            self.assertEqual(metadata["checkpoint_sha256"], original)
            self.assertEqual(original, utils.sha256(checkpoint))

    def test_evaluate_entry_finishes_without_calling_train_or_modifying_checkpoint(self):
        import launcher
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, checkpoint, _ = self.fixture(root)
            original = utils.sha256(checkpoint)
            observed = []
            with patch.object(launcher, "ROOT", root), \
                 patch.object(launcher, "gpu_preflight"), \
                 patch.object(launcher, "ensure_data", return_value=(manifest, None)), \
                 patch.object(launcher.shutil, "disk_usage") as disk, \
                 patch.object(engine, "device_check", side_effect=self.checked_device_on_cpu(observed)), \
                 patch.object(engine, "train", side_effect=AssertionError("Completed training must not restart")):
                # This small fixture checks orchestration, not host disk capacity.
                disk.return_value.free = 100*2**30
                self.assertEqual(launcher.run_pipeline("evaluate", root/"settings.json"), 0)
            run = checkpoint.parent
            self.assertEqual(observed, [False, True, False])
            for name in ["latency.json", "temperature.json", "test/summary.json",
                         "test/per_patient.csv", "test/evaluation_identity.json"]:
                self.assertTrue((run/name).exists(), name)
            self.assertTrue((root/"results"/"RESULTS.md").exists())
            self.assertEqual(original, utils.sha256(checkpoint))
            self.assertFalse((run/"last.pt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
