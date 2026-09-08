import copy
import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
from torch import nn

from openstl.models.simvp_factory import (
    InputBatchNorm2d, build_simvp_model, calibrate_batch_norm,
)
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss
from openstl.methods.radaz_validation import SourceValidationDiagnostics
from openstl.utils.callbacks import SnapshotCallback
from evaluate_radaz_conditioned_factorial import build_model
from evaluate_radaz_corrected_A import validate_source_window
from radaz_metrics_v3 import (
    weighted_median, organization, exact_spearman, local_observables,
    compare_observables, band_pool, skill,
)

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)


class NormalizationTests(unittest.TestCase):
    def test_input_bn_batch_order_buffers_and_training_equivalence(self):
        torch.manual_seed(10)
        bn = InputBatchNorm2d(4).eval()
        saved = copy.deepcopy(bn.state_dict())
        x = torch.randn(3, 4, 8, 8) * torch.tensor([1, 7, 40])[:, None, None, None]
        batched = bn(x)
        singles = torch.cat([bn(x[i:i+1]) for i in range(3)])
        torch.testing.assert_close(batched, singles)
        torch.testing.assert_close(bn(x.flip(0)).flip(0), singles)
        for k, v in saved.items():
            torch.testing.assert_close(v, bn.state_dict()[k])
        reference = nn.BatchNorm2d(4).train()
        torch.testing.assert_close(reference(x[:1]), bn(x[:1]))
        with self.assertRaises(ValueError):
            bn.train()(x)

    def test_calibration_no_mutation_and_no_dropout(self):
        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.hid = nn.Sequential(nn.BatchNorm2d(2), nn.Dropout(0.99))
            def forward(self, x):
                return self.hid(x)
        model = Probe().eval()
        saved = copy.deepcopy(model.state_dict())
        inputs = [torch.arange(32).reshape(1, 2, 4, 4).float() + a for a in [0, 10]]
        first = calibrate_batch_norm(model, iter(inputs))
        second = calibrate_batch_norm(model, iter(reversed(inputs)))
        torch.testing.assert_close(first["hid.0.running_mean"], torch.tensor([12.5, 28.5]))
        for k in first:
            torch.testing.assert_close(first[k], second[k])
        for k in saved:
            torch.testing.assert_close(saved[k], model.state_dict()[k])
        with self.assertRaises(ValueError):
            calibrate_batch_norm(model, [])

    def test_real_abc_configs_checkpoint_roundtrip_and_encoder_shapes(self):
        for letter, width, channels in [("A", 64, 64), ("B", 128, 64), ("C", 64, 128)]:
            with self.subTest(cell=letter), tempfile.TemporaryDirectory() as td:
                cfg_path = ROOT / f"configs/custom/pepapic/SimVP_gSTA_radaz_red{letter}_60ep.py"
                cfg = runpy.run_path(str(cfg_path))
                cfg["in_shape"] = (10, 5, 260, 256)
                cfg["aft_seq_length"] = 10
                model = build_simvp_model(cfg).eval()
                with torch.no_grad():
                    z, _ = model.enc(torch.zeros(1, 3, 260, 256))
                self.assertEqual(tuple(z.shape), (1, channels, 65, width))
                checkpoint = Path(td) / "model.ckpt"
                torch.save({"state_dict": {"model." + k: v for k, v in model.state_dict().items()}, "epoch": 59}, checkpoint)
                loaded, epoch = build_model(cfg_path, checkpoint, torch.device("cpu"))
                self.assertEqual(epoch, 59)
                for k, v in model.state_dict().items():
                    torch.testing.assert_close(v, loaded.state_dict()[k], rtol=0, atol=0)


class MetricTests(unittest.TestCase):
    def test_weighted_median_and_sign(self):
        self.assertEqual(weighted_median([1, 100, 2], [0.1, 0.8, 0.1]), 100)
        self.assertEqual(weighted_median([1, 2], [1, 1]), 1)
        true = (np.ones((1, 2)), np.ones((1, 2)), np.array([[0.9, 0.4]], dtype=complex))
        pred = (true[0], true[1], -true[2])
        r = organization(pred, true, 0.02)
        self.assertAlmostEqual(r["sign_flip_weight"], 1)
        self.assertAlmostEqual(r["O_ratio_weighted_median"], -1)
        self.assertGreater(r["O_signed_rmse"], 1)
        empty = organization((true[0]*0, true[1]*0, true[2]*0), (true[0]*0, true[1]*0, true[2]*0), 0.02)
        self.assertFalse(empty["valid"])
        self.assertIsNone(empty["O_ratio_weighted_median"])

    def test_exact_small_sample_permutation(self):
        r = exact_spearman([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertEqual(r["permutations"], 24)
        self.assertAlmostEqual(r["p_two_sided_exact"], 2/24)
        self.assertIsNone(exact_spearman([1, 1, 1], [1, 2, 3])["rho"])
        self.assertIsNone(skill(np.ones(2), np.ones(2), np.ones(2)))

    def test_local_product_parseval_and_time_permutation(self):
        width, height = 16, 8
        y = np.arange(width)*2*np.pi/width
        fields = np.zeros((1, 3, 3, height, width))
        radial = np.array([1, -1]*4)[:, None]
        for j, a in enumerate([1, 2, 4]):
            fields[0, j, 0] = 3 + a*radial*np.sin(y)
            fields[0, j, 2] = a*radial*np.cos(y)
        pool = np.ones((1, height))/height
        obs = local_observables(fields, pool, 1.0, 0.02, max_mode=8)
        ey = -(np.roll(fields[:, :, 2], -1, axis=-1)-np.roll(fields[:, :, 2], 1, axis=-1))/2
        direct = np.einsum("rh,sth->str", pool, -(fields[:, :, 0]*ey).mean(axis=-1)/0.02)
        np.testing.assert_allclose(obs["flux_full"], direct, atol=1e-12)
        self.assertGreater(np.linalg.norm(direct), 1)
        # Mean fields cancel radial covariance: old proxy is zero.
        np.testing.assert_allclose(fields[:, :, 0].mean(axis=-2)*ey.mean(axis=-2), 0, atol=1e-14)
        reversed_obs = local_observables(fields[:, ::-1], pool, 1, .02, max_mode=8)
        copy_obs = local_observables(np.repeat(fields[:, :1], 3, axis=1), pool, 1, .02, max_mode=8)
        r = compare_observables(reversed_obs, obs, copy_obs, .02, pool)
        self.assertLess(r["mean_modal_Gamma_nrmse"], 1e-12)
        self.assertGreater(r["gamma_time_nrmse"], 0.1)

    def test_local_training_loss_matches_independent_numpy_and_finite_gradients(self):
        meta = {"props": ["electron_den", "ion_den", "phi"],
                "valid_spatial_shape": (8, 16), "model_spatial_shape": (8, 16),
                "norm_low": np.zeros(3), "norm_high": np.ones(3),
                "x_m": np.linspace(.001, .011, 8), "y_m": np.arange(16),
                "condition_names": []}
        with patch.object(PEPAPICSpectralLoss, "_load_metadata", return_value=meta):
            lm = PEPAPICSpectralLoss("unused.h5", max_mode=8, coordinate_system="integer_power_cross", radial_reduction="local_product")
        torch.manual_seed(17)
        true = torch.rand(2, 3, 3, 8, 16)
        coeff = lm._band_coefficients(true, physical_units=False)
        pn, pe, cr, ci = lm._ensemble_spectra(coeff)
        reference = local_observables(true.numpy().astype(float), lm.radial_band_pool.numpy(), 1, .02, max_mode=8)
        for actual, key in [(pn, "pn"), (pe, "pe"), (cr+1j*ci, "cross")]:
            np.testing.assert_allclose(actual.numpy(), reference[key].mean(axis=1), rtol=3e-6, atol=1e-8)
        for pred in [true.clone(), torch.zeros_like(true)]:
            pred.requires_grad_()
            terms = lm(pred, true)
            sum(terms.values()).backward()
            self.assertTrue(torch.isfinite(pred.grad).all())


class ProtocolTests(unittest.TestCase):
    def test_balanced_validation_logs_all_conditions_and_channels(self):
        path = ROOT / "workdirs/2D_RadAz/radaz_conditioned_factorial_manifests/radaz_axis_factorial_manifest.json"
        diag = SourceValidationDiagnostics(path, torch.device("cpu"))
        for index, vector in enumerate(diag.conditions):
            x = torch.zeros(1, 2, 5, 4, 4)
            x[:, :, 3:] = vector[None, None, :, None, None]
            target = torch.ones(1, 2, 3, 4, 4)
            pred = target + (index + 1)
            diag.update(x, pred, target)
        result = diag.finalize()
        self.assertEqual(len(result["per_condition"]), 6)
        self.assertAlmostEqual(result["balanced_mse_mean"], np.mean(np.arange(1, 7)**2))
        first = next(iter(result["per_condition"].values()))
        self.assertEqual(first["normalized_mse_by_channel"], [1, 1, 1])
        self.assertEqual(first["skill_vs_copy_by_channel"], [0, 0, 0])
        x[:, :, 3:] = 99
        with self.assertRaises(ValueError):
            diag.update(x, pred, target)

    def test_frozen_bundle_and_evaluation_provenance(self):
        from evaluate_radaz_corrected_A import digest
        bundle = json.loads((ROOT / "workdirs/2D_RadAz/radaz_arch_v3_plan/bundle.json").read_text())
        for name, sha in bundle["sha256"].items():
            self.assertEqual(digest(ROOT / name), sha, name)
        self.assertEqual(len(bundle["commands"]), 12)
        self.assertFalse(bundle["commands_executed"])
        protocol = json.loads((ROOT / "workdirs/2D_RadAz/radaz_corrected_A_v3/protocol.json").read_text())
        for name, sha in protocol["code_config_manifest_sha256"].items():
            executed = ROOT / "workdirs/2D_RadAz/radaz_corrected_A_v3/executed_source" / name
            self.assertEqual(digest(executed if executed.exists() else ROOT / name), sha, name)

    def test_train_only_calibration_and_disjoint_splits(self):
        m = {"normalization": {"frame_counts": {"x": {"train_end_exclusive": 1600}}}}
        c = {"role": "source", "case_key": "x"}
        validate_source_window(c, m, 1500, 1510, "calibration")
        for start, stop, split in [(1595, 1605, "calibration"), (1599, 1620, "source_validation"), (1799, 1820, "source_test")]:
            with self.assertRaises(ValueError):
                validate_source_window(c, m, start, stop, split)
        with self.assertRaises(ValueError):
            validate_source_window(dict(c, role="holdout"), m, 1200, 1210, "calibration")

    def test_completed_epoch_60_snapshot_and_no_sanity_save(self):
        with tempfile.TemporaryDirectory() as td:
            saved = []
            cb = SnapshotCallback(td, [60], "completed")
            trainer = SimpleNamespace(current_epoch=59, global_rank=0, sanity_checking=False, save_checkpoint=saved.append)
            cb.on_validation_epoch_end(trainer, None)
            self.assertEqual(len(saved), 1)
            self.assertIn("epoch=59", saved[0])
            trainer.sanity_checking = True
            cb.on_validation_epoch_end(trainer, None)
            self.assertEqual(len(saved), 1)


if __name__ == "__main__":
    unittest.main()
