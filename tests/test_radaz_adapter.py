"""CPU fixtures for the tensor-state adapter (design note 2026-09-28). No GPU, no frozen file modified."""
import os
import runpy
import unittest
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import numpy as np
import torch

from radaz_adapter_model import AdaptedSimVP, ADAPTER_INPUT_DIM, state_sha256
from radaz_adapter_state import TensorState, ARMS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/custom/pepapic/SimVP_gSTA_radaz_bc_Donly_B_60ep.py'
R3 = ROOT.parent / 'research_results/r3_20260915/R3_tensor_features.npz'


def small_backbone(seed=0):
    from openstl.models.simvp_factory import build_simvp_model
    cfg = {k: v for k, v in runpy.run_path(str(CONFIG)).items() if not k.startswith('__')}
    cfg['in_shape'] = (10, 3 + int(cfg.get('condition_dim', 0)), 32, 32)
    cfg.setdefault('aft_seq_length', 10); cfg.setdefault('simvp_direct_aft_seq', True); cfg.setdefault('out_channels', 3)
    torch.manual_seed(seed)
    return build_simvp_model(cfg).eval()


class AdapterModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.bb = small_backbone()
        self.x = torch.rand(2, 10, 5, 32, 32)
        self.aux = torch.randn(2, 10, ADAPTER_INPUT_DIM)

    def test_identity_at_initialisation_for_any_input(self):
        with torch.no_grad():
            ref = self.bb(self.x)
            for _ in range(3):
                model = AdaptedSimVP(small_backbone())
                out = model(self.x, torch.randn(2, 10, ADAPTER_INPUT_DIM))
                self.assertTrue(torch.allclose(out, ref, atol=0, rtol=0), 'zero-initialised FiLM must reproduce the backbone exactly')

    def test_backbone_frozen_and_only_adapter_receives_gradients(self):
        model = AdaptedSimVP(self.bb).train()
        before = state_sha256(model.backbone)
        y = torch.rand(2, 10, 3, 32, 32)
        loss = torch.mean((model(self.x, self.aux) - y) ** 2)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        adapter = model.adapter_parameters()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter), 'adapter must receive gradient')
        torch.optim.SGD(adapter, lr=1.0).step()
        self.assertEqual(state_sha256(model.backbone), before)
        self.assertFalse(any(k.startswith('backbone.') for k in model.adapter_state_dict()))

    def test_parameter_count_identical_across_arms(self):
        counts = {AdaptedSimVP(small_backbone(s)).adapter_parameters().__len__() for s in range(2)}
        n = [sum(p.numel() for p in AdaptedSimVP(small_backbone(s)).adapter_parameters()) for s in range(3)]
        self.assertEqual(len(set(n)), 1)
        self.assertLess(n[0], 250_000, 'adapter must stay small relative to the 13.1M backbone (design: ~202k = 1.5%)')

    def test_backbone_hash_recorded_and_film_path_required_absent(self):
        model = AdaptedSimVP(self.bb)
        self.assertEqual(model.backbone_sha256, state_sha256(self.bb))
        self.assertIsNone(self.bb.hid.film)


class TensorStateTests(unittest.TestCase):
    def synthetic(self, n=60, hat=True):
        rng = np.random.default_rng(3)
        frame = np.arange(100, 100 + n); time_us = 12.0 + 0.015 * np.arange(n)
        tf = rng.normal(size=(n, 4, 5, 2)) * np.array([1, 10, 100, 1000, 1e4])[None, None, :, None]
        NG = tf[:, :, 1:, :].reshape(n, 32)
        h = 0.5 * NG + rng.normal(size=NG.shape) if hat else None
        return frame, time_us, tf, h

    def test_train_only_statistics(self):
        frame, t, tf, h = self.synthetic()
        s1 = TensorState(frame, t, tf, train_window_us=(12.0, 12.45), hat=h)
        tf2 = tf.copy(); tf2[35:] *= 7.0            # perturb frames after the TRAIN window
        s2 = TensorState(frame, t, tf2, train_window_us=(12.0, 12.45), hat=h)
        for arm in ('FA', 'Fraw', 'Fnoise'):
            np.testing.assert_array_equal(s1.stats[arm]['mean'], s2.stats[arm]['mean'])
            np.testing.assert_array_equal(s1.stats[arm]['std'], s2.stats[arm]['std'])
        self.assertEqual(s1.stats['FA']['train_frames'], 30)

    def test_raw_equals_hat_plus_residual_and_padding(self):
        frame, t, tf, h = self.synthetic()
        s = TensorState(frame, t, tf, train_window_us=(12.0, 12.45), hat=h)
        raw = s.states['Fraw'][:, 8:40] * s.stats['Fraw']['std'][8:] + s.stats['Fraw']['mean'][8:]
        hat = s.states['Fhat'][:, 8:40] * s.stats['Fhat']['std'][8:] + s.stats['Fhat']['mean'][8:]
        res = s.states['Fres'][:, 8:40] * s.stats['Fres']['std'][8:] + s.stats['Fres']['mean'][8:]
        np.testing.assert_allclose(hat + res, raw, rtol=1e-10, atol=1e-8)
        self.assertEqual(s.stats['FA']['dim'], 8); self.assertTrue(np.all(s.states['FA'][:, 8:] == 0))
        self.assertEqual(s.stats['Fconst']['dim'], 1); self.assertTrue(np.all(s.states['Fconst'][:, 0] == 1))
        self.assertEqual(set(s.available_arms()), set(ARMS) - {'F_mom'})

    def test_window_alignment_and_bounds(self):
        frame, t, tf, h = self.synthetic()
        s = TensorState(frame, t, tf, train_window_us=(12.0, 12.45), hat=h)
        w = s.window('Fraw', 110)
        self.assertEqual(tuple(w.shape), (10, ADAPTER_INPUT_DIM))
        np.testing.assert_allclose(w.numpy(), s.states['Fraw'][10:20].astype(np.float32))
        with self.assertRaises(IndexError):
            s.window('Fraw', 155)
        b = s.batch('Fnoise', [100, 101]); self.assertEqual(tuple(b.shape), (2, 10, ADAPTER_INPUT_DIM))
        s_again = TensorState(frame, t, tf, train_window_us=(12.0, 12.45), hat=h)
        np.testing.assert_array_equal(s.states['Fnoise'], s_again.states['Fnoise'])   # deterministic noise

    @unittest.skipUnless(R3.exists(), 'R3 tensor cache not available')
    def test_real_r3_cache_loads_with_train_window(self):
        z = np.load(R3)
        s = TensorState(z['frame'], z['time_us'], z['tensor_features'])
        self.assertEqual(s.n, 1200)
        self.assertGreater(s.stats['Fraw']['train_frames'], 500)
        self.assertEqual(tuple(s.window('Fraw', int(z['frame'][0])).shape), (10, ADAPTER_INPUT_DIM))


if __name__ == '__main__':
    unittest.main()
