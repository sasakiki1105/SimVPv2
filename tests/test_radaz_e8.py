"""Fixtures for E8 (matched continuation with time-resolved cross-phase loss)."""
import os
import unittest
from types import SimpleNamespace
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import torch

from radaz_e8_experiment import ROOT, EPOCH_OFFSET
from train_radaz_bc import epoch_permutation

MANIFEST = ROOT / 'workdirs/2D_RadAz/radaz_paired_pilot_v2/source_manifest.json'


def make_loss(time_resolved, max_mode=32):
    from openstl.methods.pepapic_spectral_loss_e8 import E8CrossSpectrumLoss
    return E8CrossSpectrumLoss(data_root=str(MANIFEST), max_mode=max_mode, radial_bands=4,
                               radial_min_m=0.0009, radial_max_m=0.0119,
                               coordinate_system='integer_power_cross', radial_reduction='local_product',
                               power_eps_relative=1e-8, cross_mask_kappa=1e-3, time_resolved=time_resolved)


def channel_indices():
    """(ne, phi, other) channel positions as the frozen loss reads them from the manifest."""
    loss = make_loss(True)
    ne, phi = int(loss.electron_index), int(loss.phi_index)
    return ne, phi, 3 - ne - phi


def wave_fields(T, phase_per_frame, amplitude=1.0, n=4, height=260, width=256, seed=0):
    """Synthetic normalised [1,T,3,H,W] in the manifest's channel order: phi carries
    cos(n y) so that Ey = -d(phi)/dy is a coherent mode-n wave, ne carries cos(n y + phase_t)
    with a ne-Ey phase difference that may change per frame; the third channel is noise."""
    ne, phi, other = channel_indices()
    g = np.random.default_rng(seed)
    y = np.arange(width) * 2 * np.pi / width
    x = np.zeros((1, T, 3, height, width), np.float32)
    for t in range(T):
        x[0, t, phi] = 0.5 + amplitude * 0.1 * np.cos(n * y)[None, :]
        x[0, t, ne] = 0.5 + amplitude * 0.1 * np.cos(n * y + phase_per_frame[t])[None, :]
        x[0, t, other] = 0.5 + 0.01 * g.standard_normal((height, width))
    return torch.from_numpy(x)


class E8LossTests(unittest.TestCase):
    @unittest.skipUnless(MANIFEST.exists(), 'source manifest not available')
    def test_identical_prediction_has_zero_loss_in_both_modes(self):
        truth = wave_fields(10, np.zeros(10))
        for tr in (True, False):
            loss = make_loss(tr)(truth.clone(), truth)['cross_spectrum']
            self.assertAlmostEqual(float(loss), 0.0, places=10)

    @unittest.skipUnless(MANIFEST.exists(), 'source manifest not available')
    def test_time_resolved_equals_averaged_when_single_frame(self):
        truth = wave_fields(1, np.array([0.3]))
        pred = wave_fields(1, np.array([1.1]))
        a = float(make_loss(True)(pred, truth)['cross_spectrum'])
        b = float(make_loss(False)(pred, truth)['cross_spectrum'])
        self.assertAlmostEqual(a, b, places=8)

    @unittest.skipUnless(MANIFEST.exists(), 'source manifest not available')
    def test_temporally_wrong_phase_is_invisible_to_averaged_but_seen_by_time_resolved(self):
        # Truth: ne-Ey phase alternates +p/-p frame to frame (mean cross ~ cos p, same every pair).
        # Prediction: the same alternation shifted by one frame. Time-averaged cross is identical,
        # per-frame cross differs.  This is the oracle counterexample of the B/C protocol.
        p = 1.0
        truth = wave_fields(10, np.array([p, -p] * 5))
        pred = wave_fields(10, np.array([-p, p] * 5))
        averaged = float(make_loss(False)(pred, truth)['cross_spectrum'])
        resolved = float(make_loss(True)(pred, truth)['cross_spectrum'])
        self.assertLess(averaged, 1e-6)
        self.assertGreater(resolved, 1e-3)

    @unittest.skipUnless(MANIFEST.exists(), 'source manifest not available')
    def test_mask_excludes_modes_without_truth_power(self):
        truth = wave_fields(10, np.zeros(10), n=4)
        pred = wave_fields(10, np.zeros(10), n=4)
        # Add prediction-only power at n=20: truth has none there, so the mask must ignore it.
        y = torch.arange(256, dtype=torch.float32) * 2 * np.pi / 256
        pred[0, :, channel_indices()[0]] += 0.1 * torch.cos(20 * y)[None, None, :]
        loss = float(make_loss(True)(pred, truth)['cross_spectrum'])
        self.assertLess(loss, 1e-6)


class SamplerAndScheduleTests(unittest.TestCase):
    def test_continuation_epochs_reuse_no_bc_permutation(self):
        bc = {epoch_permutation(9486, 42, e).tobytes() for e in range(60)}
        for e in range(10):
            self.assertNotIn(epoch_permutation(9486, 42, e + EPOCH_OFFSET).tobytes(), bc)
        np.testing.assert_array_equal(epoch_permutation(9486, 42, EPOCH_OFFSET), epoch_permutation(9486, 42, 60))

    def test_step_schedule_is_constant_for_the_whole_run(self):
        from openstl.core.optim_scheduler import get_optim_scheduler
        model = torch.nn.Linear(2, 2)
        args = SimpleNamespace(opt='adam', weight_decay=0.0, filter_bias_and_bn=False, lr=1e-4,
                               opt_eps=None, opt_betas=None, momentum=0.9, sched='step', decay_epoch=100,
                               decay_rate=0.1, warmup_lr=1e-5, warmup_epoch=0, min_lr=1e-6)
        opt, sch, by_epoch = get_optim_scheduler(args, 10, model, 9486)
        self.assertTrue(by_epoch)
        for e in range(11):
            sch.step(e)
            self.assertAlmostEqual(opt.param_groups[0]['lr'], 1e-4, places=12)


class CheckpointStructureTests(unittest.TestCase):
    def test_untracked_loss_attachment_leaves_state_dict_unchanged(self):
        class Host(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.model = torch.nn.Linear(3, 3)
        h = Host(); before = set(h.state_dict().keys())
        extra = torch.nn.Module(); extra.register_buffer('pool', torch.ones(2, 2))
        object.__setattr__(h, 'pepapic_spectral_loss_module', extra)
        self.assertEqual(set(h.state_dict().keys()), before)
        self.assertNotIn('pepapic_spectral_loss_module', dict(h.named_modules()))
        self.assertIs(h.pepapic_spectral_loss_module, extra)

    def test_source_checkpoint_has_only_model_keys(self):
        p = ROOT / 'workdirs/2D_RadAz/radaz_bc_Donly_B_GN_seed42_60ep_20260912/checkpoints/last.ckpt'
        if not p.exists():
            self.skipTest('B42 checkpoint not available')
        ck = torch.load(p, map_location='cpu')
        self.assertEqual(ck['epoch'], 59)
        self.assertTrue(all(k.startswith('model.') for k in ck['state_dict']))


if __name__ == '__main__':
    unittest.main()
