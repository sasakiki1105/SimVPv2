import unittest
import numpy as np
from radaz_bc_metrics import factors, truth_weights, organization_error, spectrum_metrics, window_series


def spectra(o):
    o = np.asarray(o, dtype=float)
    return {'pn': np.ones_like(o), 'pe': np.ones_like(o), 'cross': o.astype(complex),
            'gamma': -2*o, 'flux_full': (-2*o).sum(-1)}


class OrganisationTests(unittest.TestCase):
    def test_signed_flip_is_detected_and_identity_has_zero_error(self):
        t = spectra(np.full((2,10,4,64), .5))
        p = spectra(-factors(t)['O'])
        # Uniform fixture exceeds the 0.001 per-frame weight floor.
        self.assertAlmostEqual(organization_error(factors(t)['O'], t)['O_time_rmse'], 0)
        r = organization_error(factors(p)['O'], t)
        self.assertAlmostEqual(r['O_time_rmse'], 1)
        self.assertAlmostEqual(r['O_time_sign_flip_weight'], 1)

    def test_temporal_reversal_preserves_mean_but_has_dynamic_error(self):
        o = np.broadcast_to(np.linspace(.1,.9,10)[None,:,None,None], (2,10,4,64))
        t, p = spectra(o), spectra(o[:,::-1])
        r = spectrum_metrics(p, t, o, o, t['gamma'], 1.)
        self.assertAlmostEqual(r['mean_spectrum_O']['O_signed_rmse'], 0)
        self.assertGreater(r['O_time_rmse'], .1)
        self.assertAlmostEqual(sum(map(sum, r['amplitude_O_interaction_gram_over_gamma_sse'])), 1)

    def test_mask_is_invariant_to_replicating_windows_and_zero_truth_is_invalid(self):
        t = spectra(np.full((2,10,4,64), .5))
        w,_ = truth_weights(t)
        doubled = {k:np.concatenate((v,v)) for k,v in t.items()}
        np.testing.assert_array_equal(truth_weights(doubled)[0], np.concatenate((w,w)))
        zero = spectra(np.zeros((1,10,4,64)))
        self.assertIsNone(organization_error(factors(zero)['O'], zero)['O_time_rmse'])

    def test_windows_do_not_cross_split_and_pool_all_phases(self):
        x,y = window_series(np.arange(800), 400, 600)
        self.assertEqual(x.shape, (181,10))
        self.assertEqual(y[-1,-1], 599)
        self.assertEqual(x[0,0], 400)
        self.assertEqual(x[::20].shape[0], 10)

    def test_epoch_order_survives_restart_and_ignores_model_rng(self):
        import torch
        from train_radaz_bc import EpochSampler, epoch_permutation
        first = epoch_permutation(101,42,7)
        torch.manual_seed(999)
        torch.rand(500)
        np.testing.assert_array_equal(first, epoch_permutation(101,42,7))
        self.assertEqual(sorted(first.tolist()),list(range(101)))
        self.assertFalse(np.array_equal(first,epoch_permutation(101,43,7)))
        sampler = EpochSampler(101,42,lambda:7)
        self.assertEqual(list(sampler),first.tolist())


if __name__ == '__main__':
    unittest.main()
