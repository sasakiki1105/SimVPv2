import unittest
import numpy as np
from analyze_radaz_nextstep_baselines import windows, fit_ar, predict_ar, summary_error


class BaselineTests(unittest.TestCase):
    def test_windows_do_not_cross_split(self):
        a = np.arange(80.)[:, None]
        x, y = windows(a, 20, 60, 20)
        self.assertEqual(x[:, 0, 0].tolist(), [20, 40])
        self.assertEqual(y[:, -1, 0].tolist(), [39, 59])

    def test_ar_predicts_known_recurrence(self):
        t = np.arange(300.)
        a = np.stack([np.sin(t*.3), 10+3*np.cos(t*.17)], axis=1)
        model = fit_ar(a[:200], 1e-6)
        x, y = windows(a, 200, 300, 20)
        p = predict_ar(model, x)
        self.assertLess(np.mean((p-y)**2), 1e-8)

    def test_constant_series_and_bias_decomposition(self):
        a = np.ones((200, 2, 3))*5
        model = fit_ar(a[:100], .01)
        x, y = windows(a, 100, 200, 20)
        np.testing.assert_allclose(predict_ar(model, x), y)
        r = summary_error(y+2, y, y+1)
        self.assertAlmostEqual(r['mean_bias_fraction_of_sse'], 1.)
        self.assertAlmostEqual(r['skill_vs_copy'], -3.)


if __name__ == '__main__':
    unittest.main()
