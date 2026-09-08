"""Controls for the trajectory-aware lifting result.

The trajectory variants output *real fine states*, so Poisson consistency,
full mode power and the correct 22.9 MHz line are satisfied by construction --
any contiguous chunk of the fine record has them.  Those metrics therefore
cannot, on their own, show that the lifting is using the observation.

Three controls decide whether the result is real lifting or replayed fine data.

  1. Segment ranking.  Where does the selected contiguous segment sit in the
     distribution of all 599 possible starts?  If it is typical, the
     observation contributed nothing.
  2. Shuffled-observation null.  Re-run the same path inference against a
     time-shuffled native sequence.  Metrics that stay just as good are
     metrics that any fine trajectory satisfies, not evidence of lifting.
  3. Observation agreement.  C_native(X_hat) against the real native frames,
     using the pilot's own coarse_consistency, which is the only metric that
     can distinguish a matched trajectory from an arbitrary one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np

import evaluate_radaz_e20_native_fine_only_pilot as pilot
from audit_radaz_lifting_temporal_spectrum import n2_series, spectrum
from experiment_radaz_trajectory_lifting import viterbi, best_contiguous

WORKDIR = Path(r"C:\Users\astro\research\SimVPv2\workdirs\radaz_e20_native_g2_fine_only_pilot")
OUT = WORKDIR / "trajectory_lifting"
RNG = np.random.default_rng(20260904)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((WORKDIR / "protocol.json").read_text(encoding="utf-8"))
    paths = protocol["paths"]
    low, high, _ = pilot.load_normalization(Path(paths["manifest"]))

    fine_train, _, fine_x, fine_y = pilot.load_fine(
        Path(paths["fine_h5"]), 12.0, 23.985, True)
    fine_test, test_time, _, _ = pilot.load_fine(
        Path(paths["fine_h5"]), 27.0, 30.0, True)
    native_test, _, native_x, native_y = pilot.load_native(
        Path(paths["native_h5"]), 27.0, 30.0)
    native_interp = pilot.interpolate_native_full_to_model(
        native_test, native_x, native_y[:128], fine_x, fine_y)

    candidates = fine_train[:, :, :257, :256]
    reference = fine_test[:, :, :256, :256]
    dx = float(np.median(np.diff(fine_x)))
    dy = float(np.median(np.diff(fine_y[:256])))
    dt_s = float(np.median(np.diff(test_time)))

    candidate_coarse = fine_train[:, :, ::2, ::2]
    pca, train_scores = pilot.fit_feature_pca(
        pilot.observation_features(candidate_coarse, low, high), 32)
    native_scores = pca.transform(
        pilot.observation_features(native_test, low, high)) / np.sqrt(
        np.maximum(pca.explained_variance_[None], 1.0e-12))
    distance = pilot.distance_matrix(native_scores, train_scores)
    unary = distance / np.median(distance)
    frames, states = unary.shape
    steps = np.arange(frames)

    # ---- control 1: where does the chosen contiguous segment rank? ----
    starts = np.arange(states - frames + 1)
    segment_cost = np.array([unary[steps, s + steps].mean() for s in starts])
    chosen = int(best_contiguous(unary)[0])
    chosen_cost = float(segment_cost[starts == chosen][0])
    percentile = float((segment_cost < chosen_cost).mean() * 100.0)
    print("=" * 100)
    print("CONTROL 1  segment ranking among all %d contiguous starts" % len(starts))
    print("=" * 100)
    print("  selected start        : %d   mean obs distance %.4f" % (chosen, chosen_cost))
    print("  percentile of selected: %.2f%%  (0 = best possible segment)" % percentile)
    print("  distribution          : min %.4f  p05 %.4f  median %.4f  max %.4f"
          % (segment_cost.min(), np.percentile(segment_cost, 5),
             np.median(segment_cost), segment_cost.max()))
    print("  z-score of selected   : %.2f"
          % ((chosen_cost - segment_cost.mean()) / segment_cost.std()))

    # ---- control 2: shuffled-observation null ----
    print()
    print("=" * 100)
    print("CONTROL 2  same inference against a time-shuffled native sequence")
    print("=" * 100)
    shuffled_order = RNG.permutation(frames)
    unary_shuffled = unary[shuffled_order]

    truth_series = n2_series(reference)
    f0, p0 = spectrum(truth_series, dt_s)
    keep = np.abs(f0) > 0.4
    line_freq = float(f0[keep][np.argmax(p0[keep])])

    def line_fraction(values):
        f, p = spectrum(n2_series(values[:, :, :256, :256]), dt_s)
        m = np.abs(f) > 0.4
        return float(p[m][np.abs(f[m] - line_freq) < 1.5].sum() / p[m].sum())

    def evaluate(name, path, unary_used, native_for_obs):
        values = candidates[path]
        core = np.asarray(values[:, :, :256, :256], dtype=np.float32)
        s = pilot.summarize_candidate(reference, core, dx, dy, dt_s)
        consistency = pilot.coarse_consistency(core, native_for_obs, low, high)
        return {
            "name": name,
            "obs_distance": float(np.mean(unary_used[steps, path])),
            "line_fraction": line_fraction(np.asarray(values, dtype=np.float32)),
            "poisson_median": s["poisson"]["relative_poisson_residual_median"],
            "ne_mode_log10_rmse": s["mode"]["electron_den"]["log10_power_rmse_n1_32"],
            "charge_n9_21": s["charge"]["mode"]["n9_21_power_ratio"],
            "profile_l2_ne": s["profile"]["electron_den"]["relative_l2"],
            "coarse_nrmse": consistency["normalized_rmse_all"],
        }

    rows = []
    for lam in (0.1, 1.0):
        real_path = viterbi(unary, lam)
        rows.append(evaluate("real obs  lam=%g" % lam, real_path, unary, native_test))
        shuffled_path = viterbi(unary_shuffled, lam)
        # score the shuffled-fit path against the REAL observation order
        rows.append(evaluate("shuffled  lam=%g" % lam, shuffled_path, unary, native_test))
    random_path = np.sort(RNG.integers(0, states - frames, size=1)[0] + steps)
    rows.append(evaluate("random contiguous", random_path, unary, native_test))
    rows.append(evaluate("framewise k=1", np.argmin(distance, axis=1), unary, native_test))

    idx8, w8 = pilot.neighbor_weights(distance, 8, 2.0)
    k8 = pilot.weighted_fields(candidates, idx8, w8)
    core8 = np.asarray(k8[:, :, :256, :256], dtype=np.float32)
    s8 = pilot.summarize_candidate(reference, core8, dx, dy, dt_s)
    rows.append({
        "name": "framewise k=8 (publ.)",
        "obs_distance": float(np.mean(np.sum(
            w8 * np.take_along_axis(unary, idx8, axis=1), axis=1))),
        "line_fraction": line_fraction(np.asarray(k8, dtype=np.float32)),
        "poisson_median": s8["poisson"]["relative_poisson_residual_median"],
        "ne_mode_log10_rmse": s8["mode"]["electron_den"]["log10_power_rmse_n1_32"],
        "charge_n9_21": s8["charge"]["mode"]["n9_21_power_ratio"],
        "profile_l2_ne": s8["profile"]["electron_den"]["relative_l2"],
        "coarse_nrmse": pilot.coarse_consistency(
            core8, native_test, low, high)["normalized_rmse_all"],
    })
    native_core = np.asarray(native_interp[:, :, :256, :256], dtype=np.float32)
    sn = pilot.summarize_candidate(reference, native_core, dx, dy, dt_s)
    rows.append({
        "name": "native interpolation",
        "obs_distance": float("nan"),
        "line_fraction": line_fraction(native_interp[:, :, :257, :256]),
        "poisson_median": sn["poisson"]["relative_poisson_residual_median"],
        "ne_mode_log10_rmse": sn["mode"]["electron_den"]["log10_power_rmse_n1_32"],
        "charge_n9_21": sn["charge"]["mode"]["n9_21_power_ratio"],
        "profile_l2_ne": sn["profile"]["electron_den"]["relative_l2"],
        "coarse_nrmse": pilot.coarse_consistency(
            native_core, native_test, low, high)["normalized_rmse_all"],
    })

    print()
    print("%-22s %10s %10s %11s %10s %11s %11s %11s" % (
        "variant", "obs dist", "line frac", "Poisson", "ne mode",
        "chg n9-21", "prof L2", "coarse nrmse"))
    print("-" * 110)
    for r in rows:
        print("%-22s %10.4f %10.4f %11.4g %10.4f %11.4f %11.5f %11.5f" % (
            r["name"], r["obs_distance"], r["line_fraction"], r["poisson_median"],
            r["ne_mode_log10_rmse"], r["charge_n9_21"], r["profile_l2_ne"],
            r["coarse_nrmse"]))
    print()
    print("coarse nrmse = C_native(X_hat) against the real native frames.  This is")
    print("the only column that can tell a matched trajectory from an arbitrary one:")
    print("Poisson / mode power / line fraction are satisfied by ANY fine trajectory.")

    (OUT / "trajectory_controls.json").write_text(json.dumps({
        "segment_percentile": percentile,
        "segment_zscore": float((chosen_cost - segment_cost.mean()) / segment_cost.std()),
        "segment_cost_distribution": {
            "min": float(segment_cost.min()), "p05": float(np.percentile(segment_cost, 5)),
            "median": float(np.median(segment_cost)), "max": float(segment_cost.max()),
            "selected": chosen_cost},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print("[written] " + str(OUT / "trajectory_controls.json"))


if __name__ == "__main__":
    main()
