"""Trajectory lifting with lambda selected the way the pilot selects k and T.

In the sweep, lambda was read off the native test window.  The pilot's own
protocol picks kNN hyperparameters on the *paired synthetic* validation window
(fine 24-27 us restricted to the native nodes, where the truth is the same
realization and a paired MSE is meaningful) and never touches the native test.
This applies that protocol to lambda, then evaluates once on native test.
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
from experiment_radaz_trajectory_lifting import viterbi

WORKDIR = Path(r"C:\Users\astro\research\SimVPv2\workdirs\radaz_e20_native_g2_fine_only_pilot")
OUT = WORKDIR / "trajectory_lifting"
LAMBDAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((WORKDIR / "protocol.json").read_text(encoding="utf-8"))
    paths = protocol["paths"]
    low, high, _ = pilot.load_normalization(Path(paths["manifest"]))

    fine_train, _, fine_x, fine_y = pilot.load_fine(
        Path(paths["fine_h5"]), 12.0, 23.985, True)
    fine_val, _, _, _ = pilot.load_fine(Path(paths["fine_h5"]), 24.0, 26.985, True)
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

    pca, train_scores = pilot.fit_feature_pca(
        pilot.observation_features(fine_train[:, :, ::2, ::2], low, high), 32)

    def scores(node_fields):
        return pca.transform(
            pilot.observation_features(node_fields, low, high)) / np.sqrt(
            np.maximum(pca.explained_variance_[None], 1.0e-12))

    # ---- selection on the paired synthetic validation window ----
    synthetic_val_node = fine_val[:, :, ::2, ::2]
    val_distance = pilot.distance_matrix(scores(synthetic_val_node), train_scores)
    val_unary = val_distance / np.median(val_distance)
    val_truth = fine_val[:, :, :257, :256]

    val_dt = dt_s

    def val_line_fraction(values, line):
        f, p = spectrum(n2_series(np.asarray(values, dtype=np.float32)[:, :, :256, :256]),
                        val_dt)
        m = np.abs(f) > 0.4
        return float(p[m][np.abs(f[m] - line) < 1.5].sum() / p[m].sum())

    fv, pv = spectrum(n2_series(fine_val[:, :, :256, :256]), val_dt)
    mv = np.abs(fv) > 0.4
    val_line = float(fv[mv][np.argmax(pv[mv])])
    val_line_truth = val_line_fraction(fine_val[:, :, :257, :256], val_line)

    print("=" * 96)
    print("SELECTION on paired synthetic validation (fine 24-27 us at native nodes)")
    print("validation truth: line at %.2f MHz, line fraction %.4f"
          % (val_line, val_line_truth))
    print("=" * 96)
    print("%-10s %14s %12s %14s %8s" % (
        "lambda", "paired MSE", "line frac", "|line gap|", "jumps"))
    print("-" * 96)
    selection = {}
    line_gap = {}
    for lam in LAMBDAS:
        path = viterbi(val_unary, lam)
        estimate = candidates[path]
        mse = pilot.paired_normalized_mse(val_truth, estimate, low, high)
        lf = val_line_fraction(estimate, val_line)
        selection[lam] = mse
        line_gap[lam] = abs(lf - val_line_truth)
        print("%-10g %14.6e %12.4f %14.4f %8d" % (
            lam, mse, lf, line_gap[lam], int(np.sum(np.diff(path) != 1))))

    pilot_lambda = min(selection, key=selection.get)
    # Declared criterion for this experiment: the frame-wise method fails on
    # temporal coherence, so select on the validation temporal line-fraction
    # gap, ties broken by paired MSE.  Both are computed on the paired
    # synthetic validation window; native test is not consulted.
    best_lambda = min(LAMBDAS, key=lambda l: (round(line_gap[l], 4), selection[l]))
    print()
    print("pilot criterion   (paired MSE only) -> lambda = %g" % pilot_lambda)
    print("declared criterion (validation line gap, ties by MSE) -> lambda = %g"
          % best_lambda)
    print("native test was not consulted for either choice")

    # ---- single evaluation on native test ----
    test_distance = pilot.distance_matrix(scores(native_test), train_scores)
    test_unary = test_distance / np.median(test_distance)
    steps = np.arange(test_unary.shape[0])

    truth_series = n2_series(reference)
    f0, p0 = spectrum(truth_series, dt_s)
    keep = np.abs(f0) > 0.4
    line_freq = float(f0[keep][np.argmax(p0[keep])])

    def line_fraction(values):
        f, p = spectrum(n2_series(values[:, :, :256, :256]), dt_s)
        m = np.abs(f) > 0.4
        return float(p[m][np.abs(f[m] - line_freq) < 1.5].sum() / p[m].sum())

    def row(name, values, obs):
        core = np.asarray(values[:, :, :256, :256], dtype=np.float32)
        s = pilot.summarize_candidate(reference, core, dx, dy, dt_s)
        return {
            "name": name, "obs_distance": obs,
            "line_fraction": line_fraction(np.asarray(values, dtype=np.float32)),
            "poisson_median": s["poisson"]["relative_poisson_residual_median"],
            "ne_mode_log10_rmse": s["mode"]["electron_den"]["log10_power_rmse_n1_32"],
            "charge_n9_21": s["charge"]["mode"]["n9_21_power_ratio"],
            "charge_n1_6": s["charge"]["mode"]["n1_6_power_ratio"],
            "profile_l2_ne": s["profile"]["electron_den"]["relative_l2"],
            "coarse_nrmse": pilot.coarse_consistency(
                core, native_test, low, high)["normalized_rmse_all"],
        }

    path = viterbi(test_unary, best_lambda)
    rows = [row("trajectory lam=%g" % best_lambda, candidates[path],
                float(np.mean(test_unary[steps, path])))]
    idx8, w8 = pilot.neighbor_weights(test_distance, 8, 2.0)
    rows.append(row("framewise k=8 (publ.)",
                    pilot.weighted_fields(candidates, idx8, w8),
                    float(np.mean(np.sum(w8 * np.take_along_axis(
                        test_unary, idx8, axis=1), axis=1)))))
    rows.append(row("framewise k=1", candidates[np.argmin(test_distance, axis=1)],
                    float(np.mean(np.min(test_unary, axis=1)))))
    rows.append(row("native interpolation", native_interp[:, :, :257, :256],
                    float("nan")))
    rows.append(row("fine truth", fine_test[:, :, :257, :256], float("nan")))

    print()
    print("=" * 122)
    print("NATIVE TEST 27-30 us, evaluated once with the pre-selected lambda")
    print("=" * 122)
    print("%-22s %10s %10s %11s %10s %10s %10s %10s %11s" % (
        "variant", "obs dist", "line frac", "Poisson", "ne mode",
        "chg9-21", "chg1-6", "prof L2", "coarse nrmse"))
    print("-" * 122)
    for r in rows:
        print("%-22s %10.4f %10.4f %11.4g %10.4f %10.4f %10.4f %10.5f %11.5f" % (
            r["name"], r["obs_distance"], r["line_fraction"], r["poisson_median"],
            r["ne_mode_log10_rmse"], r["charge_n9_21"], r["charge_n1_6"],
            r["profile_l2_ne"], r["coarse_nrmse"]))

    print()
    print("Gates relative to the published framewise k=8 lifting:")
    base = rows[1]
    sel = rows[0]
    for label, key, better in (("Poisson", "poisson_median", "lower"),
                               ("mode RMSE", "ne_mode_log10_rmse", "lower"),
                               ("ECDI charge power", "charge_n9_21", "closer to 1"),
                               ("temporal line", "line_fraction", "higher"),
                               ("coarse agreement", "coarse_nrmse", "lower"),
                               ("profile L2", "profile_l2_ne", "lower")):
        a, b = sel[key], base[key]
        if better == "lower":
            ok = a <= b
        elif better == "higher":
            ok = a >= b
        else:
            ok = abs(a - 1.0) <= abs(b - 1.0)
        print("   %-20s %10.5g -> %-10.5g  %s" % (
            label, b, a, "PASS" if ok else "fail"))

    (OUT / "trajectory_prereg.json").write_text(json.dumps({
        "selection_paired_mse": {str(k): v for k, v in selection.items()},
        "selected_lambda": best_lambda,
        "true_line_MHz": line_freq,
        "native_test_rows": rows,
        "path": path.tolist(),
    }, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(OUT / "trajectory_prereg.json"))


if __name__ == "__main__":
    main()
