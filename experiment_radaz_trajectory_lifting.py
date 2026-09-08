"""Priority 1: trajectory-aware coarse lifting.

Audit 3b showed the temporal failure is NOT caused by averaging -- k=1 nearest
neighbour, which averages nothing, loses the 22.9 MHz line just as badly.  The
cause is that every frame is projected onto the fine manifold independently,
so the selected states j_t and j_{t+1} need not be connected by fine dynamics.

This replaces frame-wise selection by a path inference

    J = sum_t D_obs( C_native X_{j_t}, Y_t ) + lam * sum_t P( j_{t+1} - j_t )

with P(1) = 0 and P(d) = min(|d-1|, cap) otherwise, solved exactly by Viterbi
over the 800 candidate frames.  lam = 0 reproduces per-frame nearest
neighbour; lam -> infinity forces one contiguous segment of the real fine
record, which by construction satisfies Poisson, carries full mode power and
has the correct temporal spectrum.  The sweep therefore measures what
observation agreement costs to buy temporal coherence.

Distances use the observation-only path (alpha_latent = 0) because the latent
path needs the encoder and the GPU is busy; the published observation_only
variant is the matched baseline.
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

WORKDIR = Path(r"C:\Users\astro\research\SimVPv2\workdirs\radaz_e20_native_g2_fine_only_pilot")
OUT = WORKDIR / "trajectory_lifting"
LAMBDAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
JUMP_CAP = 50.0


def viterbi(unary, lam, cap=JUMP_CAP):
    """unary [T,N] -> best path [T]; transition P(d)=min(|d-1|,cap), 0 at d=1."""
    frames, states = unary.shape
    offsets = np.arange(states)[None, :] - np.arange(states)[:, None]   # j' - j
    transition = lam * np.minimum(np.abs(offsets - 1), cap)
    cost = unary[0].copy()
    back = np.empty((frames, states), dtype=np.int32)
    for t in range(1, frames):
        total = cost[:, None] + transition
        best = np.argmin(total, axis=0)
        back[t] = best
        cost = total[best, np.arange(states)] + unary[t]
    path = np.empty(frames, dtype=np.int64)
    path[-1] = int(np.argmin(cost))
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def best_contiguous(unary):
    """lam -> infinity limit: the best strictly consecutive segment."""
    frames, states = unary.shape
    best_start, best_cost = None, np.inf
    for start in range(states - frames + 1):
        c = unary[np.arange(frames), start + np.arange(frames)].sum()
        if c < best_cost:
            best_cost, best_start = c, start
    return best_start + np.arange(frames)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((WORKDIR / "protocol.json").read_text(encoding="utf-8"))
    paths = protocol["paths"]
    low, high, _ = pilot.load_normalization(Path(paths["manifest"]))

    fine_train, train_time, fine_x, fine_y = pilot.load_fine(
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

    # --- observation distances, exactly as the pilot builds them (alpha = 0)
    candidate_coarse = fine_train[:, :, ::2, ::2]
    observation_train = pilot.observation_features(candidate_coarse, low, high)
    observation_native = pilot.observation_features(native_test, low, high)
    pca, train_scores = pilot.fit_feature_pca(observation_train, 32)
    native_scores = pca.transform(observation_native) / np.sqrt(
        np.maximum(pca.explained_variance_[None], 1.0e-12))
    distance = pilot.distance_matrix(native_scores, train_scores)   # [201,800]
    print("distance matrix", distance.shape)

    saved = np.load(WORKDIR / "experiment_C_neighbor_weights.npz")
    reproduced = np.argmin(distance, axis=1)
    agreement = float(np.mean(reproduced == saved["observation_only_indices"][:, 0]))
    print("top-1 agreement with the saved observation_only selection: %.3f" % agreement)

    unary = distance / np.median(distance)
    frames = distance.shape[0]

    variants = {}
    paths_used = {}

    # frame-wise controls
    idx8, w8 = pilot.neighbor_weights(distance, 8, 2.0)
    variants["framewise k=8"] = pilot.weighted_fields(candidates, idx8, w8)
    variants["framewise k=1"] = candidates[np.argmin(distance, axis=1)]
    paths_used["framewise k=1"] = np.argmin(distance, axis=1)

    for lam in LAMBDAS:
        path = viterbi(unary, lam)
        name = "trajectory lam=%g" % lam
        variants[name] = candidates[path]
        paths_used[name] = path
        print("  %-22s jumps=%3d  mean obs dist=%.4f" % (
            name, int(np.sum(np.diff(path) != 1)),
            float(np.mean(unary[np.arange(frames), path]))), flush=True)

    contiguous = best_contiguous(unary)
    variants["contiguous segment"] = candidates[contiguous]
    paths_used["contiguous segment"] = contiguous
    print("  %-22s start frame %d, mean obs dist=%.4f" % (
        "contiguous", int(contiguous[0]),
        float(np.mean(unary[np.arange(frames), contiguous]))))

    # --- evaluate
    truth_series = n2_series(reference)
    freq, power = spectrum(truth_series, dt_s)
    keep = np.abs(freq) > 0.4
    line_freq = float(freq[keep][np.argmax(power[keep])])

    def line_fraction(values):
        f, p = spectrum(n2_series(values[:, :, :256, :256]), dt_s)
        m = np.abs(f) > 0.4
        return float(p[m][np.abs(f[m] - line_freq) < 1.5].sum() / p[m].sum())

    summaries = {"fine truth": pilot.summarize_candidate(
        reference, reference, dx, dy, dt_s)}
    lines = {"fine truth": line_fraction(fine_test[:, :, :257, :256])}
    obs = {"fine truth": float("nan")}
    for name, values in variants.items():
        core = np.asarray(values[:, :, :256, :256], dtype=np.float32)
        summaries[name] = pilot.summarize_candidate(reference, core, dx, dy, dt_s)
        lines[name] = line_fraction(np.asarray(values, dtype=np.float32))
        if name in paths_used:
            obs[name] = float(np.mean(unary[np.arange(frames), paths_used[name]]))
        else:
            obs[name] = float(np.mean(np.sum(w8 * np.take_along_axis(
                unary, idx8, axis=1), axis=1)))
    summaries["native interpolation"] = pilot.summarize_candidate(
        reference, np.asarray(native_interp[:, :, :256, :256], dtype=np.float32),
        dx, dy, dt_s)
    lines["native interpolation"] = line_fraction(native_interp[:, :, :257, :256])
    obs["native interpolation"] = float("nan")

    order = ["fine truth", "native interpolation", "framewise k=8", "framewise k=1"] \
        + ["trajectory lam=%g" % l for l in LAMBDAS] + ["contiguous segment"]

    print()
    print("=" * 122)
    print("TRAJECTORY-AWARE LIFTING  (true temporal line at %.2f MHz)" % line_freq)
    print("=" * 122)
    header = ("%-22s %10s %10s %11s %11s %11s %11s %11s"
              % ("variant", "obs dist", "line frac", "Poisson",
                 "ne n1-32", "chg n9-21", "chg n1-6", "prof L2 ne"))
    print(header)
    print("-" * 122)
    rows = {}
    for name in order:
        s = summaries[name]
        row = {
            "obs_distance": obs.get(name, float("nan")),
            "line_fraction": lines[name],
            "poisson_median": s["poisson"]["relative_poisson_residual_median"],
            "ne_mode_log10_rmse": s["mode"]["electron_den"]["log10_power_rmse_n1_32"],
            "charge_n9_21": s["charge"]["mode"]["n9_21_power_ratio"],
            "charge_n1_6": s["charge"]["mode"]["n1_6_power_ratio"],
            "profile_l2_ne": s["profile"]["electron_den"]["relative_l2"],
        }
        rows[name] = row
        print("%-22s %10.4f %10.4f %11.4g %11.4f %11.4f %11.4f %11.5f" % (
            name, row["obs_distance"], row["line_fraction"], row["poisson_median"],
            row["ne_mode_log10_rmse"], row["charge_n9_21"], row["charge_n1_6"],
            row["profile_l2_ne"]))
    print()
    print("obs dist  = mean observation mismatch along the chosen path (lower = matches")
    print("            the native coarse observation better)")
    print("line frac = share of phi n=2 temporal power at the true 22.9 MHz line")
    print("ne n1-32  = mode log10 power RMSE (lower better); charge ratios: 1.0 ideal")

    (OUT / "trajectory_lifting_summary.json").write_text(json.dumps({
        "true_line_MHz": line_freq,
        "top1_agreement_with_saved": agreement,
        "lambdas": LAMBDAS,
        "rows": rows,
        "paths": {k: v.tolist() for k, v in paths_used.items()},
    }, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(OUT / "trajectory_lifting_summary.json"))


if __name__ == "__main__":
    main()
