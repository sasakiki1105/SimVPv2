"""Proper significance for the trajectory selection.

The earlier z = -3.63 treated the 600 contiguous segments as if they were
independent draws.  They overlap by 200 of 201 frames, so that z-score is not
interpretable, and the selected segment is the argmin by construction, so its
"0th percentile" is vacuous.

The null has to reproduce the same selection procedure.  A circular time shift
of the native observation preserves its autocorrelation and its marginal
distribution but destroys the alignment with the fine record; applying the
identical argmin-over-segments (or Viterbi) search to each shifted sequence
gives the null distribution of the *selected* score.  The real alignment is
the zero-shift member, so its empirical rank among all shifts is an exact
permutation p-value.

Also reports a moving-block bootstrap CI on the mean observation distance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np

import evaluate_radaz_e20_native_fine_only_pilot as pilot
from experiment_radaz_trajectory_lifting import viterbi

WORKDIR = Path(r"C:\Users\astro\research\SimVPv2\workdirs\radaz_e20_native_g2_fine_only_pilot")
OUT = WORKDIR / "trajectory_lifting"
RNG = np.random.default_rng(20260904)
BLOCK = 20


def segment_scores(unary):
    frames, states = unary.shape
    steps = np.arange(frames)
    starts = np.arange(states - frames + 1)
    return np.array([unary[steps, s + steps].mean() for s in starts])


def main():
    protocol = json.loads((WORKDIR / "protocol.json").read_text(encoding="utf-8"))
    paths = protocol["paths"]
    low, high, _ = pilot.load_normalization(Path(paths["manifest"]))
    fine_train, _, _, _ = pilot.load_fine(Path(paths["fine_h5"]), 12.0, 23.985, True)
    native_test, _, _, _ = pilot.load_native(Path(paths["native_h5"]), 27.0, 30.0)

    pca, train_scores = pilot.fit_feature_pca(
        pilot.observation_features(fine_train[:, :, ::2, ::2], low, high), 32)
    native_scores = pca.transform(
        pilot.observation_features(native_test, low, high)) / np.sqrt(
        np.maximum(pca.explained_variance_[None], 1.0e-12))
    unary = pilot.distance_matrix(native_scores, train_scores)
    unary = unary / np.median(unary)
    frames, states = unary.shape
    steps = np.arange(frames)

    # ---- circular-shift permutation null ----
    shifts = np.arange(frames)
    seg_null = np.empty(frames)
    for k in shifts:
        seg_null[k] = segment_scores(unary[np.roll(steps, -k)]).min()
    real_seg = seg_null[0]
    seg_rank = int(np.sum(seg_null <= real_seg))
    seg_p = seg_rank / len(seg_null)

    print("=" * 96)
    print("CIRCULAR-SHIFT PERMUTATION NULL  (n = %d shifts, shift 0 = real alignment)"
          % frames)
    print("=" * 96)
    print("best contiguous segment")
    print("  real (shift 0)      : %.4f" % real_seg)
    print("  null min / p05 / med: %.4f / %.4f / %.4f"
          % (seg_null[1:].min(), np.percentile(seg_null[1:], 5),
             np.median(seg_null[1:])))
    print("  empirical rank      : %d of %d   ->  p = %.4f"
          % (seg_rank, len(seg_null), seg_p))

    # Viterbi path, lambda = 1 (the pre-registered choice), same null
    lam = 1.0
    path_null = np.empty(frames)
    for k in shifts:
        shifted = unary[np.roll(steps, -k)]
        path = viterbi(shifted, lam)
        path_null[k] = shifted[steps, path].mean()
    real_path = path_null[0]
    path_rank = int(np.sum(path_null <= real_path))
    path_p = path_rank / len(path_null)
    print()
    print("Viterbi path, lambda = 1")
    print("  real (shift 0)      : %.4f" % real_path)
    print("  null min / p05 / med: %.4f / %.4f / %.4f"
          % (path_null[1:].min(), np.percentile(path_null[1:], 5),
             np.median(path_null[1:])))
    print("  empirical rank      : %d of %d   ->  p = %.4f"
          % (path_rank, len(path_null), path_p))

    # ---- moving-block bootstrap on the mean observation distance ----
    chosen = viterbi(unary, lam)
    per_frame = unary[steps, chosen]
    blocks = frames - BLOCK + 1
    draws = np.empty(2000)
    for i in range(len(draws)):
        picks = RNG.integers(0, blocks, size=int(np.ceil(frames / BLOCK)))
        sample = np.concatenate([per_frame[b: b + BLOCK] for b in picks])[:frames]
        draws[i] = sample.mean()
    print()
    print("=" * 96)
    print("MOVING-BLOCK BOOTSTRAP  (block = %d frames, 2000 draws)" % BLOCK)
    print("=" * 96)
    print("  mean observation distance : %.4f" % per_frame.mean())
    print("  95%% CI                    : [%.4f, %.4f]"
          % (np.percentile(draws, 2.5), np.percentile(draws, 97.5)))
    print("  null median (shifted)     : %.4f" % np.median(path_null[1:]))

    # effective number of independent segments
    autocorr = np.correlate(per_frame - per_frame.mean(),
                            per_frame - per_frame.mean(), mode="full")
    autocorr = autocorr[len(per_frame) - 1:] / autocorr[len(per_frame) - 1]
    tau = 1.0 + 2.0 * np.sum(autocorr[1:np.argmax(autocorr < 0.05) + 1])
    print("  autocorrelation time      : %.1f frames -> effective n = %.1f"
          % (tau, frames / max(tau, 1.0)))

    (OUT / "selection_significance.json").write_text(json.dumps({
        "segment": {"real": float(real_seg), "rank": seg_rank, "p": seg_p,
                    "null_median": float(np.median(seg_null[1:]))},
        "viterbi_lambda1": {"real": float(real_path), "rank": path_rank,
                            "p": path_p,
                            "null_median": float(np.median(path_null[1:]))},
        "bootstrap_ci": [float(np.percentile(draws, 2.5)),
                         float(np.percentile(draws, 97.5))],
        "autocorrelation_time_frames": float(tau),
    }, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(OUT / "selection_significance.json"))


if __name__ == "__main__":
    main()
