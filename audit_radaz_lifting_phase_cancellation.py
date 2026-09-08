"""Audit 2/3: does phase cancellation explain Experiment C's mode loss, and
can phase-aligned lifting recover it without breaking Poisson?

Experiment C lifts by a convex combination of fine candidate frames,

    X_hat = sum_j w_j X_j ,   w_j >= 0,  sum_j w_j = 1.

For azimuthal Fourier mode n of any channel this is a vector sum of complex
numbers z_j = A_j exp(i theta_j).  If the candidates carry different wave
phases the sum cancels, and the surviving fraction is the amplitude-weighted
circular resultant

    R_n = |sum_j w_j z_j| / sum_j w_j |z_j|  in [0, 1],

so the predicted power attenuation is R_n^2.  Test 1 checks that R_n^2
actually predicts the measured mode-power deficit.

Test 2 removes the cancellation without touching the weights.  The discrete
Poisson operator is diagonal in azimuthal Fourier modes -- (d2/dx2 - k_n^2)
phi_n = -rho_n/eps0 holds mode by mode and never mixes n -- so multiplying
mode n of *every* channel by one common unit complex number leaves the Poisson
residual exactly unchanged.  Azimuthal phase is a gauge direction of the
constraint.  Two alignments are therefore possible:

  rigid   : one azimuthal shift per candidate, chosen to match the native
            target's n=2 phase.  A physically realisable translated state, but
            it aligns only n=2; mode n rotates n/2 times as far.
  permode : each mode n rotated independently onto the native target's phase.
            Not a translated state, so it is an upper bound rather than a
            physical proposal -- but it is still exactly Poisson-consistent.

The native coarse observation resolves n <= 64, so its phases are available at
test time.  The current lifting throws that information away by averaging.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np

import evaluate_radaz_e20_native_fine_only_pilot as pilot

WORKDIR = Path(r"C:\Users\astro\research\SimVPv2\workdirs\radaz_e20_native_g2_fine_only_pilot")
BANDS = {"n2": (2, 2), "MTSI n1-6": (1, 6), "ECDI n9-21": (9, 21), "n1-32": (1, 32)}
ALIGN_MODE_MAX = 32   # native G2 resolves n <= 64; align only where it is reliable


def rfft_az(fields):
    """[..., 257, 256] real -> [..., 257, 129] complex, azimuthal transform."""
    return np.fft.rfft(np.asarray(fields, dtype=np.float64), axis=-1, norm="forward")


def irfft_az(coefficients):
    return np.fft.irfft(coefficients, n=256, axis=-1, norm="forward")


def band_power(coefficients, lo, hi):
    """Mean power over modes lo..hi and over radius/frames."""
    return float(np.mean(np.abs(coefficients[..., lo:hi + 1]) ** 2))


def main():
    protocol = json.loads((WORKDIR / "protocol.json").read_text(encoding="utf-8"))
    paths = protocol["paths"]
    low, high, _ = pilot.load_normalization(Path(paths["manifest"]))

    fine_train, train_time, fine_x, fine_y = pilot.load_fine(
        Path(paths["fine_h5"]), 12.0, 23.985, True)
    fine_test, test_time, _, _ = pilot.load_fine(
        Path(paths["fine_h5"]), 27.0, 30.0, True)
    native_test, native_time, native_x, native_y = pilot.load_native(
        Path(paths["native_h5"]), 27.0, 30.0)
    native_interp = pilot.interpolate_native_full_to_model(
        native_test, native_x, native_y[:128], fine_x, fine_y)

    candidates = fine_train[:, :, :257, :256]
    reference = fine_test[:, :, :256, :256]
    native_core = native_interp[:, :, :256, :256]

    saved = np.load(WORKDIR / "experiment_C_neighbor_weights.npz")
    indices = saved["combined_indices"].astype(np.int64)
    weights = saved["combined_weights"].astype(np.float64)
    frames = len(indices)
    print("candidates=%d  test frames=%d  neighbours=%d"
          % (len(candidates), frames, indices.shape[1]))

    dx = float(np.median(np.diff(fine_x)))
    dy = float(np.median(np.diff(fine_y[:256])))
    dt_s = float(np.median(np.diff(test_time)))

    target_c = rfft_az(native_interp[:, :, :257, :256])   # native observation
    truth_c = rfft_az(fine_test[:, :, :257, :256])

    plain = np.empty((frames, 3, 257, 256), dtype=np.float32)
    rigid = np.empty_like(plain)
    permode = np.empty_like(plain)
    selfalign = np.empty_like(plain)
    resultant = np.zeros((frames, 129), dtype=np.float64)
    incoherent = np.zeros((frames, 129), dtype=np.float64)
    aligned_resultant = np.zeros((frames, 129), dtype=np.float64)

    ky = 2.0 * np.pi * np.arange(129) / (256 * dy)

    for t in range(frames):
        picked = candidates[indices[t]]                    # [k,3,257,256]
        w = weights[t][:, None, None, None]
        coefficients = rfft_az(picked)                     # [k,3,257,129]

        plain[t] = np.sum(w * picked, axis=0)

        # --- Test 1: amplitude-weighted circular resultant, electron density
        z = coefficients[:, 0]                             # [k,257,129]
        wr = weights[t][:, None, None]
        vector_sum = np.abs(np.sum(wr * z, axis=0))
        scalar_sum = np.sum(wr * np.abs(z), axis=0)
        # radius-average with the incoherent power as weight
        resultant[t] = np.sum(vector_sum ** 2, axis=0)
        incoherent[t] = np.sum(scalar_sum ** 2, axis=0)

        # Phase offsets are estimated by radial cross-correlation, not by the
        # argument of the radial sum: that sum is itself phase-cancelled, so
        # its argument is a noisy estimator and rotating by it destroys the
        # coherence the candidates already had.
        #   alpha_j(n) = arg sum_r z_j(r,n) conj(z_ref(r,n))
        reference_mode = target_c[t, 0]                              # [257,129]
        overlap = np.sum(coefficients[:, 0] * np.conj(reference_mode)[None],
                         axis=1)                                     # [k,129]
        alpha = -np.angle(overlap)                                   # [k,129]

        # --- Test 2a: rigid azimuthal shift, set by the n=2 overlap phase.
        # A shift by xi rotates mode n by n*(alpha_2/2): one physical translation.
        modes = np.arange(129)
        rotation = np.exp(1j * alpha[:, 2:3] * (modes[None, :] / 2.0))
        rigid[t] = np.sum(
            w * irfft_az(coefficients * rotation[:, None, None, :]), axis=0)

        # --- Test 2b: per-mode alignment (upper bound; still Poisson-exact)
        phase = alpha.copy()
        phase[:, 0] = 0.0
        phase[:, ALIGN_MODE_MAX + 1:] = 0.0
        rotation = np.exp(1j * phase)
        aligned = coefficients * rotation[:, None, None, :]
        permode[t] = np.sum(w * irfft_az(aligned), axis=0)

        # resultant after per-mode alignment, same definition as Test 1
        za = aligned[:, 0]
        aligned_resultant[t] = np.sum(
            np.abs(np.sum(wr * za, axis=0)) ** 2, axis=0)

        # --- Test 2c: align to the top-weighted candidate instead of the
        # native target.  Fine and native G2 are independent realizations, so
        # the target's instantaneous phase is unrelated to the candidates';
        # using the heaviest candidate as the common reference removes that
        # dependence and gives the cleanest coherence upper bound.
        self_ref = coefficients[0, 0]                                # [257,129]
        self_overlap = np.sum(coefficients[:, 0] * np.conj(self_ref)[None],
                              axis=1)
        self_phase = -np.angle(self_overlap)
        self_phase[:, 0] = 0.0
        self_phase[:, ALIGN_MODE_MAX + 1:] = 0.0
        selfalign[t] = np.sum(
            w * irfft_az(coefficients * np.exp(1j * self_phase)[:, None, None, :]),
            axis=0)

        if (t + 1) % 50 == 0:
            print("  aligned %d/%d" % (t + 1, frames), flush=True)

    # ---------------- Test 1 report ----------------
    print()
    print("=" * 108)
    print("TEST 1  phase cancellation: does R_n^2 predict the measured mode deficit?")
    print("=" * 108)
    # Aggregate power-weighted (sum of surviving power over sum of incoherent
    # power), NOT as an unweighted mean of per-mode ratios: the measured band
    # ratio is power-weighted, so a mode-uniform mean is not comparable to it.
    total_resultant = resultant.sum(axis=0)
    total_incoherent = incoherent.sum(axis=0)
    total_aligned = aligned_resultant.sum(axis=0)
    plain_c = rfft_az(plain)
    cand_mean_power = np.zeros(129)
    for t in range(frames):
        cand_mean_power += np.sum(
            weights[t][:, None, None]
            * np.abs(rfft_az(candidates[indices[t]])[:, 0]) ** 2, axis=0).mean(axis=0)
    cand_mean_power /= frames

    print("%-12s %10s %10s %10s %10s %12s" % (
        "band", "R^2 pred", "measured", "cand/truth", "ratio", "R^2 aligned"))
    print("-" * 108)
    rows = []
    for band, (lo, hi) in BANDS.items():
        pred = float(total_resultant[lo:hi + 1].sum()
                     / max(total_incoherent[lo:hi + 1].sum(), 1e-300))
        after = float(total_aligned[lo:hi + 1].sum()
                      / max(total_incoherent[lo:hi + 1].sum(), 1e-300))
        measured = band_power(plain_c[:, 0], lo, hi) / band_power(truth_c[:, 0], lo, hi)
        cand_ratio = float(np.mean(cand_mean_power[lo:hi + 1])) \
            / band_power(truth_c[:, 0], lo, hi)
        rows.append((band, pred, measured, cand_ratio, after))
        print("%-12s %10.4f %10.4f %10.4f %10.4f %12.4f" % (
            band, pred, measured, cand_ratio,
            measured / max(pred * cand_ratio, 1e-12), after))
    print()
    print("R^2 pred  = amplitude-weighted circular resultant squared, from the")
    print("            saved neighbour weights alone (no reference to truth).")
    print("measured  = lifted mode power / fine-truth mode power.")
    print("cand/truth= candidate power / truth power, i.e. the input level.")
    print("ratio     = measured / (R^2 * cand/truth); 1.0 means cancellation")
    print("            fully accounts for the deficit.")

    # ---------------- Test 2 report ----------------
    print()
    print("=" * 108)
    print("TEST 2  phase-aligned lifting (identical weights, Poisson-preserving rotations)")
    print("=" * 108)
    variants = {
        "native interpolation": native_core,
        "lifting (published)": plain[:, :, :256, :256],
        "lifting + rigid n=2 align": rigid[:, :, :256, :256],
        "lifting + per-mode align": permode[:, :, :256, :256],
        "per-mode, self-reference": selfalign[:, :, :256, :256],
    }
    summaries = {"fine_reference": pilot.summarize_candidate(
        reference, reference, dx, dy, dt_s)}
    for name, values in variants.items():
        summaries[name] = pilot.summarize_candidate(
            reference, np.asarray(values, dtype=np.float32), dx, dy, dt_s)

    def show(label, getter):
        print("%-30s" % label
              + " ".join("%13s" % ("%.4g" % getter(summaries[n]))
                         for n in ["fine_reference"] + list(variants)))

    header = ["truth"] + [n[:14] for n in variants]
    print("%-30s" % "metric" + " ".join("%13s" % h for h in header))
    print("-" * 116)
    show("Poisson residual (median)",
         lambda s: s["poisson"]["relative_poisson_residual_median"])
    show("mode log10 RMSE  ne",
         lambda s: s["mode"]["electron_den"]["log10_power_rmse_n1_32"])
    show("mode log10 RMSE  phi",
         lambda s: s["mode"]["phi"]["log10_power_rmse_n1_32"])
    show("charge mode log10 RMSE",
         lambda s: s["charge"]["mode"]["log10_power_rmse_n1_32"])
    show("charge n9-21 power ratio",
         lambda s: s["charge"]["mode"]["n9_21_power_ratio"])
    show("charge n1-6 power ratio",
         lambda s: s["charge"]["mode"]["n1_6_power_ratio"])
    show("phi n=2 peak freq [MHz]",
         lambda s: s["temporal"]["candidate_peak_frequency_MHz"])
    show("phi n=2 RMS ratio", lambda s: s["temporal"]["phi_n2_rms_ratio"])
    show("profile rel L2  ne",
         lambda s: s["profile"]["electron_den"]["relative_l2"])

    # direct mode-power ratios of the lifted electron density
    print()
    print("%-30s" % "ne band power / truth"
          + " ".join("%13s" % h for h in header))
    for band, (lo, hi) in BANDS.items():
        values = []
        for n in ["fine_reference"] + list(variants):
            arr = reference if n == "fine_reference" else np.asarray(variants[n])
            c = rfft_az(arr)
            values.append(band_power(c[:, 0], lo, hi)
                          / band_power(truth_c[:, 0], lo, hi))
        print("%-30s" % ("  " + band)
              + " ".join("%13.4f" % v for v in values))

    payload = {
        "test1_resultant": {b: {"R2_predicted": p, "measured": m,
                                "cand_over_truth": c, "R2_after_alignment": a}
                            for b, p, m, c, a in rows},
        "test2_summaries": pilot.json_safe(summaries),
    }
    out = WORKDIR / "phase_cancellation_audit.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(out))


if __name__ == "__main__":
    main()
