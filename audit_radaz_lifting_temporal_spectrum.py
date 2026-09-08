"""Audit 3b: is the 22.67 MHz oscillation absent from the lifted sequence, or
present but not dominant?

summarize_candidate reports the argmax of the phi n=2 temporal PSD, which is a
fragile statistic: a large slow drift can win the argmax while the fast
oscillation is still there.  This measures the PSD directly at the true line
and reports the fraction of band power it carries, for every lifting variant.

Re-runs the same alignments as audit_radaz_lifting_phase_cancellation.py, so
the variants are identical; only the diagnostic differs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np

import evaluate_radaz_e20_native_fine_only_pilot as pilot
from audit_radaz_lifting_phase_cancellation import rfft_az, irfft_az, ALIGN_MODE_MAX

WORKDIR = Path(r"C:\Users\astro\research\SimVPv2\workdirs\radaz_e20_native_g2_fine_only_pilot")


def n2_series(fields):
    """Radially integrated azimuthal n=2 coefficient of phi, per frame."""
    coefficients = rfft_az(np.asarray(fields, dtype=np.float64)[:, 2, :256, :])
    return np.sum(coefficients[..., 2], axis=-1)          # [T] complex


def spectrum(series, dt_s):
    """Two-sided spectrum.  The series is COMPLEX (the azimuthal n=2 Fourier
    coefficient), and the wave travels in one azimuthal direction, so its line
    sits at a *negative* frequency.  Masking to freq > 0 discards the dominant
    half and leaves only the residual slow content -- the mistake that produced
    the spurious "1.33 MHz" reading."""
    values = np.asarray(series) - np.mean(series)
    window = np.hanning(len(values))
    transformed = np.fft.fft(values * window)
    freq = np.fft.fftfreq(len(values), d=dt_s) * 1.0e-6   # MHz
    order = np.argsort(freq)
    return freq[order], np.abs(transformed[order]) ** 2


def report(name, series, dt_s, truth_freq):
    freq, power = spectrum(series, dt_s)
    keep = np.abs(freq) > 0.4
    f = freq[keep]
    p = power[keep]
    total = p.sum()
    peak = f[np.argmax(p)]
    # power within +-1.5 MHz of the true line (same sign), and the slow band
    line = p[np.abs(f - truth_freq) < 1.5].sum() / total
    slow = p[np.abs(f) < 5.0].sum() / total
    print("%-28s %10.2f %12.4f %12.4f" % (name, peak, line, slow))
    return {"peak_MHz": float(peak), "line_fraction": float(line),
            "slow_fraction": float(slow)}


def main():
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
    saved = np.load(WORKDIR / "experiment_C_neighbor_weights.npz")
    indices = saved["combined_indices"].astype(np.int64)
    weights = saved["combined_weights"].astype(np.float64)
    frames = len(indices)
    dt_s = float(np.median(np.diff(test_time)))

    target_c = rfft_az(native_interp[:, :, :257, :256])

    plain = np.empty((frames, 3, 257, 256), dtype=np.float32)
    permode = np.empty_like(plain)
    selfalign = np.empty_like(plain)
    nearest = np.empty_like(plain)

    for t in range(frames):
        picked = candidates[indices[t]]
        w = weights[t][:, None, None, None]
        coefficients = rfft_az(picked)
        plain[t] = np.sum(w * picked, axis=0)
        nearest[t] = picked[0]                       # k=1 control, no averaging

        overlap = np.sum(coefficients[:, 0] * np.conj(target_c[t, 0])[None], axis=1)
        phase = -np.angle(overlap)
        phase[:, 0] = 0.0
        phase[:, ALIGN_MODE_MAX + 1:] = 0.0
        permode[t] = np.sum(
            w * irfft_az(coefficients * np.exp(1j * phase)[:, None, None, :]), axis=0)

        self_overlap = np.sum(
            coefficients[:, 0] * np.conj(coefficients[0, 0])[None], axis=1)
        self_phase = -np.angle(self_overlap)
        self_phase[:, 0] = 0.0
        self_phase[:, ALIGN_MODE_MAX + 1:] = 0.0
        selfalign[t] = np.sum(
            w * irfft_az(coefficients * np.exp(1j * self_phase)[:, None, None, :]),
            axis=0)

    truth_series = n2_series(fine_test[:, :, :256, :256])
    truth_freq, truth_power = spectrum(truth_series, dt_s)
    keep = np.abs(truth_freq) > 0.4
    line_freq = float(truth_freq[keep][np.argmax(truth_power[keep])])

    print("=" * 76)
    print("phi n=2 temporal spectrum; true line at %.2f MHz" % line_freq)
    print("=" * 76)
    print("%-28s %10s %12s %12s" % ("series", "peak MHz", "line frac", "<5MHz frac"))
    print("-" * 76)
    collected = {}
    collected["fine truth"] = report("fine truth", truth_series, dt_s, line_freq)
    collected["native interpolation"] = report(
        "native interpolation", n2_series(native_interp[:, :, :256, :256]),
        dt_s, line_freq)
    for name, values in (("lifting (published)", plain),
                         ("lifting + per-mode align", permode),
                         ("per-mode, self-reference", selfalign),
                         ("nearest neighbour k=1", nearest)):
        collected[name] = report(name, n2_series(values[:, :, :256, :256]),
                                 dt_s, line_freq)
    print()
    print("line frac  = share of |f|>0.4 MHz power within +-1.5 MHz of the true line")
    print("<5MHz frac = share carried by the slow band")
    print()
    print("k=1 is the decisive control: it does no averaging at all, so if its")
    print("line fraction is also low the temporal failure is not caused by")
    print("averaging but by the neighbour *sequence* being slowly varying.")

    (WORKDIR / "temporal_spectrum_audit.json").write_text(
        json.dumps(collected, indent=2), encoding="utf-8")
    print("[written] " + str(WORKDIR / "temporal_spectrum_audit.json"))


if __name__ == "__main__":
    main()
