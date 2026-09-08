"""Audit 1: is the q-space mode skill an artifact of the q interpolation?

PEPAPICSpectralLoss._interpolate_to_q linearly interpolates the *complex*
coefficients of adjacent azimuthal modes n and n+1 onto the q = n/n0 grid.
Adjacent azimuthal modes have essentially uncorrelated phases, so

    (1-a) N_n + a N_{n+1}

can be much smaller than either term: the interpolation itself cancels phase.
Both prediction and truth pass through the same linear operator, so the loss
is still a valid quadratic form -- but it is blind to the error components in
the operator's null space, and the *target* it compares against is partly an
interpolation artifact.

Two things are measured here.

1.  Interpolation attenuation
        A = mean |interp(N)|^2 / mean [ (1-a)|N_n|^2 + a|N_{n+1}|^2 ]
    A = 1 means the interpolation preserves power; A << 1 means the q-space
    target is dominated by cancellation between neighbouring modes.

2.  Raw-n mode skill.  The same unexplained-power score reported earlier, but
    computed on fixed azimuthal modes with no q interpolation, over
    MTSI (n=1-6), ECDI (n=9-21) and the n0-centred band.  If the earlier
    "mode skill is real" result survives here, it was not an artifact.

Also reports how many distinct azimuthal modes the q grid actually covers per
condition, which varies with n0 and is a separate design issue.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    AFT, CELLS, MANIFEST, MODEL_H, MODEL_W, PRE, TEST_START, TEST_STOP,
    build_model, condition_vector, load_case_frames, predict_direct10,
)
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

CELL_ORDER = ["U-D", "C-D", "U-P", "C-P"]
BANDS = {"MTSI n1-6": (1, 6), "ECDI n9-21": (9, 21), "all n1-32": (1, 32)}


def band_coeff(loss_module, normalized, device):
    """[B,T,3,260,256] -> numpy complex128 [B,T,F,R,N], n = 1..max_mode,
    F = (phi, ne, Ey).  Returned on the host: this torch build cannot JIT
    complex/float64 elementwise kernels, and the array is small."""
    tensor = torch.from_numpy(normalized).to(device)
    real = loss_module._band_coefficients(tensor, physical_units=False)
    values = real.detach().cpu().numpy()
    return (values[..., 0] + 1j * values[..., 1]).astype(np.complex128)


def unexplained(pred, true, lo, hi):
    """Unexplained power over azimuthal modes lo..hi inclusive (1-indexed n)."""
    sl = slice(lo - 1, hi)
    err = np.abs(pred[..., sl] - true[..., sl]) ** 2
    power = np.abs(true[..., sl]) ** 2
    return float(np.sum(err) / max(np.sum(power), 1e-300))


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    device = torch.device("cuda:0")
    loss_module = PEPAPICSpectralLoss(
        data_root=str(MANIFEST), max_mode=64, radial_bands=4,
        radial_min_m=0.09e-2, radial_max_m=1.19e-2,
        coordinate_system="q_normalized", q_min=0.30, q_max=1.50, q_bins=49,
    ).to(device)
    q_grid = loss_module.q_grid.to(torch.float64).cpu().numpy()

    models = dict(
        (n, build_model(CELLS[n]["config"],
                        CELLS[n]["workdir"] / "checkpoints" / "best.ckpt", device)[0])
        for n in CELL_ORDER)
    starts = [s for s in range(TEST_START, TEST_STOP, PRE + AFT)
              if s + PRE + AFT <= TEST_STOP]

    print("=" * 118)
    print("A. q-grid coverage and interpolation attenuation")
    print("=" * 118)
    print("%-11s %8s %14s %10s %12s %12s" % (
        "case", "n0", "n range", "distinct n", "attn ne", "attn Ey"))
    print("-" * 118)

    collected = {}
    for case in manifest["cases"]:
        key = case["case_key"]
        condition, drift, mode_n0 = condition_vector(case, manifest)
        model_frames, _, _, _ = load_case_frames(
            case, low, high, TEST_START, TEST_START + len(starts) * (PRE + AFT))
        inputs = np.empty((len(starts), PRE, 5, MODEL_H, MODEL_W), dtype=np.float32)
        truth = np.empty((len(starts), AFT, 3, MODEL_H, MODEL_W), dtype=np.float32)
        for index, s in enumerate(starts):
            i0 = s - TEST_START
            inputs[index, :, :3] = model_frames[i0: i0 + PRE]
            inputs[index, :, 3:] = condition[None, :, None, None]
            truth[index] = model_frames[i0 + PRE: i0 + PRE + AFT]

        true_c = band_coeff(loss_module, truth, device)      # [B,T,F,R,N]
        n_lo = float(0.30 * mode_n0)
        n_hi = float(1.50 * mode_n0)
        distinct = int(np.floor(n_hi)) - int(np.ceil(n_lo)) + 1

        # interpolation attenuation, per field
        positions = mode_n0 * q_grid - 1.0
        lower = np.clip(np.floor(positions).astype(int), 0, true_c.shape[-1] - 1)
        upper = np.clip(lower + 1, 0, true_c.shape[-1] - 1)
        frac = np.clip(positions - lower, 0.0, 1.0)
        # tiny array ([B,T,R,Q]); do it in numpy to avoid float64 CUDA fusion
        attn = {}
        for field_index, field_name in ((1, "ne"), (2, "Ey")):
            values = true_c[:, :, field_index]
            a = values[..., lower]
            b = values[..., upper]
            interp = a + frac * (b - a)
            incoherent = (1.0 - frac) * np.abs(a) ** 2 + frac * np.abs(b) ** 2
            attn[field_name] = float(
                np.sum(np.abs(interp) ** 2) / max(np.sum(incoherent), 1e-300))
        print("%-11s %8.2f %14s %10d %12.4f %12.4f" % (
            key, mode_n0, "%.1f-%.1f" % (n_lo, n_hi), distinct,
            attn["ne"], attn["Ey"]))

        entry = {"n0": mode_n0, "role": case["role"], "distinct_modes": distinct,
                 "interp_attenuation": attn, "raw": {}}
        for name in CELL_ORDER:
            pred = predict_direct10(models[name], inputs, device)
            pred_c = band_coeff(loss_module, pred, device)
            entry["raw"][name] = dict(
                (band, unexplained(pred_c[:, :, 1], true_c[:, :, 1], lo, hi))
                for band, (lo, hi) in BANDS.items())
            entry["raw"][name].update(dict(
                ("Ey " + band, unexplained(pred_c[:, :, 2], true_c[:, :, 2], lo, hi))
                for band, (lo, hi) in BANDS.items()))
        collected[key] = entry
        del model_frames, inputs, truth, true_c

    print()
    print("attn = mean|interp|^2 / mean[incoherent interpolation of |N|^2].")
    print("1.0 = the q interpolation preserves power; << 1 = the q-space target")
    print("is dominated by cancellation between adjacent azimuthal modes.")

    skill = json.loads(
        (MANIFEST.parent.parent / "radaz_conditioned_factorial_evaluation"
         / "factorial_skill_scores.json").read_text(encoding="utf-8"))

    for band in BANDS:
        print()
        print("=" * 118)
        print("B. RAW-n unexplained power, electron density, %s  (no q interpolation)"
              % band)
        print("=" * 118)
        print("%-11s %-7s " % ("case", "role")
              + " ".join("%9s" % c for c in CELL_ORDER)
              + "   | q-space (for comparison)")
        print("-" * 118)
        for key, entry in collected.items():
            raw = " ".join("%9.3f" % entry["raw"][c][band] for c in CELL_ORDER)
            qsp = " ".join("%7.3f" % skill[key][c]["mode_unexplained"]
                           for c in CELL_ORDER)
            print("%-11s %-7s %s   | %s" % (key, entry["role"][:6], raw, qsp))
        for scope in ("source", "axis_holdout"):
            subset = [k for k in collected if collected[k]["role"] == scope]
            raw = " ".join("%9.3f" % np.median(
                [collected[k]["raw"][c][band] for k in subset]) for c in CELL_ORDER)
            qsp = " ".join("%7.3f" % np.median(
                [skill[k][c]["mode_unexplained"] for k in subset])
                for c in CELL_ORDER)
            print("%-11s %-7s %s   | %s" % ("MEDIAN", scope[:6], raw, qsp))

    out = (MANIFEST.parent.parent / "radaz_conditioned_factorial_evaluation"
           / "q_interpolation_audit.json")
    out.write_text(json.dumps(collected, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(out))


if __name__ == "__main__":
    main()
