"""Test whether the factorial cells beat copy on the q-complex-mode loss only
because they suppress azimuthal fluctuation amplitude.

The q complex-mode loss is an absolute MSE on complex Fourier coefficients.
A prediction with no azimuthal structure at all has every n>=1 coefficient
equal to zero, so its loss is exactly the true q-band power.  If that
"zero-fluctuation" baseline scores at or below the trained models, then
"complex_mode_loss / copy < 1" certifies smoothing, not mode recovery.

Adds, for the same protocol as evaluate_radaz_conditioned_factorial.py:
  * zero_fluctuation baseline (each frame replaced by its azimuthal mean)
  * amplitude_log_bias    mean log(|pred_q| / |true_q|)  (0 = calibrated)
  * fraction of true q-band power retained by each cell
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    AFT, CELLS, CLOSED_LOOP_BLOCKS, MANIFEST, MODEL_H, MODEL_W, PRE,
    TEST_START, TEST_STOP, VALID_H, VALID_W,
    build_model, condition_vector, load_case_frames, predict_direct10,
    q_observables,
)
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

CELL_ORDER = ["U-D", "C-D", "U-P", "C-P"]


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

    models = {}
    for name in CELL_ORDER:
        spec = CELLS[name]
        models[name] = build_model(
            spec["config"], spec["workdir"] / "checkpoints" / "best.ckpt", device)[0]

    starts = [s for s in range(TEST_START, TEST_STOP, PRE + AFT)
              if s + PRE + AFT <= TEST_STOP]

    print("=" * 118)
    print("Mode-collapse diagnostic: is complex_mode_loss/copy < 1 just amplitude suppression?")
    print("=" * 118)
    header = ("%-11s %-7s | %-10s %-10s | " % ("case", "role", "copy", "zero_flu")
              + " ".join("%-22s" % c for c in CELL_ORDER))
    print(header)
    print("%-11s %-7s | %-10s %-10s | " % ("", "", "cml", "cml/copy")
          + " ".join("%-22s" % "cml/copy  ampbias  pwr" for _ in CELL_ORDER))
    print("-" * 118)

    rows = []
    for case in manifest["cases"]:
        key = case["case_key"]
        condition, drift, mode_n0 = condition_vector(case, manifest)
        need_stop = max(TEST_START + len(starts) * (PRE + AFT),
                        TEST_START + PRE + CLOSED_LOOP_BLOCKS * AFT)
        model_frames, _, _, _ = load_case_frames(case, low, high, TEST_START, need_stop)

        inputs = np.empty((len(starts), PRE, 5, MODEL_H, MODEL_W), dtype=np.float32)
        truth_norm = np.empty((len(starts), AFT, 3, MODEL_H, MODEL_W), dtype=np.float32)
        for index, s in enumerate(starts):
            i0 = s - TEST_START
            inputs[index, :, :3] = model_frames[i0: i0 + PRE]
            inputs[index, :, 3:] = condition[None, :, None, None]
            truth_norm[index] = model_frames[i0 + PRE: i0 + PRE + AFT]

        copy_norm = np.repeat(inputs[:, -1:, :3], AFT, axis=1)
        # zero-fluctuation: replace every azimuthal row by its own mean, so
        # every n >= 1 coefficient vanishes while the radial profile stays.
        zero_norm = np.repeat(
            truth_norm[..., :VALID_W].mean(axis=-1, keepdims=True), MODEL_W, axis=-1
        ).astype(np.float32)

        copy_q = q_observables(loss_module, copy_norm, truth_norm,
                               mode_n0, drift, device)
        zero_q = q_observables(loss_module, zero_norm, truth_norm,
                               mode_n0, drift, device)
        row = [key, case["role"][:6], copy_q["complex_mode_loss"],
               zero_q["complex_mode_loss"] / copy_q["complex_mode_loss"]]
        cells = {}
        for name in CELL_ORDER:
            pred = predict_direct10(models[name], inputs, device)
            q = q_observables(loss_module, pred, truth_norm, mode_n0, drift, device)
            cells[name] = q
            row.append((q["complex_mode_loss"] / copy_q["complex_mode_loss"],
                        q["amplitude_log_bias"],
                        float(np.exp(q["amplitude_log_bias"]))))
        rows.append(row)
        print("%-11s %-7s | %-10.4g %-10.4f | " % (row[0], row[1], row[2], row[3])
              + " ".join("%-22s" % ("%.4f  %+6.2f  %5.3f" % (c[0], c[1], c[2]))
                         for c in row[4:]))
        del model_frames, inputs, truth_norm

    print()
    print("cml = complex_mode_loss.  zero_flu is the loss of a prediction with no")
    print("azimuthal fluctuation at all, expressed as a ratio to copy.  ampbias =")
    print("mean log(|pred_q|/|true_q|); pwr = exp(ampbias) = retained amplitude fraction.")
    print("A cell whose cml/copy is close to zero_flu and whose pwr << 1 has collapsed")
    print("the fluctuation rather than predicted it.")


if __name__ == "__main__":
    main()
