"""Audit 5: is the positive factorial interaction a gradient conflict?

C-P does not collect the sum of the conditioning gain and the physics-loss
gain; the interaction is positive in almost every metric.  One explanation is
that the data loss, the q-complex-mode loss and the modal-transport loss pull
the shared representation in different directions.  That is a measurable
claim, and it should be measured before reaching for PCGrad/CAGrad -- if the
cosines are positive, the problem is the representation, not the optimiser.

Computes, on the same pre-registered windows and for each trained cell,

    g_D = grad L_data ,  g_Q = grad L_complex_mode ,  g_G = grad L_transport

and their pairwise cosine similarities, per parameter group (encoder, gSTA
translator, FiLM conditioner, decoder).  Gradients are taken at the trained
weights, so this is the conflict the optimiser saw at convergence.
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
    build_model, condition_vector, load_case_frames,
)
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

CELL_ORDER = ["U-D", "C-D", "U-P", "C-P"]
GROUPS = {
    "encoder": lambda n: n.startswith("enc."),
    "gSTA": lambda n: n.startswith("hid.") and ".film." not in n,
    "FiLM": lambda n: ".film." in n,
    "decoder": lambda n: n.startswith("dec."),
}


def group_of(name):
    for group, test in GROUPS.items():
        if test(name):
            return group
    return None


def pair_cosines(model, grads_a, grads_b):
    """Cosine per parameter group, accumulating scalars.

    Concatenating the full gradient (about 1.4e8 entries) into one vector per
    group needs GBs on an 8 GB card and was what made the first attempt
    unusable.  Dot products and norms are additive, so accumulate them.
    """
    stats = {}
    for (name, _), ga, gb in zip(model.named_parameters(), grads_a, grads_b):
        if ga is None or gb is None:
            continue
        keys = [k for k in (group_of(name), "all") if k is not None]
        dot = float(torch.sum(ga.detach() * gb.detach()))
        na = float(torch.sum(ga.detach() * ga.detach()))
        nb = float(torch.sum(gb.detach() * gb.detach()))
        for key in keys:
            acc = stats.setdefault(key, [0.0, 0.0, 0.0])
            acc[0] += dot
            acc[1] += na
            acc[2] += nb
    out = {}
    for key in list(GROUPS) + ["all"]:
        if key not in stats:
            out[key] = float("nan")
            continue
        dot, na, nb = stats[key]
        out[key] = dot / np.sqrt(na * nb) if na > 0 and nb > 0 else float("nan")
    return out


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
    starts = [s for s in range(TEST_START, TEST_STOP, PRE + AFT)
              if s + PRE + AFT <= TEST_STOP]

    # one window from each of the six source conditions
    batches = []
    max_cases = int(os.environ.get("GRAD_AUDIT_CASES", "2"))
    for case in manifest["cases"]:
        if case["role"] != "source" or len(batches) >= max_cases:
            continue
        condition, drift, mode_n0 = condition_vector(case, manifest)
        frames_needed = TEST_START + PRE + AFT
        model_frames, _, _, _ = load_case_frames(
            case, low, high, TEST_START, frames_needed)
        x = np.empty((1, PRE, 5, MODEL_H, MODEL_W), dtype=np.float32)
        x[0, :, :3] = model_frames[:PRE]
        x[0, :, 3:] = condition[None, :, None, None]
        y = model_frames[PRE: PRE + AFT][None]
        batches.append((case["case_key"], x, y.astype(np.float32)))
        del model_frames

    results = {}
    for name in CELL_ORDER:
        model = build_model(CELLS[name]["config"],
                            CELLS[name]["workdir"] / "checkpoints" / "best.ckpt",
                            device)[0]
        model.train(False)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        parameters = [p for p in model.parameters()]
        per_case = {}
        for key, x, y in batches:
            bx = torch.from_numpy(x).to(device)
            by = torch.from_numpy(y).to(device)
            prediction = model(bx)
            data_loss = torch.mean((prediction - by) ** 2)
            spectral = loss_module(prediction, by, bx)
            losses = {"data": data_loss,
                      "qmode": spectral["complex_mode"],
                      "transport": spectral["transport"]}
            raw = {}
            for label, value in losses.items():
                raw[label] = torch.autograd.grad(
                    value, parameters, retain_graph=True, allow_unused=True)
            pairs = {
                "D_Q": pair_cosines(model, raw["data"], raw["qmode"]),
                "D_G": pair_cosines(model, raw["data"], raw["transport"]),
                "Q_G": pair_cosines(model, raw["qmode"], raw["transport"]),
            }
            row = {}
            for group in list(GROUPS) + ["all"]:
                row[group] = {k: pairs[k][group] for k in pairs}
            row["_loss"] = {k: float(v) for k, v in losses.items()}
            del raw
            per_case[key] = row
            del prediction
            torch.cuda.empty_cache()
            print("   %s / %s  cos(D,G)_all = %+.4f"
                  % (name, key, row["all"]["D_G"]), flush=True)
        results[name] = per_case
        del model
        torch.cuda.empty_cache()
        print("[done] " + name, flush=True)

    pairs = [("D_Q", "data vs q-mode"), ("D_G", "data vs transport"),
             ("Q_G", "q-mode vs transport")]
    for pair, title in pairs:
        print()
        print("=" * 100)
        print("GRADIENT COSINE  %s   (negative = conflicting objectives)" % title)
        print("=" * 100)
        print("%-9s " % "cell" + " ".join(
            "%12s" % g for g in list(GROUPS) + ["all"]))
        print("-" * 100)
        for name in CELL_ORDER:
            values = []
            for group in list(GROUPS) + ["all"]:
                v = [results[name][k][group][pair] for k in results[name]]
                v = [x for x in v if np.isfinite(x)]
                values.append(np.median(v) if v else float("nan"))
            print("%-9s " % name + " ".join("%12.4f" % v for v in values))
        print("(median over the six source conditions)")

    print()
    print("=" * 100)
    print("PER-CASE cosine, data vs transport, 'all' parameters")
    print("=" * 100)
    keys = list(results[CELL_ORDER[0]])
    print("%-9s " % "cell" + " ".join("%12s" % k for k in keys))
    for name in CELL_ORDER:
        print("%-9s " % name + " ".join(
            "%12.4f" % results[name][k]["all"]["D_G"] for k in keys))

    out = (MANIFEST.parent.parent / "radaz_conditioned_factorial_evaluation"
           / "gradient_conflict_audit.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print("[written] " + str(out))


if __name__ == "__main__":
    main()
