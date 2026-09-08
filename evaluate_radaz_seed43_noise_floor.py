"""Noise floor for the frozen D/P primary endpoint (seed-42 vs seed-43 D cells).

Computes, on the same matched development windows as the pilot evaluation, the
per-condition skill(D_seed43, truth, D_seed42) for the gamma and flux_full
observables, and the pre-registered noise scale
    N = median over the six source_test conditions of |gamma skill(D43 vs D42)|
(RADAZ_PAIRED_PILOT_PRIMARY_ENDPOINT_ADDENDUM.md, sha256 8d654ed1...). The two
cells share config, data order, and everything except the training seed, so this
measures run-to-run variation, not a loss intervention.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    build_model, condition_vector, load_case_frames, predict_direct10, denormalize, VALID_H, PRE, AFT,
)
from radaz_metrics_v3 import band_pool, local_observables, skill
from run_radaz_paired_pilot import ROOT, OUT as PILOT_OUT, digest, verify_bundle, checkpoint_path, checkpoint_epoch

SEED43_OUT = ROOT / "workdirs/2D_RadAz/radaz_seed43_replicate"
SEED43_CKPT = ROOT / "workdirs/2D_RadAz/radaz_pilot_D_GN_A_seed43_60ep/checkpoints/last.ckpt"
SPLITS = {"source_validation": (1600, 1800), "source_test": (1800, 2000)}


def main() -> None:
    result_path = SEED43_OUT / "noise_floor.json"
    if result_path.exists():
        raise RuntimeError("Completed noise-floor result exists; refusing overwrite")
    bundle = verify_bundle()
    manifest = json.loads(Path(bundle["manifest"]).read_text(encoding="utf-8"))
    cases = manifest["cases"]
    low, high = (np.asarray(manifest["normalization"][k], dtype=float) for k in ("low", "high"))
    checkpoints = {"D42": checkpoint_path("D"), "D43": SEED43_CKPT}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "definition": "skill(D43, truth, D42) = 1 - SSE(D43)/SSE(D42); "
                            "N = median |gamma skill| over source_test conditions",
              "checkpoint_sha256": {}, "splits": {}}
    torch.set_num_threads(4)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {}
    for name, path in checkpoints.items():
        if checkpoint_epoch(path) != 59:
            raise RuntimeError("Terminal checkpoint (index 59) required: " + str(path))
        result["checkpoint_sha256"][name] = digest(path)
        models[name], _ = build_model(ROOT / bundle["configs"]["D"], path, torch.device("cpu"))
    for split, (start, stop) in SPLITS.items():
        rows = {}
        for case in cases:
            key = case["case_key"]
            frames, _, x_m, y_m = load_case_frames(case, low, high, start, stop)
            cond, _, n0 = condition_vector(case, manifest)
            x = np.empty((10, PRE, 5, *frames.shape[-2:]), dtype=np.float32)
            target = np.empty((10, AFT, 3, *frames.shape[-2:]), dtype=np.float32)
            for j in range(10):
                x[j, :, :3] = frames[j * 20:j * 20 + PRE]
                x[j, :, 3:] = cond[None, :, None, None]
                target[j] = frames[j * 20 + PRE:(j + 1) * 20]
            del frames
            pool = band_pool(x_m[:VALID_H])
            dy, magnetic_t = float(np.median(np.diff(y_m))), float(case["B_mT"]) * .001
            ts = local_observables(denormalize(target, low, high), pool, dy, magnetic_t)
            observables = {}
            for name in models:
                model = models[name].to(dev)
                pred = predict_direct10(model, x, dev)
                model.cpu()
                observables[name] = local_observables(denormalize(pred, low, high), pool, dy, magnetic_t)
                del pred
            rows[key] = {
                observable: {
                    "skill_D43_vs_D42": skill(observables["D43"][observable], ts[observable],
                                              observables["D42"][observable]),
                    "lead_skill_D43_vs_D42": [skill(observables["D43"][observable][:, j], ts[observable][:, j],
                                                    observables["D42"][observable][:, j]) for j in range(AFT)],
                } for observable in ("gamma", "flux_full")
            }
            rows[key]["n0"] = n0
            print(split, key, "gamma skill(D43 vs D42)", rows[key]["gamma"]["skill_D43_vs_D42"], flush=True)
            del x, target, ts, observables
        gamma_values = [rows[k]["gamma"]["skill_D43_vs_D42"] for k in rows]
        result["splits"][split] = {
            "per_condition": rows,
            "gamma_skill_median": float(np.median(gamma_values)),
            "gamma_abs_skill_median": float(np.median(np.abs(gamma_values))),
            "flux_full_abs_skill_median": float(np.median(np.abs(
                [rows[k]["flux_full"]["skill_D43_vs_D42"] for k in rows]))),
        }
    result["noise_scale_N_gamma_source_test"] = result["splits"]["source_test"]["gamma_abs_skill_median"]
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("Written", result_path, "N =", result["noise_scale_N_gamma_source_test"], flush=True)


if __name__ == "__main__":
    main()
