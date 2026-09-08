"""Evaluate both terminal D/P checkpoints on matched development windows."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    build_model, condition_vector, load_case_frames, predict_direct10, denormalize, VALID_H, PRE, AFT,
)
from radaz_metrics_v3 import (
    VERSION, band_pool, local_observables, compare_observables, field_diagnostics, summarize, skill,
)
from run_radaz_paired_pilot import ROOT, OUT, digest, verify_bundle, checkpoint_path, checkpoint_epoch

BASELINES = ROOT / "workdirs/2D_RadAz/radaz_nextstep_baselines_v1"
SPLITS = {"source_validation": (1600, 1800), "source_test": (1800, 2000)}


def matched_baseline(observable, split, case_key, truth):
    path = BASELINES / f"{split}_{case_key}_{observable}_predictions.npz"
    with np.load(path) as archive:
        np.testing.assert_allclose(truth, archive["truth"], rtol=1e-4,
                                   atol=float(np.max(np.abs(truth))) * 1e-8)
        return archive["ar10_validation_selected"].copy()


def evaluate():
    bundle = verify_bundle()
    output = OUT / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise RuntimeError("Completed paired evaluation exists; refusing overwrite")
    manifest = json.loads(Path(bundle["manifest"]).read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if len(cases) != 6 or any(c["role"] != "source" for c in cases):
        raise RuntimeError("Expected six source conditions only")
    low, high = (np.asarray(manifest["normalization"][k], dtype=float) for k in ("low", "high"))
    result = {"status": "exploratory_source_development_pilot_seed42", "metric_version": VERSION,
              "bundle_sha256": digest(OUT / "bundle.json"), "checkpoint_sha256": {},
              "checkpoint_epoch_indices": {}, "input_hashes": {}, "evaluations": {},
              "paired_comparison": {}, "baseline_results_sha256": digest(BASELINES / "results.json"),
              "limitations": ["One training seed and one PIC realization per condition",
                              "Source-test has already been used for development",
                              "AR is condition-specific and fits late-train 400 frames",
                              "No architecture gate, checkpoint selection or significance claim"]}
    torch.set_num_threads(4)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {}
    for cell in ("D", "P"):
        path = checkpoint_path(cell)
        if checkpoint_epoch(path) != 59:
            raise RuntimeError("Both terminal checkpoints are required")
        result["checkpoint_sha256"][cell] = digest(path)
        models[cell], epoch = build_model(ROOT / bundle["configs"][cell], path, torch.device("cpu"))
        result["checkpoint_epoch_indices"][cell] = epoch
    for split, (start, stop) in SPLITS.items():
        rows = {v: {} for v in ("copy", "D", "P")}
        paired = {}
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
            result["input_hashes"][split + "/" + key] = hashlib.sha256(x.tobytes() + target.tobytes()).hexdigest()
            persistence = np.repeat(x[:, -1:, :3], AFT, axis=1)
            pool = band_pool(x_m[:VALID_H])
            dy, magnetic_t = float(np.median(np.diff(y_m))), float(case["B_mT"]) * .001
            ts = local_observables(denormalize(target, low, high), pool, dy, magnetic_t)
            cs = local_observables(denormalize(persistence, low, high), pool, dy, magnetic_t)
            ar = {k: matched_baseline(k, split, key, ts[k]) for k in ("gamma", "flux_full")}
            physical_predictions = {}
            for cell in rows:
                if cell == "copy":
                    pred, ps = persistence, cs
                else:
                    model = models[cell].to(dev)
                    pred = predict_direct10(model, x, dev)
                    model.cpu()
                    ps = local_observables(denormalize(pred, low, high), pool, dy, magnetic_t)
                row = compare_observables(ps, ts, cs, magnetic_t, pool)
                row.update(field_diagnostics(pred[..., :VALID_H, :], target[..., :VALID_H, :],
                                             persistence[..., :VALID_H, :], low, high))
                row["n0"] = n0
                row["AR_comparison"] = {k: {"skill_vs_AR": skill(ps[k], ts[k], ar[k]),
                    "lead_skill_vs_AR": [skill(ps[k][:, j], ts[k][:, j], ar[k][:, j]) for j in range(AFT)]}
                    for k in ar}
                rows[cell][key] = row
                physical_predictions[cell] = {k: ps[k].copy() for k in ar}
                spectra = {prefix + k: obs[k] for prefix, obs in (("pred_", ps), ("truth_", ts))
                           for k in ("pn", "pe", "cross", "gamma", "flux_full", "flux_omitted")}
                np.savez_compressed(output / f"{split}_{key}_{cell}.npz", **spectra)
                print(split, key, cell, "Gamma skill", row["gamma_skill_vs_copy"],
                      "full-flux skill", row["flux_full_skill_vs_copy"], flush=True)
                del spectra
                if cell != "copy":
                    del ps, pred
            paired[key] = {k: {"P_skill_vs_D": skill(physical_predictions["P"][k], ts[k], physical_predictions["D"][k]),
                "P_lead_skill_vs_D": [skill(physical_predictions["P"][k][:, j], ts[k][:, j],
                                            physical_predictions["D"][k][:, j]) for j in range(AFT)]} for k in ar}
            del x, target, persistence, ts, cs, physical_predictions
        result["evaluations"][split] = {v: {"per_condition": data, "summary": summarize(data)} for v, data in rows.items()}
        result["paired_comparison"][split] = paired
        (output / "partial_results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    verify_bundle()
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("Written", output / "results.json", flush=True)


if __name__ == "__main__":
    evaluate()
