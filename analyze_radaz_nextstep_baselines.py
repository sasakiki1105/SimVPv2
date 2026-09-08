"""Source-only residual/time-scale audit and train/validation-fitted baselines.

Exploratory development data; never an independent test of the old model.
AR is a condition-specific scalar-per-observable model, not zero-shot transfer.
No target from the test interval is used for fitting or hyperparameter selection.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np

from evaluate_radaz_conditioned_factorial import MANIFEST, load_case_frames, denormalize, VALID_H
from evaluate_radaz_corrected_A import digest
from radaz_metrics_v3 import band_pool, local_observables, nrmse, skill

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "workdirs/2D_RadAz/radaz_nextstep_baselines_v1"
OLD = ROOT / "workdirs/2D_RadAz/radaz_corrected_A_v3"
TRAIN = (1200, 1600)
SPLITS = {"source_validation": (1600, 1800), "source_test": (1800, 2000)}
RIDGES = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
PRE, AFT = 10, 10


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def windows(series, start=0, stop=None, stride=1):
    stop = len(series) if stop is None else stop
    if start < 0 or stop > len(series) or stop-start < PRE+AFT:
        raise ValueError("Invalid disjoint window interval")
    starts = np.arange(start, stop-PRE-AFT+1, stride)
    x = np.stack([series[s:s+PRE] for s in starts])
    y = np.stack([series[s+PRE:s+PRE+AFT] for s in starts])
    return x, y


def fit_ar(series, ridge):
    """One scalar AR(10) per component, direct ten leads, ridge on slopes.

    Centers/scales/intercepts fit on TRAIN only. All ridge values are chosen
    before evaluation; one value per observable is selected across six source
    validation conditions, then held fixed for source-test.
    """
    shape = series.shape[1:]
    a = series.reshape(len(series), -1)
    mean, std = a.mean(axis=0), a.std(axis=0)
    floor = max(float(std.max())*1e-12, np.finfo(float).tiny)
    std = np.maximum(std, floor)
    x, y = windows((a-mean)/std)
    x, y = x.transpose(2, 0, 1), y.transpose(2, 0, 1)
    x = np.concatenate([np.ones((*x.shape[:2], 1)), x], axis=2)
    xtx = np.einsum("fsi,fsj->fij", x, x)/x.shape[1]
    xty = np.einsum("fsi,fsj->fij", x, y)/x.shape[1]
    penalty = np.eye(PRE+1)*ridge
    penalty[0, 0] = 0
    beta = np.linalg.solve(xtx+penalty, xty)
    return {"mean": mean, "std": std, "beta": beta, "shape": shape}


def predict_ar(model, histories):
    x = histories.reshape(len(histories), PRE, -1)
    x = ((x-model["mean"])/model["std"]).transpose(2, 0, 1)
    x = np.concatenate([np.ones((*x.shape[:2], 1)), x], axis=2)
    y = np.einsum("fsi,fih->shf", x, model["beta"])
    y = y*model["std"]+model["mean"]
    return y.reshape(len(histories), AFT, *model["shape"])


def summary_error(pred, truth, copy):
    e = pred-truth
    axes = (0, 1)
    sse = float(np.sum(e**2))
    bias_sse = float(len(pred)*pred.shape[1]*np.sum(e.mean(axis=axes)**2))
    pc, tc = pred-pred.mean(axis=axes), truth-truth.mean(axis=axes)
    norm = float(np.linalg.norm(pc)*np.linalg.norm(tc))
    return {"nrmse": nrmse(pred, truth), "skill_vs_copy": skill(pred, truth, copy),
            "rmse_SI": float(np.sqrt(np.mean(e**2))),
            "mean_bias_fraction_of_sse": bias_sse/sse if sse else None,
            "centered_correlation": float(np.sum(pc*tc)/norm) if norm else None,
            "lead_skill": [skill(pred[:, j], truth[:, j], copy[:, j]) for j in range(AFT)],
            "lead_rmse_SI": [float(np.sqrt(np.mean(e[:, j]**2))) for j in range(AFT)]}


def timescale(series):
    """Descriptive late-TRAIN ACF; not an independent-sample count."""
    centered = series-series.mean(axis=0)
    total_energy, variance = float(np.mean(series**2)), float(np.mean(centered**2))
    acf = []
    for lag in range(1, 201):
        a, b = centered[:-lag], centered[lag:]
        den = float(np.linalg.norm(a)*np.linalg.norm(b))
        acf.append(float(np.sum(a*b)/den) if den else None)
    crossing = next((j+1 for j, r in enumerate(acf) if r is not None and r <= 1/np.e), None)
    return {"acf_lags_1_to_200": acf, "first_1_over_e_crossing_ns": 15*crossing if crossing else None,
            "fluctuation_rms_over_total_rms": float(np.sqrt(variance/total_energy)) if total_energy else None,
            "persistence_nrmse_by_lead_1_to_40": [nrmse(series[:-h], series[h:]) for h in range(1, 41)],
            "train_mean_forecast_reference_only": "ACF of 1200..1599; oscillatory crossing is not a universal predictability limit"}


def residual_geometry(archive, magnetic_t):
    p, t = archive["pred_gamma"], archive["truth_gamma"]
    error = p-t
    sse = float(np.sum(error**2))
    bin_sse = np.sum(error**2, axis=(0, 1))
    top = np.argsort(bin_sse.ravel())[::-1][:10]
    at = np.sqrt(archive["truth_pn"]*archive["truth_pe"])
    ap = np.sqrt(archive["pred_pn"]*archive["pred_pe"])
    ot = np.divide(archive["truth_cross"].real, at, out=np.zeros_like(at), where=at>0)
    op = np.divide(archive["pred_cross"].real, ap, out=np.zeros_like(ap), where=ap>0)
    terms = [-2/magnetic_t*(ap-at)*ot, -2/magnetic_t*at*(op-ot), -2/magnetic_t*(ap-at)*(op-ot)]
    np.testing.assert_allclose(sum(terms), error, rtol=1e-6, atol=max(float(np.max(abs(error)))*1e-10, 1e-30))
    gram = [[float(np.sum(a*b)/sse) if sse else None for b in terms] for a in terms]
    return {"radial_band_sse_shares": (bin_sse.sum(axis=1)/sse).tolist(),
            "mode_sse_shares_n1_to_64": (bin_sse.sum(axis=0)/sse).tolist(),
            "top_bins": [{"radial_band_1_based": int(k//64+1), "n": int(k%64+1),
                          "sse_share": float(bin_sse.ravel()[k]/sse)} for k in top],
            "amplitude_organization_interaction_gram_over_total_sse": gram,
            "decomposition": "e=e_amplitude+e_organization+e_interaction; sum of all 9 Gram entries is 1; cross terms may cancel"}


def cache_case(case, manifest):
    if case["role"] != "source":
        raise ValueError("Source fields only")
    key = case["case_key"]
    path = OUT / (key+"_spectra.npz")
    meta_path = OUT / (key+"_data.json")
    stat = Path(case["path"]).stat()
    fingerprint = {"path": case["path"], "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                   "manifest_sha256": digest(MANIFEST), "frames": [1200, 1999]}
    if path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["fingerprint"] != fingerprint:
            raise RuntimeError("Cache provenance changed")
        return dict(np.load(path)), meta
    low, high = (np.asarray(manifest["normalization"][k], dtype=float) for k in ("low", "high"))
    pieces = {k: [] for k in ("pn", "pe", "cross", "gamma", "flux_full")}
    used = hashlib.sha256()
    for start in range(1200, 2000, 32):
        frames, _, x_m, y_m = load_case_frames(case, low, high, start, min(start+32, 2000))
        used.update(frames.tobytes())
        physical = denormalize(frames[None], low, high)
        obs = local_observables(physical, band_pool(x_m[:VALID_H]), float(np.median(np.diff(y_m))), case["B_mT"]*.001)
        for k in pieces:
            pieces[k].append(obs[k][0])
        del frames, physical, obs
    result = {k: np.concatenate(v) for k, v in pieces.items()}
    meta = {"fingerprint": fingerprint, "normalized_used_frames_sha256": used.hexdigest()}
    np.savez_compressed(path, **result)
    write_json(meta_path, meta)
    print("Cached source frames:", key, flush=True)
    return result, meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise SystemExit("Completed baseline result exists; refusing overwrite")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = [c for c in manifest["cases"] if c["role"] == "source"]
    code = [Path(__file__), ROOT / "radaz_metrics_v3.py", ROOT / "evaluate_radaz_conditioned_factorial.py"]
    protocol = {"status": "exploratory_on_already_inspected_source_data", "train_frames": TRAIN,
                "split_frames": SPLITS, "pre": PRE, "aft": AFT, "stride_eval": 20,
                "ridge_grid": RIDGES, "selection": "one lambda per target, minimize equal-condition validation NRMSE squared",
                "ar": "condition-specific, separate scalar component, 10 lags -> 10 direct leads; train-only scale and intercept",
                "source_conditions": [c["case_key"] for c in cases],
                "not_claimed": "zero-shot transfer, independent PIC replication, causal identification",
                "hashes": {str(p.relative_to(ROOT)): digest(p) for p in code}, "manifest_sha256": digest(MANIFEST)}
    frozen = OUT / "protocol.json"
    encoded = json.dumps(protocol, indent=2, allow_nan=False)
    if frozen.exists() and frozen.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("Protocol changed; version the experiment explicitly")
    frozen.write_text(encoded, encoding="utf-8")
    cache = {}
    result = {"protocol_sha256": digest(frozen), "cases": {}, "selected_ridge": {}}
    fitted = {}
    for case in cases:
        key = case["case_key"]
        values, meta = cache_case(case, manifest)
        cache[key] = values
        result["cases"][key] = {"data": meta, "timescale_late_train": {}, "residuals": {}, "comparisons": {}}
        for observable in ("gamma", "flux_full"):
            training = values[observable][:400]
            result["cases"][key]["timescale_late_train"][observable] = timescale(training)
            for alpha in RIDGES:
                fitted[key, observable, alpha] = fit_ar(training, alpha)
        for split, (lo, hi) in SPLITS.items():
            with np.load(OLD / (split+"_"+key+"_last_input.npz")) as a:
                _, truth = windows(values["gamma"], lo-1200, hi-1200, 20)
                np.testing.assert_allclose(truth, a["truth_gamma"], rtol=1e-4, atol=float(np.max(abs(truth)))*1e-8)
                result["cases"][key]["residuals"][split] = residual_geometry(a, case["B_mT"]*.001)
    for observable in ("gamma", "flux_full"):
        scores = {}
        for alpha in RIDGES:
            per_case = []
            for case in cases:
                key = case["case_key"]
                x, y = windows(cache[key][observable], 400, 600, 20)
                pred = predict_ar(fitted[key, observable, alpha], x)
                per_case.append(nrmse(pred, y)**2)
            scores[str(alpha)] = float(np.mean(per_case))
        chosen = min(RIDGES, key=lambda a: scores[str(a)])
        result["selected_ridge"][observable] = {"alpha": chosen, "validation_scores": scores}
        for case in cases:
            key = case["case_key"]
            model = fitted[key, observable, chosen]
            np.savez_compressed(OUT / (key+"_"+observable+"_ar.npz"), **model)
            for split, (lo, hi) in SPLITS.items():
                x, y = windows(cache[key][observable], lo-1200, hi-1200, 20)
                copy = np.repeat(x[:, -1:], AFT, axis=1)
                t = np.arange(PRE)-(PRE-1)/2
                slope = np.einsum("s t ...,t->s ...", x, t)/np.sum(t*t)
                lead = np.arange(PRE, PRE+AFT)-(PRE-1)/2
                predictions = {
                    "copy": copy,
                    "input_mean": np.repeat(x.mean(axis=1, keepdims=True), AFT, axis=1),
                    "late_train_mean": np.broadcast_to(cache[key][observable][:400].mean(axis=0), y.shape),
                    "input_linear_trend": x.mean(axis=1, keepdims=True)+slope[:, None]*lead.reshape(1, AFT, *([1]*(x.ndim-2))),
                    "ar10_validation_selected": predict_ar(model, x),
                }
                with np.load(OLD / (split+"_"+key+"_last_input.npz")) as a:
                    predictions["A_last_input"] = a["pred_"+observable]
                rows = {label: summary_error(p, y, copy) for label, p in predictions.items()}
                result["cases"][key]["comparisons"].setdefault(split, {})[observable] = rows
                np.savez_compressed(OUT / (split+"_"+key+"_"+observable+"_predictions.npz"), truth=y, **predictions)
                print(split, key, observable, {k: round(v["skill_vs_copy"], 4) for k,v in rows.items()}, flush=True)
    write_json(OUT / "results.json", result)
    print("Completed", OUT / "results.json", flush=True)


if __name__ == "__main__":
    main()
