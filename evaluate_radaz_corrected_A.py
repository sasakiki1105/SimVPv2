"""Exploratory reanalysis of existing A; never authorizes architecture runs.

Source-only matched validation/test windows, fixed best AND last checkpoints,
three explicitly named inference policies and persistence. No checkpoint or
policy is selected by these already-inspected test data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import h5py
import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    MANIFEST, build_model, condition_vector, load_case_frames, predict_direct10,
    denormalize, VALID_H, PRE, AFT,
)
from openstl.models.simvp_factory import configure_translator_norm, calibrate_batch_norm
from radaz_metrics_v3 import (
    VERSION, band_pool, local_observables, compare_observables, field_diagnostics,
    summarize, exact_spearman,
)

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs/custom/pepapic/SimVP_gSTA_radaz_redA_60ep.py"
WORK = ROOT / "workdirs/2D_RadAz/radaz_arch_redA_az64_hidT256_NT4_60ep"
DEFAULT_OUTPUT = ROOT / "workdirs/2D_RadAz/radaz_corrected_A_v3"
CALIBRATION_STARTS = (1200, 1300, 1400, 1500)
SPLITS = {"source_validation": (1600, 1800), "source_test": (1800, 2000)}
POLICIES = ("running", "input", "source_train_calibrated")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_source_window(case, manifest, start, stop, split):
    if case["role"] != "source":
        raise ValueError("Corrected A diagnostic may open SOURCE cases only")
    end = int(manifest["normalization"]["frame_counts"][case["case_key"]]["train_end_exclusive"])
    lo, hi = (0, end) if split == "calibration" else SPLITS[split]
    if end != SPLITS["source_validation"][0] or not (lo <= start < stop <= hi):
        raise ValueError("Window violates the frozen frame-disjoint protocol")


def read_inputs(case, manifest, low, high, start, stop, split):
    validate_source_window(case, manifest, start, stop, split)
    frames, _, x_m, y_m = load_case_frames(case, low, high, start, stop)
    cond, _, n0 = condition_vector(case, manifest)
    return frames, cond, n0, x_m, y_m


def calibration_inputs(cases, manifest, low, high, device):
    for case in cases:
        for start in CALIBRATION_STARTS:
            frames, cond, *_ = read_inputs(case, manifest, low, high, start, start + PRE, "calibration")
            x = np.empty((1, PRE, 5, *frames.shape[-2:]), dtype=np.float32)
            x[0, :, :3] = frames
            x[0, :, 3:] = cond[None, :, None, None]
            yield torch.from_numpy(x).to(device)


def freeze_protocol(output, manifest, cases, splits, checkpoints, policies):
    code = [Path(__file__), ROOT / "radaz_metrics_v3.py", ROOT / "evaluate_radaz_conditioned_factorial.py",
            ROOT / "openstl/models/simvp_factory.py", ROOT / "openstl/models/simvp_model.py",
            ROOT / "openstl/modules/simvp_modules.py", ROOT / "openstl/methods/pepapic_spectral_loss.py"]
    assets = {str(p.relative_to(ROOT)): digest(p) for p in code + [CONFIG, MANIFEST]}
    checkpoint_hashes = {k: digest(WORK / "checkpoints" / (k + ".ckpt")) for k in checkpoints}
    data = {}
    for case in cases:
        # This metadata read also enforces source-only membership.
        validate_source_window(case, manifest, 1200, 1210, "calibration")
        path = Path(case["path"])
        stat = path.stat()
        with h5py.File(path, "r") as f:
            data[case["case_key"]] = {"path": str(path), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "field_shapes": {k: list(f["fields/" + k].shape) for k in case["channels"]},
                "axes_sha256": hashlib.sha256(np.asarray(f["axes/x_m"]).tobytes() +
                                              np.asarray(f["axes/y_m"]).tobytes()).hexdigest()}
    protocol = {"metric_version": VERSION, "status": "exploratory_reanalysis_of_inspected_data",
        "primary_claim": "No confirmatory gate; compare all fixed variants with persistence",
        "architecture_continuation_authorized": False,
        "split_windows": {s: {"start": SPLITS[s][0], "stop_exclusive": SPLITS[s][1],
                               "stride": PRE + AFT, "input": PRE, "output": AFT} for s in splits},
        "checkpoint_names": checkpoints, "checkpoint_sha256": checkpoint_hashes,
        "inference_policies": policies, "calibration_source_train_starts": CALIBRATION_STARTS,
        "calibration_description": "equal-window mean of per-input BN moments; upstream input normalization; 24 windows; no targets",
        "metrics": {"radial_reduction": "local_product", "radial_bands": 4,
                    "radial_interval_m": [0.0009, 0.0119], "retained_modes": [1, 64],
                    "Ey": "periodic central difference", "one_sided_flux_factor": 2,
                    "Nyquist_flux_factor": 1, "O_floor": 0.05, "Gamma_squared_weight_floor": 1e-3,
                    "O_ratio_aggregation": "inverse-CDF true-Gamma-squared weighted median",
                    "temporal_metrics": "individual output frames before any averaging",
                    "p_value": "all condition-label permutations, descriptive only"},
        "selection": "none; no selection using source_test or old holdouts",
        "replication": "six fixed source conditions, one PIC realization per condition, training seed 42",
        "data_fingerprints": data,
        "data_hash_scope": "metadata and axes frozen here; exact normalized evaluation arrays hashed in result (not a full H5 hash)",
        "code_config_manifest_sha256": assets,
        "software": {"torch": str(torch.__version__), "numpy": np.__version__}}
    text = json.dumps(protocol, indent=2, sort_keys=True)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError("Frozen protocol/code/assets changed; use a new explicitly versioned output directory")
    if (output / "results.json").exists():
        raise RuntimeError("Completed results already exist; refusing to overwrite")
    path.write_text(text, encoding="utf-8")
    return digest(path)


def evaluate(args):
    torch.set_num_threads(4)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = [c for c in manifest["cases"] if c["role"] == "source"]
    if len(cases) != 6:
        raise ValueError("Expected the six registered source conditions")
    low, high = (np.asarray(manifest["normalization"][k], dtype=float) for k in ("low", "high"))
    protocol_sha = freeze_protocol(args.output, manifest, cases, args.splits, args.checkpoints, args.policies)
    result = {"version": VERSION, "protocol_sha256": protocol_sha, "status": "exploratory",
              "architecture_continuation_authorized": False, "evaluations": {}, "input_hashes": {}}
    models = {}
    epochs = {}
    # Keep models on CPU between predictions to fit the 8 GB GPU.
    for ckpt in args.checkpoints:
        original, epochs[ckpt] = build_model(CONFIG, WORK / "checkpoints" / (ckpt + ".ckpt"), dev)
        for policy in args.policies:
            import copy
            m = copy.deepcopy(original)
            if policy == "input":
                configure_translator_norm(m, "batch_input")
            elif policy == "source_train_calibrated":
                buffers = calibrate_batch_norm(original, calibration_inputs(cases, manifest, low, high, dev))
                state = m.state_dict()
                state.update(buffers)
                m.load_state_dict(state, strict=True)
                torch.save(buffers, args.output / (ckpt + "_source_train_bn_buffers.pt"))
            models[ckpt + "/" + policy] = m.cpu().eval()
        del original
    result["checkpoint_epoch_indices"] = epochs
    for split in args.splits:
        rows = {variant: {} for variant in ["copy"] + list(models)}
        for case in cases:
            key = case["case_key"]
            start, stop = SPLITS[split]
            frames, cond, n0, x_m, y_m = read_inputs(case, manifest, low, high, start, stop, split)
            x = np.empty((10, PRE, 5, *frames.shape[-2:]), dtype=np.float32)
            target = np.empty((10, AFT, 3, *frames.shape[-2:]), dtype=np.float32)
            for j in range(10):
                x[j, :, :3] = frames[j * 20:j * 20 + PRE]
                x[j, :, 3:] = cond[None, :, None, None]
                target[j] = frames[j * 20 + PRE:(j + 1) * 20]
            result["input_hashes"][split + "/" + key] = hashlib.sha256(x.tobytes() + target.tobytes()).hexdigest()
            del frames
            persistence = np.repeat(x[:, -1:, :3], AFT, axis=1)
            pool = band_pool(x_m[:VALID_H])
            dy, magnetic_t = float(np.median(np.diff(y_m))), float(case["B_mT"]) * 0.001
            ts = local_observables(denormalize(target, low, high), pool, dy, magnetic_t)
            cs = local_observables(denormalize(persistence, low, high), pool, dy, magnetic_t)
            for variant in rows:
                if variant == "copy":
                    pred, ps = persistence, cs
                else:
                    model = models[variant].to(dev)
                    pred = predict_direct10(model, x, dev)
                    model.cpu()
                    ps = local_observables(denormalize(pred, low, high), pool, dy, magnetic_t)
                row = compare_observables(ps, ts, cs, magnetic_t, pool)
                row.update(field_diagnostics(pred[..., :VALID_H, :], target[..., :VALID_H, :],
                                             persistence[..., :VALID_H, :], low, high))
                row["n0"] = n0
                rows[variant][key] = row
                spectra = {prefix + k: d[k] for prefix, d in (("pred_", ps), ("truth_", ts))
                           for k in ("pn", "pe", "cross", "gamma", "flux_full", "flux_omitted")}
                np.savez_compressed(args.output / (split + "_" + key + "_" + variant.replace("/", "_") + ".npz"), **spectra)
                print(f"{split} {key} {variant}: Gamma(time)={row['gamma_time_nrmse']:.6g}, skill={row['gamma_skill_vs_copy']:.6g}", flush=True)
                if variant != "copy":
                    del pred, ps
            del x, target, persistence, ts, cs
        result["evaluations"][split] = {}
        for variant, conditions in rows.items():
            result["evaluations"][split][variant] = {"per_condition": conditions,
                "summary": summarize(conditions),
                "n0_O_diagnostic": exact_spearman([r["n0"] for r in conditions.values()],
                    [r["O_ratio_weighted_median"] for r in conditions.values()])}
        (args.output / "partial_results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("Written", args.output / "results.json", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--checkpoints", nargs="+", choices=["best", "last"], default=["best", "last"])
    parser.add_argument("--policies", nargs="+", choices=list(POLICIES), default=list(POLICIES))
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
