#!/usr/bin/env python3
"""Train a carrier-envelope electric-history ROM for G8-hidden modes.

The recurrent state predicts only rotation-invariant amplitudes of n=17--21.
Their phase is supplied by an SO(2)-equivariant quadratic carrier formed from
observable low modes.  In a later full-field deployment, the existing G8
SimVP prediction can replace that analytic carrier while this ROM corrects its
hidden-band amplitudes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import train_radaz_electric_history_hidden_band_rom as base
from radaz_electric_history_hidden_band_rom import (
    FIELD_NAMES,
    HIDDEN_MODES,
    INPUT_FIELD_NAMES,
    apply_hidden_amplitude_to_carrier,
    invariant_observable_features,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "workdirs/radaz_electric_history_hidden_band_envelope_rom"
TINY = np.finfo(np.float64).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-steps", type=int, default=24)
    parser.add_argument("--observable-components", type=int, default=96)
    parser.add_argument("--hidden-components", type=int, default=64)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def training_indices(trajectories: dict[str, base.Trajectory]) -> dict[str, np.ndarray]:
    return {
        name: base.interval_indices(trajectory, "train")
        for name, trajectory in trajectories.items()
        if "train" in base.SPLITS[name]
    }


def rms_scale(arrays: list[np.ndarray]) -> np.ndarray:
    square_sum = sum(np.sum(np.abs(array.astype(np.complex128)) ** 2, axis=0) for array in arrays)
    count = sum(len(array) for array in arrays)
    scale = np.sqrt(square_sum / count)
    positive = scale[scale > 0]
    floor = float(np.median(positive)) * 1.0e-8
    return np.maximum(scale, max(floor, TINY))


def fit_transforms(
    trajectories: dict[str, base.Trajectory],
    observable_components: int,
    hidden_components: int,
) -> dict:
    indices = training_indices(trajectories)
    visible_train = [
        trajectories[name].coefficients[selected, : len(INPUT_FIELD_NAMES), :, :17]
        for name, selected in indices.items()
    ]
    hidden_train = [
        trajectories[name].coefficients[selected, :, :, 17:22]
        for name, selected in indices.items()
    ]
    observable_amplitude_scale = rms_scale(visible_train)
    hidden_amplitude_scale = rms_scale(hidden_train)
    observable_raw = np.concatenate(
        [
            invariant_observable_features(values, observable_amplitude_scale)
            for values in visible_train
        ]
    )
    hidden_raw = np.concatenate(
        [
            np.log1p(np.abs(values) / hidden_amplitude_scale[None, ...]).reshape(len(values), -1)
            for values in hidden_train
        ]
    )
    observable_scaler = StandardScaler().fit(observable_raw)
    hidden_scaler = StandardScaler().fit(hidden_raw)
    hidden_log_upper = np.quantile(hidden_raw, 0.999, axis=0).reshape(4, 8, 5) * 1.20
    observable_pca = PCA(
        n_components=min(observable_components, *observable_raw.shape),
        whiten=True,
        svd_solver="randomized",
        random_state=10,
    ).fit(observable_scaler.transform(observable_raw))
    hidden_pca = PCA(
        n_components=min(hidden_components, *hidden_raw.shape),
        whiten=True,
        svd_solver="randomized",
        random_state=11,
    ).fit(hidden_scaler.transform(hidden_raw))
    for trajectory in trajectories.values():
        visible = trajectory.coefficients[:, : len(INPUT_FIELD_NAMES), :, :17]
        observable = invariant_observable_features(visible, observable_amplitude_scale)
        hidden = np.log1p(
            np.abs(trajectory.coefficients[:, :, :, 17:22])
            / hidden_amplitude_scale[None, ...]
        ).reshape(len(trajectory.time_us), -1)
        trajectory.observable = observable_pca.transform(
            observable_scaler.transform(observable)
        ).astype(np.float32)
        trajectory.hidden = hidden_pca.transform(
            hidden_scaler.transform(hidden)
        ).astype(np.float32)
    return {
        "representation": "rotation_invariant_carrier_envelope",
        "observable_amplitude_scale": observable_amplitude_scale,
        "hidden_amplitude_scale": hidden_amplitude_scale,
        "hidden_log_upper": hidden_log_upper,
        "observable_scaler": observable_scaler,
        "observable_pca": observable_pca,
        "hidden_scaler": hidden_scaler,
        "hidden_pca": hidden_pca,
    }


def decode_amplitude(encoded: np.ndarray, transforms: dict) -> np.ndarray:
    standardized = transforms["hidden_pca"].inverse_transform(encoded)
    log_amplitude = transforms["hidden_scaler"].inverse_transform(standardized)
    log_amplitude = np.clip(
        log_amplitude.reshape(len(encoded), 4, 8, 5),
        0.0,
        transforms["hidden_log_upper"][None, ...],
    )
    return transforms["hidden_amplitude_scale"][None, ...] * np.expm1(log_amplitude)


def carrier_candidates(coefficients: np.ndarray, mode: int, radial: int):
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    candidates = []
    labels = []
    for first_mode in range(max(1, mode - 16), min(16, mode - 1) + 1):
        second_mode = mode - first_mode
        if not 1 <= second_mode <= 16:
            continue
        for first_field in range(len(INPUT_FIELD_NAMES)):
            for second_field in range(len(INPUT_FIELD_NAMES)):
                candidates.append(
                    coefficients[:, first_field, radial, first_mode]
                    * coefficients[:, second_field, radial, second_mode]
                )
                labels.append((first_field, second_field, first_mode, second_mode))
    return np.column_stack(candidates), labels


def fit_phase_carriers(trajectories: dict[str, base.Trajectory]) -> list[dict]:
    indices = training_indices(trajectories)
    records = []
    for local_mode, mode in enumerate(HIDDEN_MODES):
        for radial in range(8):
            candidate_blocks = []
            labels = None
            truth_blocks = []
            for name, selected in indices.items():
                candidates, current_labels = carrier_candidates(
                    trajectories[name].coefficients[selected], mode, radial
                )
                candidate_blocks.append(candidates)
                truth_blocks.append(trajectories[name].coefficients[selected, :, radial, mode])
                labels = current_labels
            candidates = np.concatenate(candidate_blocks)
            truth = np.concatenate(truth_blocks)
            candidate_energy = np.sum(np.abs(candidates) ** 2, axis=0)
            for field in range(4):
                target = truth[:, field]
                target_energy = float(np.vdot(target, target).real)
                cross = np.sum(np.conj(candidates) * target[:, None], axis=0)
                coherence = np.abs(cross) / np.sqrt(
                    np.maximum(candidate_energy * target_energy, TINY)
                )
                best = int(np.argmax(coherence))
                phase_offset = cross[best] / max(abs(cross[best]), TINY)
                first_field, second_field, first_mode, second_mode = labels[best]
                records.append(
                    {
                        "field": field,
                        "radial": radial,
                        "local_mode": local_mode,
                        "mode": mode,
                        "first_field": first_field,
                        "second_field": second_field,
                        "first_mode": first_mode,
                        "second_mode": second_mode,
                        "phase_offset_real": float(phase_offset.real),
                        "phase_offset_imag": float(phase_offset.imag),
                        "training_coherence": float(coherence[best]),
                    }
                )
    return records


def analytic_carrier(coefficients: np.ndarray, records: list[dict]) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    result = np.zeros((len(coefficients), 4, 8, 5), dtype=np.complex128)
    for record in records:
        q = (
            coefficients[:, record["first_field"], record["radial"], record["first_mode"]]
            * coefficients[:, record["second_field"], record["radial"], record["second_mode"]]
        )
        offset = record["phase_offset_real"] + 1j * record["phase_offset_imag"]
        result[:, record["field"], record["radial"], record["local_mode"]] = q * offset
    return result


def predict_latent(model, trajectory, indices, args, device, use_control):
    allowed = set(map(int, indices))
    targets = np.asarray(
        [
            int(i)
            for i in indices
            if int(i) - args.history_steps + 1 >= 0
            and all(j in allowed for j in range(int(i) - args.history_steps + 1, int(i) + 1))
        ],
        dtype=np.int64,
    )
    values = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(targets), args.batch_size):
            selected = targets[begin : begin + args.batch_size]
            x = np.stack([trajectory.observable[i-args.history_steps+1:i+1] for i in selected]).astype(np.float32)
            u = np.stack([trajectory.controls[i-args.history_steps+1:i+1] for i in selected]).astype(np.float32)
            value = base.forward_model(
                model, torch.from_numpy(x).to(device), torch.from_numpy(u).to(device), use_control
            )
            values.append(value.cpu().numpy())
    return targets, np.concatenate(values)


def centered_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel().copy()
    right = np.asarray(right, dtype=np.float64).ravel().copy()
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / max(denominator, TINY))


def envelope_metrics(
    truth: np.ndarray,
    amplitude: np.ndarray,
    reconstruction: np.ndarray,
    radial_weights: np.ndarray,
    time_us: np.ndarray,
) -> dict:
    truth_amplitude = np.abs(truth)
    amplitude_relative_l2 = math.sqrt(
        float(np.sum((amplitude - truth_amplitude) ** 2))
        / max(float(np.sum(truth_amplitude**2)), TINY)
    )
    weights = radial_weights.reshape((1,) * (truth.ndim - 3) + (1, 8, 1))
    truth_power = np.sum(np.abs(truth) ** 2 * weights, axis=tuple(range(1, truth.ndim)))
    predicted_power = np.sum(amplitude**2 * weights, axis=tuple(range(1, amplitude.ndim)))
    power_relative_l2 = math.sqrt(
        float(np.sum((predicted_power - truth_power) ** 2))
        / max(float(np.sum(truth_power**2)), TINY)
    )
    complex_values = base.complex_metrics(truth, reconstruction)
    spectrum = base.temporal_spectrum_metrics(
        truth_power[:, None].astype(np.complex128),
        predicted_power[:, None].astype(np.complex128),
        time_us,
    )
    return {
        "amplitude_relative_l2": amplitude_relative_l2,
        "mean_power_ratio": float(np.mean(predicted_power) / max(float(np.mean(truth_power)), TINY)),
        "power_series_relative_l2": power_relative_l2,
        "power_time_correlation": centered_correlation(truth_power, predicted_power),
        "power_spectral_cosine": spectrum["spectral_cosine"],
        "complex_relative_l2": complex_values["relative_l2"],
        "complex_coherence": complex_values["coherence"],
    }


def audit(models, trajectories, transforms, carriers, args, device):
    rows = []
    series = {}
    for case, trajectory in trajectories.items():
        for split in ("validation", "test"):
            if split not in base.SPLITS[case]:
                continue
            split_indices = base.interval_indices(trajectory, split)
            valid = split_indices[args.history_steps - 1 :]
            truth = trajectory.coefficients[valid, :, :, 17:22]
            carrier = analytic_carrier(trajectory.coefficients[valid], carriers)
            oracle_reconstruction = apply_hidden_amplitude_to_carrier(carrier, np.abs(truth))
            oracle_metrics = envelope_metrics(
                truth, np.abs(truth), oracle_reconstruction, trajectory.radial_weights, trajectory.time_us[valid]
            )
            rows.append({"case": case, "split": split, "model": "analytic_carrier_amplitude_oracle", "field": "all", "mode": "17-21", "frames": len(valid), **oracle_metrics})
            for model_name, (model, _, _, use_control) in models.items():
                indices, latent = predict_latent(model, trajectory, split_indices, args, device, use_control)
                amplitude = decode_amplitude(latent, transforms)
                truth = trajectory.coefficients[indices, :, :, 17:22]
                carrier = analytic_carrier(trajectory.coefficients[indices], carriers)
                reconstruction = apply_hidden_amplitude_to_carrier(carrier, amplitude)
                metrics = envelope_metrics(
                    truth, amplitude, reconstruction, trajectory.radial_weights, trajectory.time_us[indices]
                )
                rows.append({"case": case, "split": split, "model": model_name, "field": "all", "mode": "17-21", "frames": len(indices), **metrics})
                for field_index, field in enumerate(FIELD_NAMES):
                    for local_mode, mode in enumerate(HIDDEN_MODES):
                        item = envelope_metrics(
                            truth[:, field_index:field_index+1, :, local_mode:local_mode+1],
                            amplitude[:, field_index:field_index+1, :, local_mode:local_mode+1],
                            reconstruction[:, field_index:field_index+1, :, local_mode:local_mode+1],
                            trajectory.radial_weights,
                            trajectory.time_us[indices],
                        )
                        rows.append({"case": case, "split": split, "model": model_name, "field": field, "mode": mode, "frames": len(indices), **item})
                weights = trajectory.radial_weights[None, None, :, None]
                series[f"{case}:{split}:{model_name}"] = {
                    "time": trajectory.time_us[indices],
                    "truth": np.sum(np.abs(truth)**2 * weights, axis=(1,2,3)),
                    "prediction": np.sum(amplitude**2 * weights, axis=(1,2,3)),
                }
    return rows, series


def make_plots(history_rows, series, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in sorted(set(row["model"] for row in history_rows)):
        selected = [row for row in history_rows if row["model"] == name]
        ax.semilogy([row["epoch"] for row in selected], [row["validation_loss"] for row in selected], label=name)
    ax.set(xlabel="epoch", ylabel="validation latent MSE", title="Electric-history envelope ROM")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(output / "training_validation_loss.png", dpi=180); plt.close(fig)
    cases = ("E20_to_E22p5", "E22p5_to_E20", "E25_stationary_condition_holdout")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    colors = {"instant_electric":"tab:orange", "history_no_electric":"tab:green", "history_electric":"tab:blue"}
    for ax, case in zip(axes, cases):
        plotted = False
        for model, color in colors.items():
            item = series.get(f"{case}:test:{model}")
            if item is None: continue
            if not plotted:
                ax.plot(item["time"], item["truth"], color="black", lw=1.4, label="truth"); plotted=True
            ax.plot(item["time"], item["prediction"], color=color, lw=1.0, label=model)
        ax.set(title=case, ylabel="n=17-21 power"); ax.grid(True, alpha=0.25); ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("time [us]"); fig.tight_layout()
    fig.savefig(output / "hidden_band_envelope_power.png", dpi=180); plt.close(fig)


def summarize(rows, transforms, carriers, datasets, args):
    aggregate = [row for row in rows if row["split"] == "test" and row["field"] == "all"]
    by_model = {}
    for model in sorted(set(row["model"] for row in aggregate)):
        selected = [row for row in aggregate if row["model"] == model]
        keys = ("amplitude_relative_l2", "mean_power_ratio", "power_series_relative_l2", "power_time_correlation", "power_spectral_cosine", "complex_relative_l2", "complex_coherence")
        by_model[model] = {f"mean_{key}": float(np.mean([row[key] for row in selected])) for key in keys}
    e25 = {row["model"]: {k:v for k,v in row.items() if k not in ("case","split","model","field","mode")} for row in aggregate if row["case"] == "E25_stationary_condition_holdout"}
    electric = by_model["history_electric"]
    no_electric = by_model["history_no_electric"]
    instant = by_model["instant_electric"]
    return {
        "status": "trained_development_not_primary_confirmed",
        "representation": "SO2-equivariant carrier plus electric-history amplitude envelope",
        "configuration": vars(args),
        "input_fields": list(INPUT_FIELD_NAMES),
        "visible_modes": list(range(17)),
        "predicted_hidden_modes": list(HIDDEN_MODES),
        "samples": {name: len(dataset) for name,dataset in datasets.items()},
        "pca": {
            "observable_components": int(transforms["observable_pca"].n_components_),
            "observable_explained_variance_fraction": float(np.sum(transforms["observable_pca"].explained_variance_ratio_)),
            "envelope_components": int(transforms["hidden_pca"].n_components_),
            "envelope_explained_variance_fraction": float(np.sum(transforms["hidden_pca"].explained_variance_ratio_)),
        },
        "test_aggregate_by_model": by_model,
        "e25_stationary_condition_holdout": e25,
        "electric_history_increment": {
            "power_error_reduction_vs_history_no_electric": no_electric["mean_power_series_relative_l2"] - electric["mean_power_series_relative_l2"],
            "power_error_reduction_vs_instant": instant["mean_power_series_relative_l2"] - electric["mean_power_series_relative_l2"],
            "power_spectral_cosine_gain_vs_history_no_electric": electric["mean_power_spectral_cosine"] - no_electric["mean_power_spectral_cosine"],
        },
        "analytic_carrier_mean_training_coherence": float(np.mean([record["training_coherence"] for record in carriers])),
        "primary_e25_to_e22p5_used": False,
        "primary_e25_to_e22p5_available": False,
        "claim_boundary": "Power/envelope recovery is distinct from exact complex-field recovery; primary transition remains required.",
    }


def main() -> None:
    args = parse_args()
    base.set_seed(args.seed)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    trajectories = {name: base.load_case(name, spec) for name,spec in base.CASES.items()}
    transforms = fit_transforms(trajectories, args.observable_components, args.hidden_components)
    carriers = fit_phase_carriers(trajectories)
    joblib.dump(transforms, output / "transforms.joblib")
    (output / "analytic_phase_carriers.json").write_text(json.dumps(base.json_safe(carriers), indent=2), encoding="utf-8")
    datasets = {split: base.WindowDataset(trajectories, split, args.history_steps) for split in ("train","validation","test")}
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=generator, pin_memory=device.type=="cuda")
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type=="cuda")
    specifications = base.model_specifications(args, int(transforms["observable_pca"].n_components_), int(transforms["hidden_pca"].n_components_))
    trained = {}; history_rows=[]
    for name,(model,model_type,kwargs,use_control) in specifications.items():
        history_rows.extend(base.train_model(name,model,model_type,kwargs,use_control,train_loader,validation_loader,args,output,device))
        trained[name]=(model,model_type,kwargs,use_control)
    rows, series = audit(trained, trajectories, transforms, carriers, args, device)
    base.write_csv(output / "training_history.csv", history_rows)
    base.write_csv(output / "hidden_band_envelope_metrics.csv", rows)
    make_plots(history_rows, series, output)
    summary = summarize(rows, transforms, carriers, datasets, args)
    (output / "summary.json").write_text(json.dumps(base.json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(base.json_safe(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
