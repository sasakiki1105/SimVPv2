#!/usr/bin/env python3
"""Train and audit an electric-history ROM for the G8-hidden n=17--21 band.

The primary E25 -> E22.5 transition is deliberately absent.  It remains a
future, completely unseen electric-history test.  Available stationary E20,
E25, E30 data and E20 <-> E22.5 transitions are split chronologically.  Three
models distinguish the source of any skill: current-state MLP, history-only
GRU, and history plus prescribed electric-field memory GRU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from radaz_electric_history_hidden_band_rom import (
    CONTROL_DIMENSION,
    FIELD_NAMES,
    HIDDEN_MODES,
    INPUT_FIELD_NAMES,
    VISIBLE_MODES,
    HistoryHiddenBandROM,
    InstantHiddenBandROM,
    complex_to_real,
    electric_history_controls,
    real_to_complex,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "workdirs" / "radaz_electric_history_hidden_band_rom"
TINY = np.finfo(np.float64).tiny


@dataclass
class Trajectory:
    name: str
    path: Path
    coefficients: np.ndarray
    time_us: np.ndarray
    radial_weights: np.ndarray
    current_ez_kvm: float
    source_ez_kvm: float
    transition: bool
    controls: np.ndarray
    observable: np.ndarray | None = None
    hidden: np.ndarray | None = None


CASES = {
    "E20_stationary": (
        ROOT / "workdirs/compare_radaz_local_rom_closure_map/cases/E20kVm/physical_fourier_targets.h5",
        20.0,
        20.0,
        False,
    ),
    "E25_stationary_condition_holdout": (
        ROOT / "workdirs/compare_radaz_local_rom_closure_map/cases/E25kVm/physical_fourier_targets.h5",
        25.0,
        25.0,
        False,
    ),
    "E30_stationary": (
        ROOT / "workdirs/compare_radaz_local_rom_closure_map/cases/E30kVm/physical_fourier_targets.h5",
        30.0,
        30.0,
        False,
    ),
    "E20_to_E22p5": (
        ROOT / "workdirs/radaz_e20_to_e22p5_transition/physical_fourier_targets.h5",
        22.5,
        20.0,
        True,
    ),
    "E22p5_to_E20": (
        ROOT / "workdirs/radaz_e22p5_to_e20_transition/physical_fourier_targets.h5",
        20.0,
        22.5,
        True,
    ),
}


SPLITS = {
    "E20_stationary": {"train": (15.0, 24.0), "validation": (24.0, 27.0), "test": (27.0, 29.87)},
    "E30_stationary": {"train": (15.0, 24.0), "validation": (24.0, 27.0), "test": (27.0, 29.87)},
    "E20_to_E22p5": {"train": (30.16, 36.0), "validation": (36.0, 38.0), "test": (38.0, 39.86)},
    "E22p5_to_E20": {"train": (30.16, 33.2), "validation": (33.2, 34.0), "test": (34.0, 34.87)},
    "E25_stationary_condition_holdout": {"test": (20.0, 29.87)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-steps", type=int, default=24)
    parser.add_argument("--observable-components", type=int, default=64)
    parser.add_argument("--hidden-components", type=int, default=96)
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


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_case(name: str, specification: tuple[Path, float, float, bool]) -> Trajectory:
    path, current, source, transition = specification
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        coefficients = np.asarray(handle["coefficients"], dtype=np.complex64)
        time_us = np.asarray(handle["time_us"], dtype=np.float64)
        weights = np.asarray(handle["radial_weights"], dtype=np.float64)
        fields = tuple(
            item.decode() if isinstance(item, bytes) else str(item)
            for item in handle["fields"][:]
        )
        modes = tuple(int(item) for item in handle["modes"][:])
    if fields != FIELD_NAMES or modes[:22] != tuple(range(22)):
        raise ValueError(f"unexpected Fourier layout in {path}: {fields}, {modes}")
    if coefficients.shape[1:] != (4, 8, 22):
        raise ValueError(f"unexpected coefficient shape in {path}: {coefficients.shape}")
    return Trajectory(
        name=name,
        path=path,
        coefficients=coefficients,
        time_us=time_us,
        radial_weights=weights,
        current_ez_kvm=current,
        source_ez_kvm=source,
        transition=transition,
        controls=electric_history_controls(time_us, current, source, transition),
    )


def interval_indices(trajectory: Trajectory, split: str) -> np.ndarray:
    begin, end = SPLITS[trajectory.name][split]
    return np.flatnonzero((trajectory.time_us >= begin) & (trajectory.time_us < end))


def fit_transforms(
    trajectories: dict[str, Trajectory], observable_components: int, hidden_components: int
) -> dict:
    visible_rows = []
    hidden_rows = []
    for trajectory in trajectories.values():
        if "train" not in SPLITS[trajectory.name]:
            continue
        indices = interval_indices(trajectory, "train")
        visible_rows.append(
            complex_to_real(trajectory.coefficients[indices, : len(INPUT_FIELD_NAMES), :, :17])
        )
        hidden_rows.append(complex_to_real(trajectory.coefficients[indices, ..., 17:22]))
    visible = np.concatenate(visible_rows)
    hidden = np.concatenate(hidden_rows)
    observable_scaler = StandardScaler().fit(visible)
    hidden_scaler = StandardScaler().fit(hidden)
    visible_scaled = observable_scaler.transform(visible)
    hidden_scaled = hidden_scaler.transform(hidden)
    observable_pca = PCA(
        n_components=min(observable_components, *visible_scaled.shape),
        whiten=True,
        svd_solver="randomized",
        random_state=0,
    ).fit(visible_scaled)
    hidden_pca = PCA(
        n_components=min(hidden_components, *hidden_scaled.shape),
        whiten=True,
        svd_solver="randomized",
        random_state=1,
    ).fit(hidden_scaled)
    for trajectory in trajectories.values():
        visible_flat = complex_to_real(
            trajectory.coefficients[:, : len(INPUT_FIELD_NAMES), :, :17]
        )
        hidden_flat = complex_to_real(trajectory.coefficients[..., 17:22])
        trajectory.observable = observable_pca.transform(
            observable_scaler.transform(visible_flat)
        ).astype(np.float32)
        trajectory.hidden = hidden_pca.transform(
            hidden_scaler.transform(hidden_flat)
        ).astype(np.float32)
    return {
        "observable_scaler": observable_scaler,
        "observable_pca": observable_pca,
        "hidden_scaler": hidden_scaler,
        "hidden_pca": hidden_pca,
    }


class WindowDataset(Dataset):
    def __init__(
        self,
        trajectories: dict[str, Trajectory],
        split: str,
        history_steps: int,
    ) -> None:
        self.trajectories = trajectories
        self.history_steps = history_steps
        self.samples: list[tuple[str, int]] = []
        for name, trajectory in trajectories.items():
            if split not in SPLITS[name]:
                continue
            indices = interval_indices(trajectory, split)
            if not len(indices):
                continue
            allowed = set(int(i) for i in indices)
            for target in indices:
                start = int(target) - history_steps + 1
                if start >= 0 and all(i in allowed for i in range(start, int(target) + 1)):
                    self.samples.append((name, int(target)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        name, target = self.samples[index]
        trajectory = self.trajectories[name]
        start = target - self.history_steps + 1
        return (
            torch.from_numpy(trajectory.observable[start : target + 1]),
            torch.from_numpy(trajectory.controls[start : target + 1]),
            torch.from_numpy(trajectory.hidden[target]),
        )


def model_specifications(args: argparse.Namespace, observable_dim: int, output_dim: int):
    recurrent_kwargs = {
        "observable_dimension": observable_dim,
        "output_dimension": output_dim,
        "hidden_dimension": args.hidden_dimension,
        "layers": args.layers,
        "control_dimension": CONTROL_DIMENSION,
        "dropout": 0.10,
    }
    return {
        "instant_electric": (
            InstantHiddenBandROM(
                observable_dim, output_dim, args.hidden_dimension, CONTROL_DIMENSION
            ),
            "instant",
            {
                "observable_dimension": observable_dim,
                "output_dimension": output_dim,
                "hidden_dimension": args.hidden_dimension,
                "control_dimension": CONTROL_DIMENSION,
            },
            True,
        ),
        "history_no_electric": (
            HistoryHiddenBandROM(**{**recurrent_kwargs, "control_dimension": 0}),
            "history",
            {**recurrent_kwargs, "control_dimension": 0},
            False,
        ),
        "history_electric": (
            HistoryHiddenBandROM(**recurrent_kwargs),
            "history",
            recurrent_kwargs,
            True,
        ),
    }


def forward_model(model: nn.Module, x: torch.Tensor, u: torch.Tensor, use_control: bool):
    if not use_control:
        u = u[..., :0]
    return model(x, u)


def evaluate_loss(model, loader, device, use_control) -> float:
    model.eval()
    square = 0.0
    count = 0
    with torch.inference_mode():
        for x, u, y in loader:
            x, u, y = x.to(device), u.to(device), y.to(device)
            prediction = forward_model(model, x, u, use_control)
            square += float(torch.sum((prediction - y) ** 2).item())
            count += y.numel()
    return square / max(count, 1)


def train_model(
    name: str,
    model: nn.Module,
    model_type: str,
    model_kwargs: dict,
    use_control: bool,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
    output: Path,
    device: torch.device,
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=2.0e-5
    )
    best = math.inf
    stale = 0
    rows = []
    checkpoint_path = output / f"{name}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        square = 0.0
        count = 0
        for x, u, y in train_loader:
            x, u, y = x.to(device), u.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_model(model, x, u, use_control)
            loss = torch.mean((prediction - y) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            square += float(torch.sum((prediction.detach() - y) ** 2).item())
            count += y.numel()
        train_loss = square / count
        validation_loss = evaluate_loss(model, validation_loader, device, use_control)
        scheduler.step(validation_loss)
        row = {
            "model": name,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)
        print(
            f"[{name}] epoch={epoch:03d} train={train_loss:.6e} "
            f"val={validation_loss:.6e} lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        if validation_loss < best - 1.0e-6:
            best = validation_loss
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": model_type,
                    "model_kwargs": model_kwargs,
                    "history_steps": args.history_steps,
                    "uses_electric_history": use_control,
                    "best_validation_loss": best,
                    "best_epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return rows


def decode_hidden(encoded: np.ndarray, transforms: dict) -> np.ndarray:
    scaled = transforms["hidden_pca"].inverse_transform(encoded)
    flattened = transforms["hidden_scaler"].inverse_transform(scaled)
    return real_to_complex(flattened, 4, 8, 5)


def predict_case(
    model: nn.Module,
    trajectory: Trajectory,
    indices: np.ndarray,
    history_steps: int,
    device: torch.device,
    use_control: bool,
    transforms: dict,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    allowed = set(int(index) for index in indices)
    targets = np.asarray(
        [
            int(index)
            for index in indices
            if int(index) - history_steps + 1 >= 0
            and all(
                value in allowed
                for value in range(int(index) - history_steps + 1, int(index) + 1)
            )
        ],
        dtype=np.int64,
    )
    outputs = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(targets), batch_size):
            selected = targets[begin : begin + batch_size]
            x = np.stack(
                [trajectory.observable[i - history_steps + 1 : i + 1] for i in selected]
            ).astype(np.float32)
            u = np.stack(
                [trajectory.controls[i - history_steps + 1 : i + 1] for i in selected]
            ).astype(np.float32)
            prediction = forward_model(
                model,
                torch.from_numpy(x).to(device),
                torch.from_numpy(u).to(device),
                use_control,
            )
            outputs.append(prediction.cpu().numpy())
    encoded = np.concatenate(outputs, axis=0)
    return targets, decode_hidden(encoded, transforms)


def complex_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth_energy = float(np.vdot(truth, truth).real)
    prediction_energy = float(np.vdot(prediction, prediction).real)
    error_energy = float(np.vdot(prediction - truth, prediction - truth).real)
    cross = np.vdot(prediction, truth)
    return {
        "relative_l2": math.sqrt(error_energy / max(truth_energy, TINY)),
        "power_ratio": prediction_energy / max(truth_energy, TINY),
        "coherence": float(
            abs(cross) / max(math.sqrt(prediction_energy * truth_energy), TINY)
        ),
    }


def temporal_spectrum_metrics(
    truth: np.ndarray, prediction: np.ndarray, time_us: np.ndarray
) -> dict[str, float]:
    if len(time_us) < 8:
        return {"spectral_cosine": math.nan, "truth_peak_mhz": math.nan, "prediction_peak_mhz": math.nan}
    dt_us = float(np.median(np.diff(time_us)))
    truth = np.asarray(truth, dtype=np.complex128)
    prediction = np.asarray(prediction, dtype=np.complex128)
    truth_centered = truth - np.mean(truth, axis=0, keepdims=True)
    prediction_centered = prediction - np.mean(prediction, axis=0, keepdims=True)
    truth_psd = np.sum(np.abs(np.fft.fft(truth_centered, axis=0)) ** 2, axis=tuple(range(1, truth.ndim)))
    prediction_psd = np.sum(np.abs(np.fft.fft(prediction_centered, axis=0)) ** 2, axis=tuple(range(1, prediction.ndim)))
    frequency = np.fft.fftfreq(len(time_us), d=dt_us)
    nonzero = np.abs(frequency) > 1.0e-12
    truth_psd = truth_psd[nonzero]
    prediction_psd = prediction_psd[nonzero]
    frequency = frequency[nonzero]
    truth_norm = float(np.linalg.norm(truth_psd))
    prediction_norm = float(np.linalg.norm(prediction_psd))
    cosine = (
        float(np.dot(truth_psd / truth_norm, prediction_psd / prediction_norm))
        if truth_norm > 0.0 and prediction_norm > 0.0
        else 0.0
    )
    return {
        "spectral_cosine": cosine,
        "truth_peak_mhz": float(frequency[int(np.argmax(truth_psd))]),
        "prediction_peak_mhz": (
            float(frequency[int(np.argmax(prediction_psd))])
            if prediction_norm > 0.0
            else math.nan
        ),
    }


def audit_predictions(
    models: dict,
    trajectories: dict[str, Trajectory],
    transforms: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    time_series: dict = {}
    evaluations = []
    for name in trajectories:
        for split in ("validation", "test"):
            if split in SPLITS[name]:
                evaluations.append((name, split))
    for case_name, split in evaluations:
        trajectory = trajectories[case_name]
        split_indices = interval_indices(trajectory, split)
        valid_indices = split_indices[args.history_steps - 1 :]
        truth_for_baseline = trajectory.coefficients[valid_indices, ..., 17:22]
        zero_prediction = np.zeros_like(truth_for_baseline)
        zero_metrics = complex_metrics(truth_for_baseline, zero_prediction)
        zero_metrics.update(
            temporal_spectrum_metrics(
                truth_for_baseline,
                zero_prediction,
                trajectory.time_us[valid_indices],
            )
        )
        rows.append(
            {
                "case": case_name,
                "split": split,
                "model": "ideal_g8_zero_hidden",
                "field": "all",
                "mode": "17-21",
                "frames": len(valid_indices),
                **zero_metrics,
            }
        )
        encoded_truth = trajectory.hidden[valid_indices]
        pca_prediction = decode_hidden(encoded_truth, transforms)
        pca_metrics = complex_metrics(truth_for_baseline, pca_prediction)
        pca_metrics.update(
            temporal_spectrum_metrics(
                truth_for_baseline,
                pca_prediction,
                trajectory.time_us[valid_indices],
            )
        )
        rows.append(
            {
                "case": case_name,
                "split": split,
                "model": "hidden_pca_oracle_ceiling",
                "field": "all",
                "mode": "17-21",
                "frames": len(valid_indices),
                **pca_metrics,
            }
        )
        for model_name, (model, _, _, use_control) in models.items():
            indices, prediction = predict_case(
                model,
                trajectory,
                split_indices,
                args.history_steps,
                device,
                use_control,
                transforms,
                args.batch_size,
            )
            truth = trajectory.coefficients[indices, ..., 17:22]
            aggregate = complex_metrics(truth, prediction)
            aggregate.update(temporal_spectrum_metrics(truth, prediction, trajectory.time_us[indices]))
            rows.append(
                {
                    "case": case_name,
                    "split": split,
                    "model": model_name,
                    "field": "all",
                    "mode": "17-21",
                    "frames": len(indices),
                    **aggregate,
                }
            )
            for field_index, field in enumerate(FIELD_NAMES):
                for local_mode, mode in enumerate(HIDDEN_MODES):
                    metrics = complex_metrics(
                        truth[:, field_index, :, local_mode],
                        prediction[:, field_index, :, local_mode],
                    )
                    metrics.update(
                        temporal_spectrum_metrics(
                            truth[:, field_index, :, local_mode],
                            prediction[:, field_index, :, local_mode],
                            trajectory.time_us[indices],
                        )
                    )
                    rows.append(
                        {
                            "case": case_name,
                            "split": split,
                            "model": model_name,
                            "field": field,
                            "mode": mode,
                            "frames": len(indices),
                            **metrics,
                        }
                    )
            weighted_truth = np.sum(
                np.abs(truth) ** 2 * trajectory.radial_weights[None, None, :, None],
                axis=(1, 2, 3),
            )
            weighted_prediction = np.sum(
                np.abs(prediction) ** 2 * trajectory.radial_weights[None, None, :, None],
                axis=(1, 2, 3),
            )
            key = f"{case_name}:{split}:{model_name}"
            time_series[key] = {
                "time_us": trajectory.time_us[indices],
                "truth_power": weighted_truth,
                "prediction_power": weighted_prediction,
            }
    return rows, time_series


def plot_training(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted(set(row["model"] for row in rows)):
        selected = [row for row in rows if row["model"] == model]
        ax.semilogy([row["epoch"] for row in selected], [row["validation_loss"] for row in selected], label=model)
    ax.set(xlabel="epoch", ylabel="validation latent MSE", title="Hidden-band ROM validation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_validation_loss.png", dpi=180)
    plt.close(fig)


def plot_transition_power(time_series: dict, output: Path) -> None:
    cases = ("E20_to_E22p5", "E22p5_to_E20", "E25_stationary_condition_holdout")
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 9), sharex=False)
    colors = {"instant_electric": "tab:orange", "history_no_electric": "tab:green", "history_electric": "tab:blue"}
    for ax, case in zip(axes, cases):
        split = "test"
        truth_plotted = False
        for model, color in colors.items():
            key = f"{case}:{split}:{model}"
            if key not in time_series:
                continue
            values = time_series[key]
            if not truth_plotted:
                ax.plot(values["time_us"], values["truth_power"], color="black", lw=1.5, label="truth")
                truth_plotted = True
            ax.plot(values["time_us"], values["prediction_power"], color=color, lw=1.0, label=model)
        ax.set(title=case, ylabel=r"weighted $n=17$--$21$ power")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("time [us]")
    fig.tight_layout()
    fig.savefig(output / "hidden_band_power_timeseries.png", dpi=180)
    plt.close(fig)


def summarize(rows: list[dict], transforms: dict, trajectories: dict, datasets: dict, args) -> dict:
    aggregate = [row for row in rows if row["field"] == "all"]
    test_rows = [row for row in aggregate if row["split"] == "test"]
    by_model = {}
    for model in sorted(set(row["model"] for row in test_rows)):
        selected = [row for row in test_rows if row["model"] == model]
        by_model[model] = {
            "mean_relative_l2": float(np.mean([row["relative_l2"] for row in selected])),
            "mean_power_ratio": float(np.mean([row["power_ratio"] for row in selected])),
            "mean_coherence": float(np.mean([row["coherence"] for row in selected])),
            "mean_temporal_spectral_cosine": float(np.mean([row["spectral_cosine"] for row in selected])),
        }
    e25 = {
        row["model"]: {
            key: row[key]
            for key in ("relative_l2", "power_ratio", "coherence", "spectral_cosine", "truth_peak_mhz", "prediction_peak_mhz")
        }
        for row in test_rows
        if row["case"] == "E25_stationary_condition_holdout"
    }
    history = by_model.get("history_electric", {})
    no_history = by_model.get("history_no_electric", {})
    instant = by_model.get("instant_electric", {})
    return {
        "status": "trained_development_not_primary_confirmed",
        "interpretation": (
            "The model reconstructs spatial Fourier modes n=17--21 from ideal G8-observable "
            "modes n=0--16 plus causal electric history. The unavailable E25->E22.5 run remains "
            "the locked unseen-history confirmatory test."
        ),
        "configuration": vars(args),
        "visible_modes": list(VISIBLE_MODES),
        "hidden_modes": list(HIDDEN_MODES),
        "input_field_names": list(INPUT_FIELD_NAMES),
        "field_names": list(FIELD_NAMES),
        "samples": {name: len(dataset) for name, dataset in datasets.items()},
        "pca": {
            "observable_components": int(transforms["observable_pca"].n_components_),
            "observable_explained_variance_fraction": float(np.sum(transforms["observable_pca"].explained_variance_ratio_)),
            "hidden_components": int(transforms["hidden_pca"].n_components_),
            "hidden_explained_variance_fraction": float(np.sum(transforms["hidden_pca"].explained_variance_ratio_)),
        },
        "test_aggregate_by_model": by_model,
        "e25_stationary_full_condition_holdout": e25,
        "electric_history_increment": {
            "relative_l2_reduction_vs_history_no_electric": (
                no_history.get("mean_relative_l2", math.nan) - history.get("mean_relative_l2", math.nan)
            ),
            "relative_l2_reduction_vs_instant": (
                instant.get("mean_relative_l2", math.nan) - history.get("mean_relative_l2", math.nan)
            ),
            "spectral_cosine_gain_vs_history_no_electric": (
                history.get("mean_temporal_spectral_cosine", math.nan) - no_history.get("mean_temporal_spectral_cosine", math.nan)
            ),
        },
        "primary_e25_to_e22p5_used": False,
        "primary_e25_to_e22p5_available": False,
        "data_sources": {name: str(trajectory.path) for name, trajectory in trajectories.items()},
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    trajectories = {name: load_case(name, spec) for name, spec in CASES.items()}
    transforms = fit_transforms(
        trajectories, args.observable_components, args.hidden_components
    )
    joblib.dump(transforms, output / "transforms.joblib")
    datasets = {
        split: WindowDataset(trajectories, split, args.history_steps)
        for split in ("train", "validation", "test")
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=generator, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        datasets["validation"], batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    observable_dim = int(transforms["observable_pca"].n_components_)
    output_dim = int(transforms["hidden_pca"].n_components_)
    specifications = model_specifications(args, observable_dim, output_dim)
    trained = {}
    history_rows = []
    for name, (model, model_type, kwargs, use_control) in specifications.items():
        history_rows.extend(
            train_model(
                name, model, model_type, kwargs, use_control,
                train_loader, validation_loader, args, output, device,
            )
        )
        trained[name] = (model, model_type, kwargs, use_control)
    rows, time_series = audit_predictions(
        trained, trajectories, transforms, args, device
    )
    write_csv(output / "training_history.csv", history_rows)
    write_csv(output / "hidden_band_metrics.csv", rows)
    plot_training(history_rows, output)
    plot_transition_power(time_series, output)
    summary = summarize(rows, transforms, trajectories, datasets, args)
    (output / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
