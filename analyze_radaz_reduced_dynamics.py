"""Fit reduced latent dynamics and forecast the unseen RadAz interval.

This script reuses the steady PCA scores extracted from a frozen SimVP model.
It fits all reduced models only on 20-24 us and performs an autonomous
multi-step rollout over 24-30 us.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez10kvm_latent"
    / "latent_pca_scores.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez10kvm_reduced_dynamics"
)
DEFAULT_LATENT = DEFAULT_INPUT.parent / "radaz_latent_features.h5"
DEFAULT_PCA_DIR = DEFAULT_INPUT.parent

FIT_START_US = 20.0
FORECAST_START_US = 24.0
FORECAST_END_US = 30.0
SINDY_VALIDATION_START_US = 23.0


@dataclass
class LayerData:
    name: str
    components: int
    time_us: np.ndarray
    scores: np.ndarray


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


def read_score_csv(
    path: Path,
    component_map: dict[str, int],
    latent_path: Path,
    pca_dir: Path,
) -> dict[str, LayerData]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No PCA score rows found in {path}")

    result: dict[str, LayerData] = {}
    for layer, count in component_map.items():
        time_us = np.asarray(
            [float(row[f"{layer}_time_us"]) for row in rows], dtype=np.float64
        )
        score_fields = [
            f"{layer}_steady_pc{component + 1}" for component in range(count)
        ]
        if all(field in rows[0] for field in score_fields):
            scores = np.asarray(
                [
                    [float(row[field]) for field in score_fields]
                    for row in rows
                ],
                dtype=np.float64,
            )
        else:
            pca_path = pca_dir / f"pca_{layer}_steady.npz"
            with np.load(pca_path) as pca:
                mean = np.asarray(pca["mean"], dtype=np.float64)
                components = np.asarray(
                    pca["components"][:count], dtype=np.float64
                )
            dataset_name = f"{layer}_pooled"
            with h5py.File(latent_path, "r") as source:
                features = np.asarray(
                    source[dataset_name], dtype=np.float32
                ).reshape(len(time_us), -1)
            scores = (features.astype(np.float64) - mean) @ components.T
            print(
                f"[PCA] reconstructed {layer} PC1-PC{count} "
                f"from {latent_path.name}"
            )
        result[layer] = LayerData(layer, count, time_us, scores)
    return result


def fit_standardizer(values: np.ndarray) -> Standardizer:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return Standardizer(mean, scale)


def fit_dmd(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = states[:-1].T
    y = states[1:].T
    matrix = y @ np.linalg.pinv(x, rcond=1.0e-10)
    eigenvalues = np.linalg.eigvals(matrix)
    return matrix, eigenvalues


def rollout_linear(matrix: np.ndarray, initial: np.ndarray, steps: int) -> np.ndarray:
    forecast = np.empty((steps, initial.size), dtype=np.float64)
    state = initial.copy()
    for index in range(steps):
        state = matrix @ state
        if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > 1.0e8:
            forecast[index:] = np.nan
            break
        forecast[index] = state
    return forecast


def polynomial_library(states: np.ndarray) -> tuple[np.ndarray, list[str]]:
    samples, dimensions = states.shape
    columns = [np.ones(samples, dtype=np.float64)]
    names = ["1"]
    for component in range(dimensions):
        columns.append(states[:, component])
        names.append(f"z{component + 1}")
    for left in range(dimensions):
        for right in range(left, dimensions):
            columns.append(states[:, left] * states[:, right])
            names.append(f"z{left + 1}*z{right + 1}")
    return np.column_stack(columns), names


def fit_stlsq(
    states: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    ridge: float = 1.0e-6,
    iterations: int = 12,
) -> tuple[np.ndarray, list[str]]:
    library, names = polynomial_library(states)
    feature_scale = np.sqrt(np.mean(library * library, axis=0))
    feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
    normalized = library / feature_scale
    feature_count = normalized.shape[1]
    output_count = targets.shape[1]

    gram = normalized.T @ normalized
    coefficient = np.linalg.solve(
        gram + ridge * np.eye(feature_count), normalized.T @ targets
    )
    for _ in range(iterations):
        previous = coefficient.copy()
        coefficient[np.abs(coefficient) < threshold] = 0.0
        for output in range(output_count):
            active = coefficient[:, output] != 0.0
            if not np.any(active):
                continue
            active_library = normalized[:, active]
            active_gram = active_library.T @ active_library
            coefficient[active, output] = np.linalg.solve(
                active_gram + ridge * np.eye(np.count_nonzero(active)),
                active_library.T @ targets[:, output],
            )
        if np.array_equal(coefficient == 0.0, previous == 0.0):
            break

    return coefficient / feature_scale[:, None], names


def rollout_sindy(
    coefficient: np.ndarray, initial: np.ndarray, steps: int
) -> np.ndarray:
    forecast = np.empty((steps, initial.size), dtype=np.float64)
    state = initial.copy()
    for index in range(steps):
        library, _ = polynomial_library(state[None, :])
        state = (library @ coefficient)[0]
        if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > 1.0e8:
            forecast[index:] = np.nan
            break
        forecast[index] = state
    return forecast


def select_sindy_threshold(
    standardized: np.ndarray,
    time_us: np.ndarray,
    fit_mask: np.ndarray,
    thresholds: list[float],
) -> tuple[float, list[dict]]:
    subtrain = fit_mask & (time_us < SINDY_VALIDATION_START_US)
    validation = fit_mask & (time_us >= SINDY_VALIDATION_START_US)
    train_states = standardized[subtrain]
    validation_truth = standardized[validation]
    if len(train_states) < 3 or len(validation_truth) < 1:
        raise ValueError("Insufficient internal training/validation samples for SINDy")

    trials: list[dict] = []
    for threshold in thresholds:
        coefficient, _ = fit_stlsq(
            train_states[:-1], train_states[1:], threshold
        )
        prediction = rollout_sindy(
            coefficient, train_states[-1], len(validation_truth)
        )
        finite = np.isfinite(prediction).all(axis=1)
        if np.all(finite):
            mse = float(np.mean((prediction - validation_truth) ** 2))
        else:
            mse = float("inf")
        nonzero = int(np.count_nonzero(coefficient))
        total = int(coefficient.size)
        objective = mse + 1.0e-3 * nonzero / total
        trials.append(
            {
                "threshold": threshold,
                "validation_mse": mse,
                "nonzero_coefficients": nonzero,
                "total_coefficients": total,
                "objective": objective,
            }
        )
    finite_trials = [trial for trial in trials if np.isfinite(trial["objective"])]
    if not finite_trials:
        return thresholds[-1], trials
    best = min(finite_trials, key=lambda trial: trial["objective"])
    return float(best["threshold"]), trials


def moving_average(values: np.ndarray, window: int = 21) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def safe_correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    finite = np.isfinite(prediction).all(axis=1)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    left = truth[finite].ravel()
    right = prediction[finite].ravel()
    if np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_prediction(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    time_us: np.ndarray,
) -> tuple[dict, np.ndarray]:
    finite = np.isfinite(prediction).all(axis=1)
    per_time = np.full(len(truth), np.nan, dtype=np.float64)
    per_time[finite] = np.sqrt(
        np.mean((prediction[finite] - truth[finite]) ** 2, axis=1)
    )
    persistence_error = np.sqrt(np.mean((persistence - truth) ** 2, axis=1))

    if np.all(finite):
        mse = float(np.mean((prediction - truth) ** 2))
        rmse = float(np.sqrt(mse))
    else:
        mse = float("inf")
        rmse = float("inf")
    persistence_mse = float(np.mean((persistence - truth) ** 2))
    skill = 1.0 - mse / persistence_mse if np.isfinite(mse) else float("-inf")

    smoothed = moving_average(per_time)
    persistence_smoothed = moving_average(persistence_error)
    interval_start = float(time_us[0])
    interval_duration = float(time_us[-1] - interval_start)
    failed = np.flatnonzero(~finite)
    finite_horizon = (
        float(time_us[failed[0]] - interval_start)
        if failed.size
        else interval_duration
    )
    crossing = np.flatnonzero(smoothed >= persistence_smoothed)
    better_horizon = (
        float(time_us[crossing[0]] - interval_start)
        if crossing.size
        else interval_duration
    )
    unit_crossing = np.flatnonzero(smoothed >= 1.0)
    one_sigma_horizon = (
        float(time_us[unit_crossing[0]] - interval_start)
        if unit_crossing.size
        else interval_duration
    )
    better_horizon = min(better_horizon, finite_horizon)
    one_sigma_horizon = min(one_sigma_horizon, finite_horizon)

    return (
        {
            "standardized_mse": mse,
            "standardized_rmse": rmse,
            "persistence_mse": persistence_mse,
            "skill_vs_persistence": float(skill),
            "flattened_correlation": safe_correlation(truth, prediction),
            "finite_fraction": float(np.mean(finite)),
            "fraction_frames_better_than_persistence": float(
                np.mean(per_time < persistence_error)
            ),
            "smoothed_better_than_persistence_horizon_us": better_horizon,
            "smoothed_one_sigma_horizon_us": one_sigma_horizon,
        },
        per_time,
    )


def interval_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    time_us: np.ndarray,
    start_us: float,
    end_us: float,
) -> dict:
    mask = (time_us >= start_us) & (
        time_us < end_us if end_us < FORECAST_END_US else time_us <= end_us
    )
    result, _ = evaluate_prediction(
        truth[mask], prediction[mask], persistence[mask], time_us[mask]
    )
    return result


def write_rollout_csv(
    path: Path,
    results: dict[str, dict],
) -> None:
    maximum = max(result["components"] for result in results.values())
    fields = ["layer", "time_us"]
    for prefix in ("true", "persistence", "dmd", "sindy"):
        fields.extend(f"{prefix}_pc{component + 1}" for component in range(maximum))
    fields.extend(["dmd_rmse", "sindy_rmse", "persistence_rmse"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer, result in results.items():
            for index, time_us in enumerate(result["forecast_time_us"]):
                row: dict[str, object] = {"layer": layer, "time_us": time_us}
                for prefix in ("true", "persistence", "dmd", "sindy"):
                    values = result[prefix][index]
                    for component, value in enumerate(values):
                        row[f"{prefix}_pc{component + 1}"] = value
                row["dmd_rmse"] = result["dmd_error"][index]
                row["sindy_rmse"] = result["sindy_error"][index]
                row["persistence_rmse"] = result["persistence_error"][index]
                writer.writerow(row)


def write_metrics_csv(path: Path, summary: dict) -> None:
    fields = [
        "layer",
        "components",
        "method",
        "interval_us",
        "standardized_rmse",
        "skill_vs_persistence",
        "flattened_correlation",
        "finite_fraction",
        "fraction_frames_better_than_persistence",
        "smoothed_better_than_persistence_horizon_us",
        "smoothed_one_sigma_horizon_us",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer, layer_summary in summary["layers"].items():
            for method in ("dmd", "sindy"):
                for interval, metrics in layer_summary["metrics"][method].items():
                    writer.writerow(
                        {
                            "layer": layer,
                            "components": layer_summary["components"],
                            "method": method,
                            "interval_us": interval,
                            **{
                                field: metrics.get(field, "")
                                for field in fields
                                if field
                                not in ("layer", "components", "method", "interval_us")
                            },
                        }
                    )


def plot_rollout_overview(path: Path, results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
    colors = {
        "truth": "black",
        "persistence": "#8a8a8a",
        "dmd": "#1675b8",
        "sindy": "#c63f3f",
    }
    for row, (layer, result) in enumerate(results.items()):
        train_time = result["fit_time_us"]
        train = result["fit_standardized"]
        forecast_time = result["forecast_time_us"]
        for component in range(3):
            axis = axes[row, component]
            axis.plot(
                train_time,
                train[:, component],
                color=colors["truth"],
                linewidth=1.0,
                alpha=0.65,
            )
            for source, label in (
                ("true", "truth"),
                ("persistence", "persistence"),
                ("dmd", "dmd"),
                ("sindy", "sindy"),
            ):
                axis.plot(
                    forecast_time,
                    result[source][:, component],
                    color=colors[label],
                    linewidth=1.25 if source != "persistence" else 1.0,
                    label=label,
                )
            axis.axvline(FORECAST_START_US, color="#555555", linestyle="--")
            axis.set_title(f"{layer}, PC{component + 1}")
            axis.set_ylabel("standardized score")
            axis.grid(alpha=0.2)

        axis = axes[row, 3]
        axis.plot(
            forecast_time,
            moving_average(result["persistence_error"]),
            color=colors["persistence"],
            label="persistence",
        )
        axis.plot(
            forecast_time,
            moving_average(result["dmd_error"]),
            color=colors["dmd"],
            label="DMD",
        )
        axis.plot(
            forecast_time,
            moving_average(result["sindy_error"]),
            color=colors["sindy"],
            label="SINDy",
        )
        axis.axhline(1.0, color="#555555", linestyle=":", label="1 train SD")
        axis.set_title(f"{layer}, state RMSE (21-frame mean)")
        axis.set_ylabel("standardized RMSE")
        axis.grid(alpha=0.2)
        upper = np.nanpercentile(
            np.concatenate(
                [
                    result["persistence_error"],
                    result["dmd_error"],
                    result["sindy_error"],
                ]
            ),
            95,
        )
        if np.isfinite(upper):
            axis.set_ylim(0.0, max(1.25, upper * 1.2))
        axis.legend(loc="upper right", fontsize=8)

    for axis in axes[-1]:
        axis.set_xlabel("time [us]")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Autonomous reduced-order rollout: fit 20-24 us, forecast 24-30 us"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_phase_portraits(path: Path, results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, (layer, result) in zip(axes, results.items()):
        train = result["fit_standardized"]
        axis.plot(
            train[:, 0],
            train[:, 1],
            color="#aaaaaa",
            linewidth=1.0,
            label="fit truth (20-24 us)",
        )
        for source, label, color in (
            ("true", "truth", "black"),
            ("dmd", "DMD", "#1675b8"),
            ("sindy", "SINDy", "#c63f3f"),
        ):
            axis.plot(
                result[source][:, 0],
                result[source][:, 1],
                color=color,
                linewidth=1.2,
                label=f"{label} (24-30 us)",
            )
        axis.scatter(
            result["true"][0, 0],
            result["true"][0, 1],
            color="#e69f00",
            s=35,
            zorder=5,
            label="forecast start",
        )
        axis.set_xlabel("standardized PC1")
        axis.set_ylabel("standardized PC2")
        axis.set_title(layer)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
    fig.suptitle("Reduced latent phase portraits")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dmd_eigenvalues(path: Path, results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    for axis, (layer, result) in zip(axes, results.items()):
        eigenvalues = result["dmd_eigenvalues"]
        axis.plot(np.cos(angle), np.sin(angle), color="#777777", linestyle="--")
        axis.scatter(
            np.real(eigenvalues),
            np.imag(eigenvalues),
            c=np.abs(eigenvalues),
            cmap="viridis",
            edgecolor="black",
            linewidth=0.4,
            s=55,
        )
        limit = max(1.15, float(np.max(np.abs(eigenvalues))) * 1.15)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal")
        axis.axhline(0.0, color="#bbbbbb", linewidth=0.7)
        axis.axvline(0.0, color="#bbbbbb", linewidth=0.7)
        axis.set_title(
            f"{layer}: spectral radius={np.max(np.abs(eigenvalues)):.4f}"
        )
        axis.set_xlabel("real")
        axis.set_ylabel("imaginary")
        axis.grid(alpha=0.2)
    fig.suptitle("DMD eigenvalues (unit circle indicates neutral stability)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_sindy_terms(path: Path, results: dict[str, dict], top: int = 12) -> None:
    lines = [
        "# Discrete-time SINDy maps",
        "",
        "Each equation predicts one standardized PCA state at the next 15 ns frame.",
        "Only the largest coefficients are printed; the CSV/JSON summary records sparsity.",
        "",
    ]
    for layer, result in results.items():
        coefficient = result["sindy_coefficient"]
        names = result["sindy_term_names"]
        lines.extend(
            [
                f"## {layer} ({result['components']} components)",
                "",
                f"Selected STLSQ threshold: `{result['sindy_threshold']:.6g}`",
                "",
            ]
        )
        for output in range(coefficient.shape[1]):
            active = np.flatnonzero(coefficient[:, output])
            order = active[
                np.argsort(np.abs(coefficient[active, output]))[::-1]
            ][:top]
            terms = [
                f"{coefficient[index, output]:+.6g}*{names[index]}"
                for index in order
            ]
            lines.append(f"`z{output + 1}[k+1] = {' '.join(terms)}`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_layer(
    data: LayerData,
    thresholds: list[float],
) -> tuple[dict, dict]:
    fit_mask = (data.time_us >= FIT_START_US) & (
        data.time_us < FORECAST_START_US
    )
    forecast_mask = (data.time_us >= FORECAST_START_US) & (
        data.time_us <= FORECAST_END_US
    )
    fit_scores = data.scores[fit_mask]
    truth_scores = data.scores[forecast_mask]
    fit_time = data.time_us[fit_mask]
    forecast_time = data.time_us[forecast_mask]
    if len(fit_scores) < 10 or len(truth_scores) < 1:
        raise ValueError(f"Insufficient {data.name} samples")

    standardizer = fit_standardizer(fit_scores)
    standardized = standardizer.transform(data.scores)
    fit_standardized = standardized[fit_mask]
    truth = standardized[forecast_mask]
    persistence = np.repeat(fit_standardized[-1][None, :], len(truth), axis=0)

    dmd_matrix, dmd_eigenvalues = fit_dmd(fit_standardized)
    dmd = rollout_linear(dmd_matrix, fit_standardized[-1], len(truth))

    threshold, trials = select_sindy_threshold(
        standardized, data.time_us, fit_mask, thresholds
    )
    sindy_coefficient, sindy_term_names = fit_stlsq(
        fit_standardized[:-1], fit_standardized[1:], threshold
    )
    sindy = rollout_sindy(
        sindy_coefficient, fit_standardized[-1], len(truth)
    )

    dmd_metrics, dmd_error = evaluate_prediction(
        truth, dmd, persistence, forecast_time
    )
    sindy_metrics, sindy_error = evaluate_prediction(
        truth, sindy, persistence, forecast_time
    )
    persistence_error = np.sqrt(np.mean((persistence - truth) ** 2, axis=1))

    metrics = {
        "dmd": {
            "24-30": dmd_metrics,
            "24-27": interval_metrics(
                truth, dmd, persistence, forecast_time, 24.0, 27.0
            ),
            "27-30": interval_metrics(
                truth, dmd, persistence, forecast_time, 27.0, 30.0
            ),
        },
        "sindy": {
            "24-30": sindy_metrics,
            "24-27": interval_metrics(
                truth, sindy, persistence, forecast_time, 24.0, 27.0
            ),
            "27-30": interval_metrics(
                truth, sindy, persistence, forecast_time, 27.0, 30.0
            ),
        },
    }
    summary = {
        "components": data.components,
        "fit_samples": int(len(fit_scores)),
        "forecast_samples": int(len(truth_scores)),
        "fit_interval_us": [float(fit_time[0]), float(fit_time[-1])],
        "forecast_interval_us": [
            float(forecast_time[0]),
            float(forecast_time[-1]),
        ],
        "dmd_spectral_radius": float(np.max(np.abs(dmd_eigenvalues))),
        "dmd_eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in dmd_eigenvalues
        ],
        "sindy_selected_threshold": threshold,
        "sindy_nonzero_coefficients": int(np.count_nonzero(sindy_coefficient)),
        "sindy_total_coefficients": int(sindy_coefficient.size),
        "sindy_validation_trials": trials,
        "metrics": metrics,
    }
    result = {
        "components": data.components,
        "fit_time_us": fit_time,
        "fit_standardized": fit_standardized,
        "forecast_time_us": forecast_time,
        "true": truth,
        "persistence": persistence,
        "dmd": dmd,
        "sindy": sindy,
        "dmd_error": dmd_error,
        "sindy_error": sindy_error,
        "persistence_error": persistence_error,
        "dmd_eigenvalues": dmd_eigenvalues,
        "sindy_coefficient": sindy_coefficient,
        "sindy_term_names": sindy_term_names,
        "sindy_threshold": threshold,
        "standardizer": standardizer,
    }
    return result, summary


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_readme(path: Path, summary: dict) -> None:
    rows = []
    for layer, layer_summary in summary["layers"].items():
        for method in ("dmd", "sindy"):
            metrics = layer_summary["metrics"][method]["24-30"]
            rows.append(
                "| {layer} | {components} | {method} | {rmse:.4f} | "
                "{skill:.4f} | {corr:.4f} | {fraction:.3f} |".format(
                    layer=layer,
                    components=layer_summary["components"],
                    method=method.upper(),
                    rmse=metrics["standardized_rmse"],
                    skill=metrics["skill_vs_persistence"],
                    corr=metrics["flattened_correlation"],
                    fraction=metrics["fraction_frames_better_than_persistence"],
                )
            )

    text = f"""# RadAz latent reduced-order dynamics

The steady PCA coordinates of the frozen SimVP model were used as state
variables. No SimVP retraining was performed.

- Fit interval: `20-24 us` only
- Autonomous forecast: `24-30 us`
- Encoder state: `8 PCs` (about 95% steady pooled-latent variance)
- Translator state: `15 PCs` (about 95% steady pooled-latent variance)
- DMD: linear discrete-time map
- SINDy: sparse quadratic discrete-time map selected using an internal
  `23-24 us` validation interval
- Baseline: persistence of the final fitted latent state

## Holdout rollout results

| Layer | PCs | Method | standardized RMSE | skill vs persistence | correlation | frames better than persistence |
|---|---:|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Positive skill means that the autonomous model improves on persistence.
Negative skill means that retaining the final latent state is more accurate.
The rollout is intentionally strict: after 24 us, neither DMD nor SINDy is
corrected with true latent states.

## Files

- `reduced_dynamics_summary.json`: settings, eigenvalues, sparsity and metrics
- `reduced_dynamics_metrics.csv`: interval-level quantitative comparison
- `reduced_dynamics_rollout.csv`: true and forecast PCA states at every frame
- `reduced_dynamics_rollout_overview.png`: PC trajectories and state error
- `reduced_dynamics_phase_portraits.png`: PC1-PC2 trajectories
- `dmd_eigenvalues.png`: fitted discrete-time DMD spectrum
- `sindy_equations.md`: largest terms of each sparse map equation

## 日本語メモ

学習済みSimVPの潜在特徴に対するPCAスコアを状態変数として使い、
`20-24 us`だけからDMDとSINDyを同定しました。`24-30 us`では真の潜在状態を
途中入力せず、最後の既知状態から完全に自己回帰で予測しています。

この評価でpersistenceより良ければ、潜在空間に短期的な相関だけでなく、
低次元の自律時間発展として利用できる構造が存在する証拠になります。
一方、負のskillや軌道の発散は、95%の分散を少数PCで表せても、それだけでは
未来を閉じた力学系として記述できないことを示します。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--latent", type=Path, default=DEFAULT_LATENT)
    parser.add_argument("--pca-dir", type=Path, default=DEFAULT_PCA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--encoder-components", type=int, default=8)
    parser.add_argument("--translator-components", type=int, default=15)
    parser.add_argument(
        "--sindy-thresholds",
        type=float,
        nargs="+",
        default=[1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    component_map = {
        "encoder": args.encoder_components,
        "translator": args.translator_components,
    }
    layers = read_score_csv(
        args.scores, component_map, args.latent, args.pca_dir
    )
    results: dict[str, dict] = {}
    layer_summaries: dict[str, dict] = {}
    for layer, data in layers.items():
        print(f"[FIT] {layer}: {data.components} components")
        result, summary = analyze_layer(data, args.sindy_thresholds)
        results[layer] = result
        layer_summaries[layer] = summary
        print(
            f"[RESULT] {layer}: "
            f"DMD skill={summary['metrics']['dmd']['24-30']['skill_vs_persistence']:.4f}, "
            f"SINDy skill={summary['metrics']['sindy']['24-30']['skill_vs_persistence']:.4f}"
        )

    summary = {
        "status": "PASS",
        "source_scores": str(args.scores),
        "fit_interval_us": [FIT_START_US, FORECAST_START_US],
        "forecast_interval_us": [FORECAST_START_US, FORECAST_END_US],
        "sindy_internal_validation_interval_us": [
            SINDY_VALIDATION_START_US,
            FORECAST_START_US,
        ],
        "state_standardization": "mean/std from 20-24 us",
        "models": {
            "dmd": "linear discrete-time map",
            "sindy": "quadratic discrete-time map fitted by normalized STLSQ",
            "baseline": "persistence of final fitted PCA state",
        },
        "layers": layer_summaries,
        "notes": [
            "The steady PCA bases were originally fitted only on SimVP training windows.",
            "No 24-30 us state is used for fitting or SINDy threshold selection.",
            "All reported forecasts are autonomous multi-step rollouts.",
            "The PCA scores originate from 8x8 average-pooled latent features.",
        ],
    }
    safe_summary = json_safe(summary)
    (args.output / "reduced_dynamics_summary.json").write_text(
        json.dumps(safe_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_rollout_csv(args.output / "reduced_dynamics_rollout.csv", results)
    write_metrics_csv(args.output / "reduced_dynamics_metrics.csv", safe_summary)
    write_sindy_terms(args.output / "sindy_equations.md", results)
    plot_rollout_overview(
        args.output / "reduced_dynamics_rollout_overview.png", results
    )
    plot_phase_portraits(
        args.output / "reduced_dynamics_phase_portraits.png", results
    )
    plot_dmd_eigenvalues(args.output / "dmd_eigenvalues.png", results)
    write_readme(args.output / "README.md", summary)
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
