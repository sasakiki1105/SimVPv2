"""Evaluate delay-embedded DMD and HAVOK on RadAz SimVP latent states."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez10kvm_hankel_havok"
)
DEFAULT_PHYSICAL = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez10kvm_latent"
    / "physical_metrics_by_frame.csv"
)

FIT_START_US = 20.0
VALIDATION_START_US = 23.0
FORECAST_START_US = 24.0
FORECAST_END_US = 30.0

PHYSICAL_METRICS = (
    "phi_a_mtsi",
    "phi_a_ecdi",
    "electron_den_a_mtsi",
    "electron_den_a_ecdi",
    "electron_den_std",
    "transport_total",
    "transport_mtsi",
    "transport_ecdi",
)


@dataclass
class HankelModel:
    delay: int
    rank: int
    state_dimensions: int
    delay_mean: np.ndarray
    basis: np.ndarray
    matrix: np.ndarray
    eigenvalues: np.ndarray
    singular_values: np.ndarray

    def project(self, delay_vectors: np.ndarray) -> np.ndarray:
        return (delay_vectors - self.delay_mean) @ self.basis

    def reconstruct(self, coordinates: np.ndarray) -> np.ndarray:
        return coordinates @ self.basis.T + self.delay_mean


@dataclass
class HavokModel:
    matrix: np.ndarray
    forcing_vector: np.ndarray
    forcing_mean: float
    forcing_scale: float


def make_delay_vectors(states: np.ndarray, delay: int) -> np.ndarray:
    if delay < 1 or len(states) < delay:
        raise ValueError(f"Invalid delay={delay} for {len(states)} states")
    return np.asarray(
        [
            np.concatenate(
                [states[index - lag] for lag in range(delay)], axis=0
            )
            for index in range(delay - 1, len(states))
        ],
        dtype=np.float64,
    )


def fit_hankel_dmd(
    states: np.ndarray,
    delay: int,
    rank: int,
) -> HankelModel:
    delay_vectors = make_delay_vectors(states, delay)
    delay_mean = np.mean(delay_vectors, axis=0)
    centered = delay_vectors - delay_mean
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    maximum_rank = min(len(delay_vectors) - 1, right.shape[0])
    if rank > maximum_rank:
        raise ValueError(
            f"rank={rank} exceeds maximum {maximum_rank} for delay={delay}"
        )
    basis = right[:rank].T
    coordinates = centered @ basis
    x = coordinates[:-1].T
    y = coordinates[1:].T
    matrix = y @ np.linalg.pinv(x, rcond=1.0e-10)
    return HankelModel(
        delay=delay,
        rank=rank,
        state_dimensions=states.shape[1],
        delay_mean=delay_mean,
        basis=basis,
        matrix=matrix,
        eigenvalues=np.linalg.eigvals(matrix),
        singular_values=singular_values,
    )


def rollout_hankel(
    model: HankelModel,
    history: np.ndarray,
    steps: int,
) -> np.ndarray:
    initial_delay = make_delay_vectors(history[-model.delay :], model.delay)[0]
    coordinate = model.project(initial_delay[None, :])[0]
    forecast = np.empty(
        (steps, model.state_dimensions), dtype=np.float64
    )
    for index in range(steps):
        coordinate = model.matrix @ coordinate
        if (
            not np.all(np.isfinite(coordinate))
            or np.max(np.abs(coordinate)) > 1.0e8
        ):
            forecast[index:] = np.nan
            break
        reconstructed = model.reconstruct(coordinate[None, :])[0]
        forecast[index] = reconstructed[: model.state_dimensions]
    return forecast


def candidate_search(
    standardized: np.ndarray,
    time_us: np.ndarray,
    fit_mask: np.ndarray,
    delays: list[int],
    ranks: list[int],
) -> tuple[dict, list[dict]]:
    subtrain_mask = fit_mask & (time_us < VALIDATION_START_US)
    validation_mask = fit_mask & (time_us >= VALIDATION_START_US)
    subtrain = standardized[subtrain_mask]
    validation = standardized[validation_mask]
    persistence = np.repeat(subtrain[-1][None, :], len(validation), axis=0)

    trials: list[dict] = []
    for delay in delays:
        for rank in ranks:
            try:
                model = fit_hankel_dmd(subtrain, delay, rank)
                forecast = rollout_hankel(
                    model, subtrain, len(validation)
                )
                metrics, _ = reduced.evaluate_prediction(
                    validation,
                    forecast,
                    persistence,
                    time_us[validation_mask],
                )
                mse = metrics["standardized_mse"]
                radius = float(np.max(np.abs(model.eigenvalues)))
            except (ValueError, np.linalg.LinAlgError):
                mse = float("inf")
                radius = float("nan")
                metrics = {}
            trials.append(
                {
                    "delay": delay,
                    "history_us": delay * 0.015,
                    "rank": rank,
                    "validation_mse": mse,
                    "validation_skill_vs_persistence": metrics.get(
                        "skill_vs_persistence", float("-inf")
                    ),
                    "validation_correlation": metrics.get(
                        "flattened_correlation", float("nan")
                    ),
                    "spectral_radius": radius,
                }
            )
    finite = [
        row for row in trials if np.isfinite(row["validation_mse"])
    ]
    if not finite:
        raise RuntimeError("Every Hankel DMD candidate failed")
    best = min(
        finite,
        key=lambda row: (
            row["validation_mse"],
            row["delay"],
            row["rank"],
        ),
    )
    return best, trials


def fit_havok(
    model: HankelModel,
    fit_states: np.ndarray,
) -> tuple[HavokModel, np.ndarray]:
    delay_vectors = make_delay_vectors(fit_states, model.delay)
    coordinates = model.project(delay_vectors)
    resolved = coordinates[:, :-1]
    forcing = coordinates[:, -1]
    features = np.column_stack([resolved[:-1], forcing[:-1]])
    coefficient = np.linalg.lstsq(
        features, resolved[1:], rcond=1.0e-10
    )[0]
    matrix = coefficient[:-1].T
    forcing_vector = coefficient[-1].copy()
    forcing_mean = float(np.mean(forcing))
    forcing_scale = float(np.std(forcing, ddof=1))
    if forcing_scale < 1.0e-12:
        forcing_scale = 1.0
    return (
        HavokModel(
            matrix=matrix,
            forcing_vector=forcing_vector,
            forcing_mean=forcing_mean,
            forcing_scale=forcing_scale,
        ),
        coordinates,
    )


def rollout_havok_zero_forcing(
    hankel: HankelModel,
    havok: HavokModel,
    fit_states: np.ndarray,
    steps: int,
) -> np.ndarray:
    initial_delay = make_delay_vectors(
        fit_states[-hankel.delay :], hankel.delay
    )[0]
    coordinate = hankel.project(initial_delay[None, :])[0]
    resolved = coordinate[:-1]
    forecast = np.empty(
        (steps, hankel.state_dimensions), dtype=np.float64
    )
    for index in range(steps):
        resolved = havok.matrix @ resolved
        if (
            not np.all(np.isfinite(resolved))
            or np.max(np.abs(resolved)) > 1.0e8
        ):
            forecast[index:] = np.nan
            break
        full_coordinate = np.concatenate([resolved, np.zeros(1)])
        reconstructed = hankel.reconstruct(
            full_coordinate[None, :]
        )[0]
        forecast[index] = reconstructed[: hankel.state_dimensions]
    return forecast


def read_physical_metrics(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        "time_us": np.asarray(
            [float(row["time_us"]) for row in rows], dtype=np.float64
        )
    }
    for metric in PHYSICAL_METRICS:
        result[metric] = np.asarray(
            [float(row[metric]) for row in rows], dtype=np.float64
        )
    return result


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    if (
        np.std(left[finite]) < 1.0e-12
        or np.std(right[finite]) < 1.0e-12
    ):
        return float("nan")
    return float(spearmanr(left[finite], right[finite]).statistic)


def forcing_diagnostics(
    layer: reduced.LayerData,
    standardized: np.ndarray,
    hankel: HankelModel,
    havok: HavokModel,
    physical: dict[str, np.ndarray],
) -> tuple[dict, list[dict], dict]:
    delay_vectors = make_delay_vectors(standardized, hankel.delay)
    delay_time = layer.time_us[hankel.delay - 1 :]
    coordinates = hankel.project(delay_vectors)
    resolved = coordinates[:, :-1]
    forcing = coordinates[:, -1]
    forcing_z = (forcing - havok.forcing_mean) / havok.forcing_scale

    fit = (delay_time >= FIT_START_US) & (
        delay_time < FORECAST_START_US
    )
    holdout = (delay_time >= FORECAST_START_US) & (
        delay_time <= FORECAST_END_US
    )
    transition = (
        (delay_time[:-1] >= FORECAST_START_US)
        & (delay_time[1:] <= FORECAST_END_US)
    )
    true_next = resolved[1:][transition]
    current = resolved[:-1][transition]
    current_forcing = forcing[:-1][transition]
    without_forcing = current @ havok.matrix.T
    with_forcing = without_forcing + np.outer(
        current_forcing, havok.forcing_vector
    )
    mse_without = float(np.mean((without_forcing - true_next) ** 2))
    mse_with = float(np.mean((with_forcing - true_next) ** 2))

    threshold = float(np.quantile(np.abs(forcing_z[fit]), 0.95))
    holdout_events = np.flatnonzero(
        holdout & (np.abs(forcing_z) >= threshold)
    )
    event_rows = [
        {
            "time_us": float(delay_time[index]),
            "forcing_z": float(forcing_z[index]),
        }
        for index in holdout_events
    ]

    correlations: list[dict] = []
    interpolated: dict[str, np.ndarray] = {}
    for metric in PHYSICAL_METRICS:
        values = np.interp(
            delay_time, physical["time_us"], physical[metric]
        )
        interpolated[metric] = values
        delta = np.empty_like(values)
        delta[0] = 0.0
        delta[1:] = np.diff(values)
        for scope, mask in (("fit", fit), ("holdout", holdout)):
            correlations.append(
                {
                    "metric": metric,
                    "scope": scope,
                    "spearman_abs_forcing_vs_value": safe_spearman(
                        np.abs(forcing_z[mask]), values[mask]
                    ),
                    "spearman_abs_forcing_vs_abs_delta": safe_spearman(
                        np.abs(forcing_z[mask]), np.abs(delta[mask])
                    ),
                    "samples": int(np.count_nonzero(mask)),
                }
            )

    summary = {
        "forcing_training_abs_quantile_95": threshold,
        "holdout_forcing_events": len(event_rows),
        "holdout_forcing_event_fraction": float(
            len(event_rows) / np.count_nonzero(holdout)
        ),
        "holdout_one_step_resolved_mse_without_forcing": mse_without,
        "holdout_one_step_resolved_mse_with_true_forcing": mse_with,
        "holdout_one_step_forcing_error_reduction": float(
            1.0 - mse_with / mse_without
        ),
        "largest_holdout_events": sorted(
            event_rows, key=lambda row: abs(row["forcing_z"]), reverse=True
        )[:20],
    }
    series = {
        "time_us": delay_time,
        "forcing_z": forcing_z,
        "fit_mask": fit,
        "holdout_mask": holdout,
        "threshold": threshold,
        "physical": interpolated,
    }
    return summary, correlations, series


def evaluate_method_intervals(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    time_us: np.ndarray,
) -> tuple[dict, np.ndarray]:
    complete, error = reduced.evaluate_prediction(
        truth, prediction, persistence, time_us
    )
    intervals = {
        "24-30": complete,
        "24-27": reduced.interval_metrics(
                truth, prediction, persistence, time_us, 24.0, 27.0
            ),
        "27-30": reduced.interval_metrics(
                truth, prediction, persistence, time_us, 27.0, 30.0
            ),
    }
    masks = {
        "24-30": np.ones(len(time_us), dtype=bool),
        "24-27": (time_us >= 24.0) & (time_us < 27.0),
        "27-30": (time_us >= 27.0) & (time_us <= 30.0),
    }
    for interval, mask in masks.items():
        subset_truth = truth[mask]
        subset_prediction = prediction[mask]
        finite = np.isfinite(subset_prediction).all(axis=1)
        climatology_error = np.sqrt(
            np.mean(subset_truth * subset_truth, axis=1)
        )
        if np.all(finite):
            prediction_mse = float(
                np.mean((subset_prediction - subset_truth) ** 2)
            )
        else:
            prediction_mse = float("inf")
        climatology_mse = float(np.mean(subset_truth**2))
        intervals[interval].update(
            {
                "climatology_mse": climatology_mse,
                "skill_vs_training_mean": float(
                    1.0 - prediction_mse / climatology_mse
                )
                if np.isfinite(prediction_mse)
                else float("-inf"),
                "fraction_frames_better_than_training_mean": float(
                    np.mean(
                        error[mask]
                        < climatology_error
                    )
                ),
            }
        )
    return intervals, error


def analyze_layer(
    layer: reduced.LayerData,
    physical: dict[str, np.ndarray],
    delays: list[int],
    ranks: list[int],
) -> tuple[dict, dict, list[dict], list[dict]]:
    fit_mask = (layer.time_us >= FIT_START_US) & (
        layer.time_us < FORECAST_START_US
    )
    forecast_mask = (layer.time_us >= FORECAST_START_US) & (
        layer.time_us <= FORECAST_END_US
    )
    standardizer = reduced.fit_standardizer(layer.scores[fit_mask])
    standardized = standardizer.transform(layer.scores)
    fit_states = standardized[fit_mask]
    truth = standardized[forecast_mask]
    fit_time = layer.time_us[fit_mask]
    forecast_time = layer.time_us[forecast_mask]
    persistence = np.repeat(fit_states[-1][None, :], len(truth), axis=0)
    climatology = np.zeros_like(truth)

    selected, candidates = candidate_search(
        standardized, layer.time_us, fit_mask, delays, ranks
    )
    hankel = fit_hankel_dmd(
        fit_states, selected["delay"], selected["rank"]
    )
    hankel_prediction = rollout_hankel(
        hankel, fit_states, len(truth)
    )

    standard_matrix, standard_eigenvalues = reduced.fit_dmd(fit_states)
    standard_prediction = reduced.rollout_linear(
        standard_matrix, fit_states[-1], len(truth)
    )

    havok, _ = fit_havok(hankel, fit_states)
    havok_zero = rollout_havok_zero_forcing(
        hankel, havok, fit_states, len(truth)
    )

    metrics: dict[str, dict] = {}
    errors: dict[str, np.ndarray] = {}
    for method, prediction in (
        ("standard_dmd", standard_prediction),
        ("hankel_dmd", hankel_prediction),
        ("havok_zero_forcing", havok_zero),
    ):
        metrics[method], errors[method] = evaluate_method_intervals(
            truth, prediction, persistence, forecast_time
        )

    havok_summary, correlations, forcing_series = forcing_diagnostics(
        layer, standardized, hankel, havok, physical
    )
    summary = {
        "components": layer.components,
        "fit_samples": int(len(fit_states)),
        "forecast_samples": int(len(truth)),
        "selected_delay": hankel.delay,
        "selected_history_us": hankel.delay * 0.015,
        "selected_rank": hankel.rank,
        "validation_metrics": selected,
        "hankel_spectral_radius": float(
            np.max(np.abs(hankel.eigenvalues))
        ),
        "standard_dmd_spectral_radius": float(
            np.max(np.abs(standard_eigenvalues))
        ),
        "hankel_eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in hankel.eigenvalues
        ],
        "singular_value_energy_at_selected_rank": float(
            np.sum(hankel.singular_values[: hankel.rank] ** 2)
            / np.sum(hankel.singular_values**2)
        ),
        "metrics": metrics,
        "havok": havok_summary,
    }
    result = {
        "components": layer.components,
        "fit_time_us": fit_time,
        "fit_states": fit_states,
        "forecast_time_us": forecast_time,
        "truth": truth,
        "persistence": persistence,
        "climatology": climatology,
        "standard_dmd": standard_prediction,
        "hankel_dmd": hankel_prediction,
        "havok_zero_forcing": havok_zero,
        "errors": errors,
        "persistence_error": np.sqrt(
            np.mean((persistence - truth) ** 2, axis=1)
        ),
        "climatology_error": np.sqrt(
            np.mean((climatology - truth) ** 2, axis=1)
        ),
        "hankel_eigenvalues": hankel.eigenvalues,
        "forcing": forcing_series,
    }
    return result, summary, candidates, correlations


def write_candidate_csv(path: Path, candidate_rows: list[dict]) -> None:
    fields = [
        "layer",
        "delay",
        "history_us",
        "rank",
        "validation_mse",
        "validation_skill_vs_persistence",
        "validation_correlation",
        "spectral_radius",
        "selected",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)


def write_metric_csv(path: Path, summary: dict) -> None:
    fields = [
        "layer",
        "components",
        "delay",
        "rank",
        "method",
        "interval_us",
        "standardized_rmse",
        "skill_vs_persistence",
        "skill_vs_training_mean",
        "flattened_correlation",
        "finite_fraction",
        "fraction_frames_better_than_persistence",
        "fraction_frames_better_than_training_mean",
        "smoothed_better_than_persistence_horizon_us",
        "smoothed_one_sigma_horizon_us",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer, layer_summary in summary["layers"].items():
            for method, intervals in layer_summary["metrics"].items():
                for interval, metrics in intervals.items():
                    writer.writerow(
                        {
                            "layer": layer,
                            "components": layer_summary["components"],
                            "delay": layer_summary["selected_delay"],
                            "rank": layer_summary["selected_rank"],
                            "method": method,
                            "interval_us": interval,
                            **{
                                field: metrics.get(field, "")
                                for field in fields[6:]
                            },
                        }
                    )


def write_rollout_csv(path: Path, results: dict[str, dict]) -> None:
    maximum = max(result["components"] for result in results.values())
    methods = (
        "truth",
        "persistence",
        "climatology",
        "standard_dmd",
        "hankel_dmd",
        "havok_zero_forcing",
    )
    fields = ["layer", "time_us"]
    for method in methods:
        fields.extend(
            f"{method}_pc{component + 1}" for component in range(maximum)
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer, result in results.items():
            for index, time_us in enumerate(result["forecast_time_us"]):
                row: dict[str, object] = {
                    "layer": layer,
                    "time_us": time_us,
                }
                for method in methods:
                    values = result[method][index]
                    for component, value in enumerate(values):
                        row[f"{method}_pc{component + 1}"] = value
                writer.writerow(row)


def write_forcing_csv(
    path: Path, correlation_rows: list[dict]
) -> None:
    fields = [
        "layer",
        "metric",
        "scope",
        "spearman_abs_forcing_vs_value",
        "spearman_abs_forcing_vs_abs_delta",
        "samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(correlation_rows)


def plot_rollout(path: Path, results: dict[str, dict]) -> None:
    colors = {
        "truth": "black",
        "persistence": "#8a8a8a",
        "climatology": "#009e73",
        "standard_dmd": "#4d9bc4",
        "hankel_dmd": "#0068b5",
        "havok_zero_forcing": "#c34a36",
    }
    labels = {
        "truth": "truth",
        "persistence": "persistence",
        "climatology": "training mean",
        "standard_dmd": "standard DMD",
        "hankel_dmd": "Hankel DMD",
        "havok_zero_forcing": "HAVOK zero forcing",
    }
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
    for row, (layer, result) in enumerate(results.items()):
        for component in range(3):
            axis = axes[row, component]
            axis.plot(
                result["fit_time_us"],
                result["fit_states"][:, component],
                color="#aaaaaa",
                linewidth=1.0,
            )
            for method in labels:
                axis.plot(
                    result["forecast_time_us"],
                    result[method][:, component],
                    color=colors[method],
                    linewidth=1.2,
                    label=labels[method],
                )
            axis.axvline(
                FORECAST_START_US, color="#555555", linestyle="--"
            )
            axis.set_title(f"{layer}, PC{component + 1}")
            axis.set_ylabel("standardized score")
            axis.grid(alpha=0.2)

        axis = axes[row, 3]
        axis.plot(
            result["forecast_time_us"],
            reduced.moving_average(result["persistence_error"]),
            color=colors["persistence"],
            label=labels["persistence"],
        )
        axis.plot(
            result["forecast_time_us"],
            reduced.moving_average(result["climatology_error"]),
            color=colors["climatology"],
            linestyle="--",
            label=labels["climatology"],
        )
        for method in (
            "standard_dmd",
            "hankel_dmd",
            "havok_zero_forcing",
        ):
            axis.plot(
                result["forecast_time_us"],
                reduced.moving_average(result["errors"][method]),
                color=colors[method],
                label=labels[method],
            )
        axis.axhline(1.0, color="#555555", linestyle=":")
        axis.set_title(f"{layer}, state RMSE (21-frame mean)")
        axis.set_ylabel("standardized RMSE")
        axis.grid(alpha=0.2)
        candidates = [
            result["persistence_error"],
            result["climatology_error"],
        ]
        candidates.extend(result["errors"].values())
        upper = np.nanpercentile(np.concatenate(candidates), 95)
        if np.isfinite(upper):
            axis.set_ylim(0.0, max(1.25, upper * 1.2))
        axis.legend(loc="upper right", fontsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("time [us]")
    fig.suptitle(
        "Delay-embedded autonomous rollout: fit 20-24 us, forecast 24-30 us"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_phase_portrait(path: Path, results: dict[str, dict]) -> None:
    colors = {
        "truth": "black",
        "standard_dmd": "#4d9bc4",
        "hankel_dmd": "#0068b5",
        "havok_zero_forcing": "#c34a36",
    }
    labels = {
        "truth": "truth",
        "standard_dmd": "standard DMD",
        "hankel_dmd": "Hankel DMD",
        "havok_zero_forcing": "HAVOK zero forcing",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, (layer, result) in zip(axes, results.items()):
        axis.plot(
            result["fit_states"][:, 0],
            result["fit_states"][:, 1],
            color="#aaaaaa",
            label="fit truth",
        )
        for method in labels:
            axis.plot(
                result[method][:, 0],
                result[method][:, 1],
                color=colors[method],
                linewidth=1.2,
                label=labels[method],
            )
        axis.set_xlabel("standardized PC1")
        axis.set_ylabel("standardized PC2")
        axis.set_title(layer)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
    fig.suptitle("Hankel DMD and HAVOK latent phase portraits")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_eigenvalues(path: Path, results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    for axis, (layer, result) in zip(axes, results.items()):
        eigenvalues = result["hankel_eigenvalues"]
        axis.plot(
            np.cos(angle),
            np.sin(angle),
            color="#777777",
            linestyle="--",
        )
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
            f"{layer}: radius={np.max(np.abs(eigenvalues)):.4f}"
        )
        axis.set_xlabel("real")
        axis.set_ylabel("imaginary")
        axis.grid(alpha=0.2)
    fig.suptitle("Selected Hankel DMD eigenvalues")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def standardized_series(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mean = np.mean(values[mask])
    scale = np.std(values[mask], ddof=1)
    if scale < 1.0e-12:
        scale = 1.0
    return (values - mean) / scale


def plot_forcing(
    path: Path,
    results: dict[str, dict],
    correlation_rows: list[dict],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    for row, (layer, result) in enumerate(results.items()):
        forcing = result["forcing"]
        time_us = forcing["time_us"]
        interval = (time_us >= FIT_START_US) & (
            time_us <= FORECAST_END_US
        )
        axis = axes[row, 0]
        axis.plot(
            time_us[interval],
            forcing["forcing_z"][interval],
            color="#6a3d9a",
            linewidth=1.0,
        )
        axis.axhline(
            forcing["threshold"], color="#777777", linestyle=":"
        )
        axis.axhline(
            -forcing["threshold"], color="#777777", linestyle=":"
        )
        axis.axvline(
            FORECAST_START_US, color="#555555", linestyle="--"
        )
        axis.set_ylabel("normalized HAVOK forcing")
        axis.set_title(f"{layer}: forcing coordinate")
        axis.grid(alpha=0.2)

        layer_rows = [
            item
            for item in correlation_rows
            if item["layer"] == layer and item["scope"] == "holdout"
        ]
        ranked = sorted(
            layer_rows,
            key=lambda item: abs(
                item["spearman_abs_forcing_vs_abs_delta"]
            ),
            reverse=True,
        )
        chosen = [item["metric"] for item in ranked[:2]]
        axis = axes[row, 1]
        for metric, color in zip(chosen, ("#0072b2", "#d55e00")):
            values = forcing["physical"][metric]
            scaled = standardized_series(values, forcing["fit_mask"])
            axis.plot(
                time_us[interval],
                scaled[interval],
                color=color,
                label=metric,
            )
        axis.plot(
            time_us[interval],
            np.abs(forcing["forcing_z"][interval]),
            color="#6a3d9a",
            alpha=0.65,
            label="abs(HAVOK forcing)",
        )
        axis.axvline(
            FORECAST_START_US, color="#555555", linestyle="--"
        )
        axis.set_ylabel("training-standardized value")
        axis.set_title(f"{layer}: strongest holdout delta associations")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time [us]")
    fig.suptitle(
        "HAVOK forcing diagnostics (forcing after 24 us is diagnostic, not forecast)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(path: Path, summary: dict) -> None:
    rows = []
    for layer, layer_summary in summary["layers"].items():
        for method in (
            "standard_dmd",
            "hankel_dmd",
            "havok_zero_forcing",
        ):
            metrics = layer_summary["metrics"][method]["24-30"]
            rows.append(
                "| {layer} | {method} | {delay} | {rank} | {rmse:.4f} | "
                "{skill:.4f} | {mean_skill:.4f} | {corr:.4f} |".format(
                    layer=layer,
                    method=method,
                    delay=layer_summary["selected_delay"],
                    rank=layer_summary["selected_rank"],
                    rmse=metrics["standardized_rmse"],
                    skill=metrics["skill_vs_persistence"],
                    mean_skill=metrics["skill_vs_training_mean"],
                    corr=metrics["flattened_correlation"],
                )
            )
    forcing_rows = []
    for layer, layer_summary in summary["layers"].items():
        havok = layer_summary["havok"]
        forcing_rows.append(
            "| {layer} | {events} | {gain:.4f} |".format(
                layer=layer,
                events=havok["holdout_forcing_events"],
                gain=havok["holdout_one_step_forcing_error_reduction"],
            )
        )
    text = f"""# RadAz Hankel DMD and HAVOK analysis

No SimVP retraining was performed. The steady PCA states from the frozen
SimVP model were used.

- Candidate delays: `10, 20, 40, 80` frames
- Candidate Hankel ranks: `8, 15, 20, 30`
- Hyperparameter selection: autonomous rollout over `23-24 us`
- Final fit: `20-24 us`
- Strict holdout rollout: `24-30 us`

## Autonomous rollout

| Layer | Method | delay | rank | standardized RMSE | skill vs persistence | skill vs training mean | correlation |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## HAVOK forcing diagnostic

| Layer | holdout forcing events | one-step error reduction with observed forcing |
|---|---:|---:|
{chr(10).join(forcing_rows)}

`havok_zero_forcing` is a strict autonomous forecast. The error reduction
reported for observed forcing is diagnostic only because future forcing is not
available during a real forecast.

## Interpretation

Hankel DMD improves on standard DMD only when explicit temporal history makes
the PCA state closer to a closed Markov state. HAVOK tests a different idea:
whether the main delay coordinates follow approximately linear dynamics while
intermittent events enter through a low-energy forcing coordinate.

Skill against persistence can improve merely by returning toward the training
mean. Therefore skill against the training-mean climatology and trajectory
correlation must also be checked before claiming that phase dynamics were
predicted.

## Files

- `hankel_havok_summary.json`
- `hankel_candidate_metrics.csv`
- `hankel_havok_metrics.csv`
- `hankel_havok_rollout.csv`
- `havok_forcing_correlations.csv`
- `hankel_havok_rollout_overview.png`
- `hankel_havok_phase_portraits.png`
- `hankel_dmd_eigenvalues.png`
- `havok_forcing_diagnostics.png`

## 日本語メモ

通常DMDは現在のPCA状態だけから次時刻を予測します。Hankel DMDは過去の
PCA状態を明示的に連結し、位相・進行方向・成長減衰の履歴を含めます。
HAVOKでは遅延座標の最後の低エネルギー成分をforcingとして分離しました。

予測時に将来forcingを真値から与えることはできないため、厳密な予測性能は
zero-forcing自律予測で評価しています。観測forcingを使った1-step誤差改善と
物理量との相関は、forcingの解釈可能性を調べる診断結果です。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=reduced.DEFAULT_INPUT)
    parser.add_argument("--latent", type=Path, default=reduced.DEFAULT_LATENT)
    parser.add_argument(
        "--pca-dir", type=Path, default=reduced.DEFAULT_PCA_DIR
    )
    parser.add_argument(
        "--physical", type=Path, default=DEFAULT_PHYSICAL
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--delays", type=int, nargs="+", default=[10, 20, 40, 80]
    )
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[8, 15, 20, 30]
    )
    parser.add_argument("--encoder-components", type=int, default=8)
    parser.add_argument("--translator-components", type=int, default=15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    component_map = {
        "encoder": args.encoder_components,
        "translator": args.translator_components,
    }
    layers = reduced.read_score_csv(
        args.scores, component_map, args.latent, args.pca_dir
    )
    physical = read_physical_metrics(args.physical)

    results: dict[str, dict] = {}
    layer_summaries: dict[str, dict] = {}
    candidate_rows: list[dict] = []
    correlation_rows: list[dict] = []
    for layer_name, layer in layers.items():
        print(f"[SEARCH] {layer_name}")
        result, layer_summary, candidates, correlations = analyze_layer(
            layer, physical, args.delays, args.ranks
        )
        results[layer_name] = result
        layer_summaries[layer_name] = layer_summary
        selected = (
            layer_summary["selected_delay"],
            layer_summary["selected_rank"],
        )
        for row in candidates:
            candidate_rows.append(
                {
                    "layer": layer_name,
                    **row,
                    "selected": (
                        row["delay"],
                        row["rank"],
                    )
                    == selected,
                }
            )
        for row in correlations:
            correlation_rows.append({"layer": layer_name, **row})
        metrics = layer_summary["metrics"]["hankel_dmd"]["24-30"]
        print(
            f"[RESULT] {layer_name}: q={selected[0]}, rank={selected[1]}, "
            f"Hankel skill={metrics['skill_vs_persistence']:.4f}"
        )

    summary = {
        "status": "PASS",
        "source_scores": str(args.scores),
        "source_physical_metrics": str(args.physical),
        "candidate_delays": args.delays,
        "candidate_ranks": args.ranks,
        "components": component_map,
        "selection_interval_us": [VALIDATION_START_US, FORECAST_START_US],
        "fit_interval_us": [FIT_START_US, FORECAST_START_US],
        "forecast_interval_us": [FORECAST_START_US, FORECAST_END_US],
        "layers": layer_summaries,
        "notes": [
            "No 24-30 us state was used for model or hyperparameter fitting.",
            "Hankel DMD and HAVOK zero-forcing results are autonomous rollouts.",
            "HAVOK observed-forcing one-step metrics are diagnostic, not forecasts.",
            "PCA states originate from 8x8 average-pooled SimVP latent features.",
        ],
    }
    safe_summary = reduced.json_safe(summary)
    (args.output / "hankel_havok_summary.json").write_text(
        json.dumps(safe_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_candidate_csv(
        args.output / "hankel_candidate_metrics.csv", candidate_rows
    )
    write_metric_csv(
        args.output / "hankel_havok_metrics.csv", safe_summary
    )
    write_rollout_csv(
        args.output / "hankel_havok_rollout.csv", results
    )
    write_forcing_csv(
        args.output / "havok_forcing_correlations.csv",
        correlation_rows,
    )
    plot_rollout(
        args.output / "hankel_havok_rollout_overview.png", results
    )
    plot_phase_portrait(
        args.output / "hankel_havok_phase_portraits.png", results
    )
    plot_eigenvalues(
        args.output / "hankel_dmd_eigenvalues.png", results
    )
    plot_forcing(
        args.output / "havok_forcing_diagnostics.png",
        results,
        correlation_rows,
    )
    write_readme(args.output / "README.md", summary)
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
