#!/usr/bin/env python3
"""Test temporal-expert switching ROMs on B25 with B15 as a control.

The gate is autonomous: it uses only the predicted state's distance to each
expert's fitted delay manifold.  An oracle gate that sees the future truth is
reported only as a diagnostic upper bound.  The first B25 protocol is used to
select the smooth-gate temperature; that temperature is then frozen for the
later B25 protocol and for B15.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_b25_state_augmentation_ablation as ablation
import analyze_radaz_hankel_havok as hankel


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
MAGNETIC_ROOT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_magnetic_sweep_rom_B10_B15_B20_B25_B30mT_E10kVm"
)
B25_ROLLING = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b25_rolling_window_rom_0to40us"
)
OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b25_temporal_switching_rom"
)

DELAYS = (10, 20, 40, 80)
RANKS = (8, 15, 20, 30, 40)
TEMPERATURES = (0.25, 0.5, 1.0, 2.0, 5.0)
MODE_BANDS = {
    "long": (1, 2),
    "mtsi": (3, 6),
    "ecdi": (9, 30),
}


@dataclass(frozen=True)
class Protocol:
    case: str
    name: str
    start_us: float
    split_us: float
    fit_end_us: float
    test_end_us: float
    development: bool = False


PROTOCOLS = (
    Protocol("B25", "B25_12-18_18-24_to_30", 12.0, 18.0, 24.0, 30.0, True),
    Protocol("B25", "B25_18-24_24-30_to_36", 18.0, 24.0, 30.0, 36.0),
    Protocol("B15", "B15_12-18_18-24_to_30", 12.0, 18.0, 24.0, 30.0),
)


@dataclass
class CaseData:
    name: str
    time_us: np.ndarray
    latent_raw: np.ndarray
    physical_ap_raw: np.ndarray


@dataclass
class Representation:
    time_us: np.ndarray
    latent: np.ndarray
    lap: np.ndarray
    latent_dimensions: int
    latent_pcs95: int
    latent_capture: float


@dataclass
class Expert:
    label: str
    model: hankel.HankelModel
    residual_scale: float
    coordinate_variance: np.ndarray


def interval_mask(
    time_us: np.ndarray, start_us: float, end_us: float, *, include_end: bool = False
) -> np.ndarray:
    upper = time_us <= end_us + 1.0e-9 if include_end else time_us < end_us - 1.0e-9
    return (time_us >= start_us - 1.0e-9) & upper


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latent_path(case: str) -> Path:
    if case == "B25":
        return B25_ROLLING / "B25_latent_0to40us.h5"
    return MAGNETIC_ROOT / "latent_features" / case / "radaz_latent_features.h5"


def fourier_path(case: str) -> Path:
    if case == "B25":
        return B25_ROLLING / "B25_physical_fourier_0to40us.h5"
    return MAGNETIC_ROOT / "physical_features" / f"{case}_physical_fourier.h5"


def unpack_fourier(path: Path) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    with h5py.File(path, "r") as source:
        time_us = np.asarray(source["time_us"], dtype=np.float64)
        channels = tuple(item.decode("utf-8") for item in source["channels"][:])
        raw = np.asarray(source["features"], dtype=np.float64)
        bands = len(source["radial_band_edges"]) - 1
    packed = raw.reshape(len(raw), len(channels), bands, -1)
    max_mode = (packed.shape[-1] - 1) // 2
    coefficient = np.zeros(
        (len(raw), len(channels), bands, max_mode + 1), dtype=np.complex128
    )
    coefficient[..., 0] = packed[..., 0]
    coefficient[..., 1:] = (
        packed[..., 1 : max_mode + 1]
        + 1j * packed[..., max_mode + 1 : 2 * max_mode + 1]
    )
    return time_us, channels, coefficient


def physical_ap(path: Path) -> tuple[np.ndarray, np.ndarray]:
    time_us, channels, coefficient = unpack_fourier(path)
    electron = coefficient[:, channels.index("electron_den")]
    phi = coefficient[:, channels.index("phi")]
    mode = np.arange(phi.shape[-1], dtype=np.float64)
    ey = -1j * mode[None, None, :] * phi
    amplitudes = []
    phases = []
    for lower, upper in MODE_BANDS.values():
        upper = min(upper, phi.shape[-1] - 1)
        selected = slice(lower, upper + 1)
        amplitudes.append(
            np.sqrt(np.mean(np.abs(ey[..., selected]) ** 2, axis=(1, 2)))
        )
        cross = np.sum(
            electron[..., selected] * np.conj(ey[..., selected]), axis=(1, 2)
        )
        phases.append(np.angle(cross))
    amplitude = np.log(np.maximum(np.stack(amplitudes, axis=1), 1.0e-30))
    phase = np.stack(phases, axis=1)
    values = np.concatenate((amplitude, np.cos(phase), np.sin(phase)), axis=1)
    return time_us, values


def interpolate(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(target_time, source_time, values[:, column]) for column in range(values.shape[1])],
        axis=1,
    )


def load_case(case: str) -> CaseData:
    with h5py.File(latent_path(case), "r") as source:
        time_us = np.asarray(source["encoder_time_s"], dtype=np.float64) * 1.0e6
        latent = np.asarray(source["encoder_pooled"], dtype=np.float32)
    physical_time, values = physical_ap(fourier_path(case))
    return CaseData(
        name=case,
        time_us=time_us,
        latent_raw=latent.reshape(len(latent), -1).astype(np.float64),
        physical_ap_raw=interpolate(physical_time, values, time_us),
    )


def fit_representation(case: CaseData, start_us: float, representation_end_us: float) -> Representation:
    fit = interval_mask(case.time_us, start_us, representation_end_us)
    count = min(40, int(np.count_nonzero(fit)) - 1, case.latent_raw.shape[1])
    pca = PCA(n_components=count, svd_solver="randomized", random_state=42)
    pca.fit(case.latent_raw[fit])
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)
    dimensions = min(n95, 20)
    latent = pca.transform(case.latent_raw)[:, :dimensions]
    latent = ablation.standardize(latent, fit)
    physical = ablation.standardize(case.physical_ap_raw, fit)
    return Representation(
        time_us=case.time_us,
        latent=latent,
        lap=np.concatenate((latent, physical), axis=1),
        latent_dimensions=dimensions,
        latent_pcs95=n95,
        latent_capture=float(np.sum(pca.explained_variance_ratio_[:dimensions])),
    )


def select_hankel(
    states: np.ndarray,
    time_us: np.ndarray,
    train_start_us: float,
    train_end_us: float,
    validation_end_us: float,
    latent_dimensions: int,
    label: str,
) -> tuple[dict, list[dict]]:
    train_mask = interval_mask(time_us, train_start_us, train_end_us)
    validation_mask = interval_mask(time_us, train_end_us, validation_end_us)
    train = states[train_mask]
    truth = states[validation_mask]
    persistence = np.repeat(train[-1][None], len(truth), axis=0)
    rows = []
    for delay in DELAYS:
        try:
            factorization = ablation.delay_factorization(train, delay, max(RANKS))
        except (ValueError, np.linalg.LinAlgError):
            continue
        for rank in RANKS:
            try:
                model = ablation.radius_normalized_model(
                    ablation.model_from_factorization(factorization, rank)
                )
                prediction = hankel.rollout_hankel(model, train, len(truth))
                metrics = ablation.subset_metrics(
                    truth[:, :latent_dimensions],
                    prediction[:, :latent_dimensions],
                    persistence[:, :latent_dimensions],
                )
                objective = metrics["mse"]
                radius = float(np.max(np.abs(model.eigenvalues)))
            except (ValueError, np.linalg.LinAlgError):
                metrics = {"mse": float("inf"), "skill_persistence": float("-inf"), "correlation": float("nan")}
                objective = float("inf")
                radius = float("nan")
            rows.append(
                {
                    "selection_label": label,
                    "delay": delay,
                    "rank": rank,
                    "validation_latent_mse": metrics["mse"],
                    "validation_skill_vs_persistence": metrics["skill_persistence"],
                    "validation_correlation": metrics["correlation"],
                    "spectral_radius": radius,
                    "objective": objective,
                }
            )
    finite = [row for row in rows if np.isfinite(row["objective"])]
    if not finite:
        raise RuntimeError(f"No valid candidate for {label}")
    selected = min(finite, key=lambda row: (row["objective"], row["delay"], row["rank"]))
    for row in rows:
        row["selected"] = int(
            row["delay"] == selected["delay"] and row["rank"] == selected["rank"]
        )
    return selected, rows


def fit_model(states: np.ndarray, mask: np.ndarray, selected: dict) -> hankel.HankelModel:
    fit = states[mask]
    factorization = ablation.delay_factorization(
        fit, int(selected["delay"]), int(selected["rank"])
    )
    return ablation.radius_normalized_model(
        ablation.model_from_factorization(factorization, int(selected["rank"]))
    )


def fit_expert(label: str, states: np.ndarray, mask: np.ndarray, selected: dict) -> Expert:
    fit = states[mask]
    model = fit_model(states, mask, selected)
    vectors = hankel.make_delay_vectors(fit, model.delay)
    coordinates = model.project(vectors)
    reconstruction = model.reconstruct(coordinates)
    residual = np.mean((vectors - reconstruction) ** 2, axis=1)
    total_variance = float(np.mean(np.var(vectors, axis=0)))
    return Expert(
        label=label,
        model=model,
        residual_scale=max(float(np.quantile(residual, 0.75)), total_variance * 1.0e-5, 1.0e-12),
        coordinate_variance=np.maximum(np.var(coordinates, axis=0, ddof=1), 1.0e-10),
    )


def delay_vector(history: np.ndarray, delay: int) -> np.ndarray:
    return np.concatenate([history[-1 - lag] for lag in range(delay)], axis=0)


def expert_next(expert: Expert, history: np.ndarray) -> np.ndarray:
    vector = delay_vector(history, expert.model.delay)
    coordinate = expert.model.project(vector[None])[0]
    following = expert.model.matrix @ coordinate
    reconstructed = expert.model.reconstruct(following[None])[0]
    return reconstructed[: expert.model.state_dimensions]


def expert_distance(expert: Expert, history: np.ndarray) -> float:
    vector = delay_vector(history, expert.model.delay)
    coordinate = expert.model.project(vector[None])[0]
    reconstructed = expert.model.reconstruct(coordinate[None])[0]
    residual = float(np.mean((vector - reconstructed) ** 2)) / expert.residual_scale
    coordinate_distance = float(
        np.mean(coordinate**2 / expert.coordinate_variance)
    )
    return float(np.log1p(max(residual, 0.0)) + 0.10 * np.log1p(max(coordinate_distance, 0.0)))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(np.clip(shifted, -700.0, 0.0))
    return weights / np.sum(weights)


def gate_weights(experts: list[Expert], history: np.ndarray, kind: str, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray([expert_distance(expert, history) for expert in experts])
    if kind == "hard":
        weights = np.zeros(len(experts), dtype=np.float64)
        weights[int(np.argmin(distances))] = 1.0
    elif kind == "smooth":
        weights = softmax(-distances / temperature)
    else:
        raise ValueError(kind)
    return weights, distances


def rollout_experts(
    experts: list[Expert],
    initial_history: np.ndarray,
    steps: int,
    kind: str,
    temperature: float,
    truth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum_delay = max(expert.model.delay for expert in experts)
    history = [item.copy() for item in initial_history[-maximum_delay:]]
    oracle_history = [item.copy() for item in initial_history[-maximum_delay:]]
    prediction = np.empty((steps, initial_history.shape[1]), dtype=np.float64)
    weights = np.empty((steps, len(experts)), dtype=np.float64)
    distances = np.empty_like(weights)
    for index in range(steps):
        active_history = np.asarray(oracle_history if truth is not None else history)
        local_weights, local_distances = gate_weights(
            experts, active_history, kind, temperature
        )
        predicted_history = np.asarray(history)
        candidates = np.stack(
            [expert_next(expert, predicted_history) for expert in experts], axis=0
        )
        current = history[-1]
        following = current + np.einsum(
            "e,ed->d", local_weights, candidates - current[None]
        )
        prediction[index] = following
        weights[index] = local_weights
        distances[index] = local_distances
        history.append(following)
        history = history[-maximum_delay:]
        if truth is not None:
            oracle_history.append(truth[index].copy())
            oracle_history = oracle_history[-maximum_delay:]
    return prediction, weights, distances


def gate_summary(weights: np.ndarray) -> dict[str, float]:
    choice = np.argmax(weights, axis=1)
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1.0e-30)), axis=1)
    if weights.shape[1] > 1:
        entropy /= math.log(weights.shape[1])
    return {
        "early_expert_fraction": float(np.mean(weights[:, 0])),
        "late_expert_fraction": float(np.mean(weights[:, 1])),
        "expert_switches": int(np.count_nonzero(np.diff(choice))),
        "mean_gate_entropy": float(np.mean(entropy)),
    }


def evaluate_prediction(
    method: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    latent_dimensions: int,
    weights: np.ndarray | None,
) -> dict:
    metrics = ablation.subset_metrics(
        truth[:, :latent_dimensions],
        prediction[:, :latent_dimensions],
        persistence[:, :latent_dimensions],
    )
    result = {
        "method": method,
        "latent_mse": metrics["mse"],
        "latent_skill_vs_persistence": metrics["skill_persistence"],
        "latent_correlation": metrics["correlation"],
        "latent_std_ratio": metrics["std_ratio"],
    }
    if weights is not None:
        result.update(gate_summary(weights))
    return result


def select_temperature(
    experts: list[Expert],
    history: np.ndarray,
    truth: np.ndarray,
    latent_dimensions: int,
) -> tuple[float, list[dict]]:
    persistence = np.repeat(history[-1][None], len(truth), axis=0)
    rows = []
    for temperature in TEMPERATURES:
        prediction, weights, _ = rollout_experts(
            experts, history, len(truth), "smooth", temperature
        )
        result = evaluate_prediction(
            f"smooth_t{temperature:g}", truth, prediction, persistence, latent_dimensions, weights
        )
        result["temperature"] = temperature
        rows.append(result)
    selected = min(rows, key=lambda row: (row["latent_mse"], row["temperature"]))
    return float(selected["temperature"]), rows


def run_protocol(
    case: CaseData,
    protocol: Protocol,
    frozen_temperature: float | None,
) -> tuple[list[dict], list[dict], dict, float | None]:
    representation_end = protocol.fit_end_us - 1.0
    representation = fit_representation(case, protocol.start_us, representation_end)
    time_us = representation.time_us
    selection_rows = []

    fixed_l, trials = select_hankel(
        representation.latent,
        time_us,
        protocol.start_us,
        protocol.fit_end_us - 1.0,
        protocol.fit_end_us,
        representation.latent_dimensions,
        f"{protocol.name}:fixed_L",
    )
    selection_rows.extend(trials)
    fixed_lap, trials = select_hankel(
        representation.lap,
        time_us,
        protocol.start_us,
        protocol.fit_end_us - 1.0,
        protocol.fit_end_us,
        representation.latent_dimensions,
        f"{protocol.name}:fixed_LAP",
    )
    selection_rows.extend(trials)
    early_selected, trials = select_hankel(
        representation.lap,
        time_us,
        protocol.start_us,
        protocol.split_us - 1.0,
        protocol.split_us,
        representation.latent_dimensions,
        f"{protocol.name}:early",
    )
    selection_rows.extend(trials)
    late_selected, trials = select_hankel(
        representation.lap,
        time_us,
        protocol.split_us,
        protocol.fit_end_us - 1.0,
        protocol.fit_end_us,
        representation.latent_dimensions,
        f"{protocol.name}:late",
    )
    selection_rows.extend(trials)

    if protocol.development:
        early_selection_mask = interval_mask(
            time_us, protocol.start_us, protocol.split_us
        )
        late_selection_mask = interval_mask(
            time_us, protocol.split_us, protocol.fit_end_us - 1.0
        )
        temperature_experts = [
            fit_expert("early", representation.lap, early_selection_mask, early_selected),
            fit_expert("late", representation.lap, late_selection_mask, late_selected),
        ]
        gate_validation_history = representation.lap[
            interval_mask(time_us, protocol.start_us, protocol.fit_end_us - 1.0)
        ]
        gate_validation_truth = representation.lap[
            interval_mask(time_us, protocol.fit_end_us - 1.0, protocol.fit_end_us)
        ]
        frozen_temperature, temperature_rows = select_temperature(
            temperature_experts,
            gate_validation_history,
            gate_validation_truth,
            representation.latent_dimensions,
        )
        for row in temperature_rows:
            row.update({"case": case.name, "protocol": protocol.name})
            selection_rows.append(row)
    if frozen_temperature is None:
        raise RuntimeError("The smooth-gate temperature was not initialized")

    fixed_mask = interval_mask(time_us, protocol.start_us, protocol.fit_end_us)
    early_mask = interval_mask(time_us, protocol.start_us, protocol.split_us)
    late_mask = interval_mask(time_us, protocol.split_us, protocol.fit_end_us)
    test_mask = interval_mask(
        time_us, protocol.fit_end_us, protocol.test_end_us, include_end=True
    )
    fixed_l_model = fit_model(representation.latent, fixed_mask, fixed_l)
    fixed_lap_model = fit_model(representation.lap, fixed_mask, fixed_lap)
    experts = [
        fit_expert("early", representation.lap, early_mask, early_selected),
        fit_expert("late", representation.lap, late_mask, late_selected),
    ]
    history_l = representation.latent[fixed_mask]
    history_lap = representation.lap[fixed_mask]
    truth_l = representation.latent[test_mask]
    truth_lap = representation.lap[test_mask]
    persistence_l = np.repeat(history_l[-1][None], len(truth_l), axis=0)
    persistence_lap = np.repeat(history_lap[-1][None], len(truth_lap), axis=0)

    predictions: dict[str, np.ndarray] = {}
    gate_payload: dict[str, np.ndarray] = {}
    predictions["fixed_L"] = hankel.rollout_hankel(fixed_l_model, history_l, len(truth_l))
    predictions["fixed_LAP"] = hankel.rollout_hankel(fixed_lap_model, history_lap, len(truth_lap))
    for name, weights in (
        ("early_only_LAP", np.asarray([1.0, 0.0])),
        ("late_only_LAP", np.asarray([0.0, 1.0])),
    ):
        repeated = np.repeat(weights[None], len(truth_lap), axis=0)
        # Constant expert rollouts use the same increment blending path.
        maximum_delay = max(expert.model.delay for expert in experts)
        local_history = [item.copy() for item in history_lap[-maximum_delay:]]
        constant_prediction = np.empty_like(truth_lap)
        for index in range(len(truth_lap)):
            history_array = np.asarray(local_history)
            candidates = np.stack([expert_next(expert, history_array) for expert in experts])
            current = local_history[-1]
            following = current + np.einsum("e,ed->d", weights, candidates - current[None])
            constant_prediction[index] = following
            local_history.append(following)
            local_history = local_history[-maximum_delay:]
        predictions[name] = constant_prediction
        gate_payload[name] = repeated

    hard_prediction, hard_weights, hard_distances = rollout_experts(
        experts, history_lap, len(truth_lap), "hard", frozen_temperature
    )
    smooth_prediction, smooth_weights, smooth_distances = rollout_experts(
        experts, history_lap, len(truth_lap), "smooth", frozen_temperature
    )
    oracle_prediction, oracle_weights, oracle_distances = rollout_experts(
        experts,
        history_lap,
        len(truth_lap),
        "hard",
        frozen_temperature,
        truth=truth_lap,
    )
    predictions.update(
        {
            "hard_switch_LAP": hard_prediction,
            "smooth_switch_LAP": smooth_prediction,
            "oracle_truth_gate_LAP": oracle_prediction,
        }
    )
    gate_payload.update(
        {
            "hard_switch_LAP": hard_weights,
            "hard_switch_LAP_distance": hard_distances,
            "smooth_switch_LAP": smooth_weights,
            "smooth_switch_LAP_distance": smooth_distances,
            "oracle_truth_gate_LAP": oracle_weights,
            "oracle_truth_gate_LAP_distance": oracle_distances,
        }
    )

    rows = []
    for method, prediction in predictions.items():
        if method == "fixed_L":
            truth = truth_l
            persistence = persistence_l
            weights = None
        else:
            truth = truth_lap
            persistence = persistence_lap
            weights = gate_payload.get(method)
        result = evaluate_prediction(
            method,
            truth,
            prediction,
            persistence,
            representation.latent_dimensions,
            weights,
        )
        result.update(
            {
                "case": case.name,
                "protocol": protocol.name,
                "fit_start_us": protocol.start_us,
                "expert_split_us": protocol.split_us,
                "fit_end_us": protocol.fit_end_us,
                "test_end_us": protocol.test_end_us,
                "latent_dimensions": representation.latent_dimensions,
                "latent_pcs95": representation.latent_pcs95,
                "latent_variance_capture": representation.latent_capture,
                "smooth_temperature": frozen_temperature,
                "future_truth_used_for_gate": int(method == "oracle_truth_gate_LAP"),
            }
        )
        rows.append(result)

    payload = {
        "time_us": time_us[test_mask],
        "truth_latent": truth_l,
        "persistence_latent": persistence_l,
        "predictions": {
            method: prediction[:, : representation.latent_dimensions]
            for method, prediction in predictions.items()
        },
        "gates": gate_payload,
        "selected": {
            "fixed_L": fixed_l,
            "fixed_LAP": fixed_lap,
            "early": early_selected,
            "late": late_selected,
        },
    }
    return rows, selection_rows, payload, frozen_temperature


def plot_summary(rows: list[dict], path: Path) -> None:
    methods = (
        "fixed_L",
        "fixed_LAP",
        "late_only_LAP",
        "hard_switch_LAP",
        "smooth_switch_LAP",
        "oracle_truth_gate_LAP",
    )
    protocols = [protocol.name for protocol in PROTOCOLS]
    skill = np.full((len(methods), len(protocols)), np.nan)
    corr = np.full_like(skill, np.nan)
    lookup = {(row["method"], row["protocol"]): row for row in rows}
    for i, method in enumerate(methods):
        for j, protocol in enumerate(protocols):
            row = lookup.get((method, protocol))
            if row:
                skill[i, j] = row["latent_skill_vs_persistence"]
                corr[i, j] = row["latent_correlation"]
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (skill, corr),
        ("Latent skill versus persistence", "Latent trajectory correlation"),
    ):
        image = axis.imshow(np.clip(values, -1, 1), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(range(len(protocols)), [item.replace("_", "\n") for item in protocols])
        axis.set_yticks(range(len(methods)), methods)
        axis.set_title(title)
        for i in range(len(methods)):
            for j in range(len(protocols)):
                if np.isfinite(values[i, j]):
                    label = f"{values[i, j]:.2f}" if abs(values[i, j]) < 10 else "<-9"
                    axis.text(j, i, label, ha="center", va="center", fontsize=9)
        figure.colorbar(image, ax=axis, shrink=0.85)
    figure.suptitle("Temporal-expert switching ROM: B25 and B15 control", fontsize=17)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_rollouts(payloads: dict[str, dict], path: Path) -> None:
    figure, axes = plt.subplots(len(PROTOCOLS), 1, figsize=(15, 12), constrained_layout=True)
    colors = {
        "fixed_L": "tab:blue",
        "fixed_LAP": "tab:orange",
        "late_only_LAP": "tab:green",
        "hard_switch_LAP": "tab:red",
        "smooth_switch_LAP": "tab:purple",
        "oracle_truth_gate_LAP": "tab:brown",
    }
    for axis, protocol in zip(axes, PROTOCOLS):
        payload = payloads[protocol.name]
        axis.plot(payload["time_us"], payload["truth_latent"][:, 0], color="black", linewidth=2.2, label="truth PC1")
        for method, color in colors.items():
            axis.plot(payload["time_us"], payload["predictions"][method][:, 0], color=color, linewidth=1.2, label=method)
        axis.set_title(protocol.name)
        axis.set_ylabel("standardized latent PC1")
        axis.legend(loc="lower right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time [us]")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_gates(payloads: dict[str, dict], path: Path) -> None:
    figure, axes = plt.subplots(len(PROTOCOLS), 1, figsize=(15, 10), constrained_layout=True)
    for axis, protocol in zip(axes, PROTOCOLS):
        payload = payloads[protocol.name]
        time_us = payload["time_us"]
        for method, style in (
            ("hard_switch_LAP", "-"),
            ("smooth_switch_LAP", "--"),
            ("oracle_truth_gate_LAP", ":"),
        ):
            axis.plot(time_us, payload["gates"][method][:, 1], style, linewidth=1.5, label=f"{method}: late weight")
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("late expert weight")
        axis.set_title(protocol.name)
        axis.legend(loc="lower right")
    axes[-1].set_xlabel("Time [us]")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = {name: load_case(name) for name in ("B15", "B25")}
    all_rows: list[dict] = []
    selection_rows: list[dict] = []
    payloads: dict[str, dict] = {}
    frozen_temperature: float | None = None
    for protocol in PROTOCOLS:
        rows, trials, payload, frozen_temperature = run_protocol(
            cases[protocol.case], protocol, frozen_temperature
        )
        all_rows.extend(rows)
        selection_rows.extend(trials)
        payloads[protocol.name] = payload
        best = max(
            [row for row in rows if not row["future_truth_used_for_gate"]],
            key=lambda row: row["latent_skill_vs_persistence"],
        )
        print(
            f"[PASS] {protocol.name}: best={best['method']} "
            f"skill={best['latent_skill_vs_persistence']:.3f} "
            f"corr={best['latent_correlation']:.3f}",
            flush=True,
        )
    write_csv(OUTPUT / "switching_rom_metrics.csv", all_rows)
    write_csv(OUTPUT / "model_selection.csv", selection_rows)
    with h5py.File(OUTPUT / "switching_rom_rollouts.h5", "w") as output:
        for protocol, payload in payloads.items():
            group = output.require_group(protocol)
            group.create_dataset("time_us", data=payload["time_us"])
            group.create_dataset("truth_latent", data=payload["truth_latent"], compression="gzip", compression_opts=4)
            for method, prediction in payload["predictions"].items():
                group.create_dataset(f"prediction/{method}", data=prediction, compression="gzip", compression_opts=4)
            for method, values in payload["gates"].items():
                group.create_dataset(f"gate/{method}", data=values, compression="gzip", compression_opts=4)
    plot_summary(all_rows, OUTPUT / "switching_rom_summary.png")
    plot_rollouts(payloads, OUTPUT / "switching_rom_selected_rollouts.png")
    plot_gates(payloads, OUTPUT / "switching_rom_gate_weights.png")

    lookup = {(row["protocol"], row["method"]): row for row in all_rows}
    lines = [
        "# B25 temporal-expert switching ROM",
        "",
        "Two chronological Hankel-DMD experts are fitted before each holdout.",
        "The autonomous gate uses only predicted-state distance to the fitted delay",
        "manifolds. `oracle_truth_gate_LAP` is diagnostic and is not a deployable ROM.",
        "All operators are normalized to spectral radius <= 1 for a fair stability control.",
        "",
        f"Smooth-gate temperature `{frozen_temperature:g}` was selected only on the B25 development validation interval and then frozen.",
        "",
        "| protocol | method | skill | correlation | late fraction | switches |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for protocol in PROTOCOLS:
        for method in (
            "fixed_L",
            "fixed_LAP",
            "late_only_LAP",
            "hard_switch_LAP",
            "smooth_switch_LAP",
            "oracle_truth_gate_LAP",
        ):
            row = lookup[(protocol.name, method)]
            lines.append(
                f"| {protocol.name} | {method} | {row['latent_skill_vs_persistence']:.3f} | "
                f"{row['latent_correlation']:.3f} | {row.get('late_expert_fraction', float('nan')):.3f} | "
                f"{row.get('expert_switches', '')} |"
            )
    development_fixed = lookup[(PROTOCOLS[0].name, "fixed_L")]
    development_hard = lookup[(PROTOCOLS[0].name, "hard_switch_LAP")]
    development_oracle = lookup[(PROTOCOLS[0].name, "oracle_truth_gate_LAP")]
    later_fixed = lookup[(PROTOCOLS[1].name, "fixed_LAP")]
    later_switch = lookup[(PROTOCOLS[1].name, "smooth_switch_LAP")]
    control_fixed = lookup[(PROTOCOLS[2].name, "fixed_L")]
    control_late = lookup[(PROTOCOLS[2].name, "late_only_LAP")]
    lines.extend(
        [
            "",
            "The primary causal question is whether hard/smooth switching beats both",
            "`fixed_LAP` and `late_only_LAP`. Beating only the global fixed model does",
            "not establish switching; it may merely show that the recent expert is better.",
            "B15 is a negative control for over-sensitive gating.",
            "",
            "## Interpretation",
            "",
            f"In the first B25 holdout, fixed L is better than hard switching ({development_fixed['latent_skill_vs_persistence']:.3f} versus {development_hard['latent_skill_vs_persistence']:.3f}). The oracle truth gate makes the same all-early choice and obtains {development_oracle['latent_skill_vs_persistence']:.3f}, so autonomous gate lock-in is not the main limitation.",
            f"In the later B25 holdout, fixed L+A+P remains better than smooth switching ({later_fixed['latent_skill_vs_persistence']:.3f} versus {later_switch['latent_skill_vs_persistence']:.3f}).",
            f"B15 is already closed by one stationary model: fixed L gives {control_fixed['latent_skill_vs_persistence']:.3f}, while the late expert alone gives {control_late['latent_skill_vs_persistence']:.3f}.",
            "This simple chronological two-expert partition therefore does not recover B25 closure. It rejects the claim that merely dividing time into early and late experts is sufficient; it does not reject organization-defined switching or smoothly time-varying operators.",
            "",
            "## Outputs",
            "",
            "- `switching_rom_metrics.csv`",
            "- `model_selection.csv`",
            "- `switching_rom_rollouts.h5`",
            "- `switching_rom_summary.png`",
            "- `switching_rom_selected_rollouts.png`",
            "- `switching_rom_gate_weights.png`",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS",
        "smooth_temperature": frozen_temperature,
        "protocols": [protocol.__dict__ for protocol in PROTOCOLS],
        "metrics": all_rows,
        "future_truth_used_by_primary_models": False,
        "oracle_is_diagnostic_only": True,
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"[PASS] wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
