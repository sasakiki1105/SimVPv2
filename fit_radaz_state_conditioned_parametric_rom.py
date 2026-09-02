"""Fit a data-only state-conditioned parametric ROM for RadAz Ez sweeps.

The representation is deliberately shared across electric-field conditions:

* L: 20 frozen-SimVP Fourier latent block-PCA coordinates,
* R: 8 radial phi-envelope observables (4 radial x 2 mode bands),
* T: 2 modal-transport observables (MTSI/ECDI bands).

Local delay-coordinate DMD experts are fitted in this common 30-dimensional
state.  Their one-step increments are mixed using an Ez prior, optionally
tempered by distance to each expert's fitted delay manifold.  Ez=25 kV/m is
first held out as an interpolation validation between E20 and E30.  Only after
the gate protocol is selected is the final E20/E25/E30 model fitted.  No
Ez=22.5 kV/m data are loaded by this script; the resulting model is therefore
locked before the up/down sweep tests are inspected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import analyze_radaz_hankel_havok as hankel
import analyze_radaz_local_rom_closure_map as closure_map
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "fit_radaz_state_conditioned_parametric_rom"
)
DEFAULT_FIELDS = (20, 25, 30)
VALIDATION_FIELD = 25
FIT_START_US = 12.0
SELECTION_SPLIT_US = 18.0
SELECTION_END_US = 20.0
FINAL_FIT_END_US = 24.0
FORECAST_END_US = 30.0
STATE_GROUPS = ("latent", "radial", "transport")
STATE_LABEL = "L+R+T"
LATENT_BUDGET = "medium_20"
STABILITY_LIMIT = 1.002
EZ_RBF_BANDWIDTH_KVM = 5.0
MODE_BAND_LABELS = ("MTSI_n1_6", "ECDI_n9_21")

# This list is declared before the unseen E22.5 sweeps are loaded.  E25 is a
# development interpolation target, so selecting among these gates on E25 is
# permitted; the selected gate is then frozen for both E22.5 directions.
GATE_SPECS = (
    ("linear_ez", "linear", None),
    ("rbf_ez", "rbf", None),
    ("hybrid_linear_t0p5", "linear", 0.5),
    ("hybrid_linear_t1", "linear", 1.0),
    ("hybrid_linear_t2", "linear", 2.0),
    ("hybrid_linear_t5", "linear", 5.0),
    ("hybrid_rbf_t0p5", "rbf", 0.5),
    ("hybrid_rbf_t1", "rbf", 1.0),
    ("hybrid_rbf_t2", "rbf", 2.0),
    ("hybrid_rbf_t5", "rbf", 5.0),
)


@dataclass
class RawCase:
    ez_kvm: int
    feature_path: Path
    physical_path: Path
    features: np.ndarray
    time_us: np.ndarray
    frame: np.ndarray
    physical: augmented.PhysicalStates


@dataclass
class SharedRepresentation:
    fit_fields: tuple[int, ...]
    block_models: dict[str, block.BlockPCA]
    scaler: augmented.GroupScaler
    states: dict[int, np.ndarray]
    groups: dict[int, dict[str, np.ndarray]]
    pca_rows: list[dict]


@dataclass
class LocalExpert:
    ez_kvm: int
    model: hankel.HankelModel
    residual_scale: float
    coordinate_variance: np.ndarray
    selected_delay: int
    selected_rank: int
    selected_ridge_label: str


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def interval_mask(
    time_us: np.ndarray,
    start_us: float,
    end_us: float,
    *,
    include_end: bool = False,
) -> np.ndarray:
    upper = time_us <= end_us + 1.0e-10 if include_end else time_us < end_us - 1.0e-10
    return (time_us >= start_us - 1.0e-10) & upper


def load_case(ez_kvm: int) -> RawCase:
    feature_path = closure_map.latent_path(ez_kvm)
    physical_path = (
        closure_map.DEFAULT_OUTPUT
        / "cases"
        / f"E{ez_kvm}kVm"
        / "physical_fourier_targets.h5"
    )
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    if not physical_path.is_file():
        source = closure_map.source_path(ez_kvm)
        if not source.is_file():
            raise FileNotFoundError(source)
        physical_path.parent.mkdir(parents=True, exist_ok=True)
        import analyze_radaz_fourier_latent_to_physical_modes as physical_extract

        _, feature_time, feature_frame = block.load_features(feature_path)
        physical_extract.extract_physical_fourier(
            source,
            physical_path,
            feature_frame,
            feature_time,
            bands=8,
            maximum_mode=21,
        )
    features, time_us, frame = block.load_features(feature_path)
    physical = augmented.load_physical_states(physical_path)
    if not np.array_equal(frame, physical.frame):
        raise ValueError(f"Frame mismatch at Ez={ez_kvm}")
    if not np.allclose(time_us, physical.time_us, atol=1.0e-9, rtol=0.0):
        raise ValueError(f"Time mismatch at Ez={ez_kvm}")
    return RawCase(
        ez_kvm=ez_kvm,
        feature_path=feature_path,
        physical_path=physical_path,
        features=features,
        time_us=time_us,
        frame=frame,
        physical=physical,
    )


def fit_common_block_models(
    cases: dict[int, RawCase],
    fit_fields: tuple[int, ...],
    fit_start_us: float,
    fit_end_us: float,
) -> tuple[dict[str, block.BlockPCA], list[dict]]:
    budget = block.BUDGETS[LATENT_BUDGET]
    models: dict[str, block.BlockPCA] = {}
    rows: list[dict] = []
    for name, (mode_start, mode_end) in block.BLOCKS.items():
        fit_arrays = []
        feature_shape = None
        for field in fit_fields:
            case = cases[field]
            mask = interval_mask(
                case.time_us, fit_start_us, fit_end_us
            )
            values = block.block_slice(
                case.features, mode_start, mode_end
            )
            feature_shape = values.shape[1:]
            fit_arrays.append(values[mask].reshape(np.count_nonzero(mask), -1))
        fit = np.concatenate(fit_arrays, axis=0)
        full_mean = np.mean(fit, axis=0)
        variance = np.var(fit, axis=0)
        floor = max(float(np.max(variance)) * 1.0e-12, 1.0e-20)
        active = variance > floor
        components = min(
            int(budget[name]),
            int(np.count_nonzero(active)),
            len(fit) - 1,
        )
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=0,
            iterated_power=5,
        )
        pca.fit(fit[:, active])
        model = block.BlockPCA(
            name=name,
            mode_start=mode_start,
            mode_end=mode_end,
            components=components,
            feature_shape=tuple(feature_shape),
            full_mean=full_mean,
            active=active,
            pca=pca,
        )
        models[name] = model
        rows.append(
            {
                "block": name,
                "mode_start": mode_start,
                "mode_end": mode_end,
                "retained_components": components,
                "active_features": int(np.count_nonzero(active)),
                "total_features": int(fit.shape[1]),
                "explained_variance": float(
                    np.sum(pca.explained_variance_ratio_)
                ),
                "fit_fields_kvm": ",".join(map(str, fit_fields)),
            }
        )
    return models, rows


def transform_latent(
    models: dict[str, block.BlockPCA], features: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [models[name].transform(features) for name in block.BLOCKS],
        axis=1,
    )


def fit_shared_representation(
    cases: dict[int, RawCase],
    fit_fields: tuple[int, ...],
    fit_start_us: float,
    fit_end_us: float,
) -> SharedRepresentation:
    models, pca_rows = fit_common_block_models(
        cases, fit_fields, fit_start_us, fit_end_us
    )
    groups: dict[int, dict[str, np.ndarray]] = {}
    for field, case in cases.items():
        physical = augmented.flatten_physical(case.physical)
        groups[field] = {
            "latent": transform_latent(models, case.features),
            "radial": physical["radial"],
            "transport": physical["transport"],
        }
    stacked = {
        name: np.concatenate(
            [
                groups[field][name][
                    interval_mask(
                        cases[field].time_us,
                        fit_start_us,
                        fit_end_us,
                    )
                ]
                for field in fit_fields
            ],
            axis=0,
        )
        for name in STATE_GROUPS
    }
    scaler = augmented.GroupScaler.fit(
        stacked, np.ones(len(stacked[STATE_GROUPS[0]]), dtype=bool)
    )
    states = {field: scaler.transform(value) for field, value in groups.items()}
    dimensions = {name: groups[fit_fields[0]][name].shape[1] for name in STATE_GROUPS}
    if sum(dimensions.values()) != 30:
        raise ValueError(f"Expected 30 L+R+T states, got {dimensions}")
    return SharedRepresentation(
        fit_fields=fit_fields,
        block_models=models,
        scaler=scaler,
        states=states,
        groups=groups,
        pca_rows=pca_rows,
    )


def build_rank_model_from_svd(
    states: np.ndarray,
    delay: int,
    rank: int,
    delay_vectors: np.ndarray,
    delay_mean: np.ndarray,
    right: np.ndarray,
    singular_values: np.ndarray,
) -> hankel.HankelModel:
    return block.make_rank_model(
        states,
        delay,
        rank,
        delay_vectors,
        delay_mean,
        right,
        singular_values,
    )


def select_expert_hyperparameters(
    field: int,
    case: RawCase,
    states: np.ndarray,
    delays: tuple[int, ...],
    ranks: tuple[int, ...],
) -> tuple[dict, list[dict]]:
    train_mask = interval_mask(
        case.time_us, FIT_START_US, SELECTION_SPLIT_US
    )
    validation_mask = interval_mask(
        case.time_us,
        SELECTION_SPLIT_US,
        SELECTION_END_US,
    )
    train = states[train_mask]
    truth = states[validation_mask]
    persistence = np.repeat(train[-1:, :], len(truth), axis=0)
    rows: list[dict] = []
    for delay in delays:
        delay_vectors = hankel.make_delay_vectors(train, delay)
        delay_mean = np.mean(delay_vectors, axis=0)
        centered = delay_vectors - delay_mean
        _, singular_values, right = np.linalg.svd(
            centered, full_matrices=False
        )
        maximum_rank = min(len(delay_vectors) - 1, right.shape[0])
        for rank in ranks:
            if rank > maximum_rank:
                continue
            try:
                model = build_rank_model_from_svd(
                    train,
                    delay,
                    rank,
                    delay_vectors,
                    delay_mean,
                    right,
                    singular_values,
                )
                prediction = hankel.rollout_hankel(
                    model, train, len(truth)
                )
                metrics, _ = reduced.evaluate_prediction(
                    truth,
                    prediction,
                    persistence,
                    case.time_us[validation_mask],
                )
                mse = float(metrics["standardized_mse"])
                radius = float(np.max(np.abs(model.eigenvalues)))
                objective = mse + 100.0 * max(
                    0.0, radius - STABILITY_LIMIT
                )
            except (ValueError, np.linalg.LinAlgError):
                mse = float("inf")
                radius = float("inf")
                objective = float("inf")
                metrics = {}
            rows.append(
                {
                    "electric_field_kvm": field,
                    "delay": delay,
                    "history_us": delay * float(np.median(np.diff(case.time_us))),
                    "rank": rank,
                    "validation_mse": mse,
                    "validation_skill_vs_persistence": metrics.get(
                        "skill_vs_persistence", float("nan")
                    ),
                    "validation_correlation": metrics.get(
                        "flattened_correlation", float("nan")
                    ),
                    "spectral_radius": radius,
                    "objective": objective,
                }
            )
    finite = [row for row in rows if np.isfinite(row["objective"])]
    if not finite:
        raise RuntimeError(f"No valid expert candidate for E{field}")
    selected = min(
        finite,
        key=lambda row: (row["objective"], row["delay"], row["rank"]),
    )
    for row in rows:
        row["selected"] = bool(
            row["delay"] == selected["delay"]
            and row["rank"] == selected["rank"]
        )
    return selected, rows


def fit_expert(
    field: int,
    case: RawCase,
    states: np.ndarray,
    selected: dict,
    fit_end_us: float,
) -> LocalExpert:
    fit_mask = interval_mask(
        case.time_us, FIT_START_US, fit_end_us
    )
    fit_states = states[fit_mask]
    model = hankel.fit_hankel_dmd(
        fit_states,
        int(selected["delay"]),
        int(selected["rank"]),
    )
    delay_vectors = hankel.make_delay_vectors(fit_states, model.delay)
    coordinates = model.project(delay_vectors)
    reconstructed = model.reconstruct(coordinates)
    residual = np.mean((delay_vectors - reconstructed) ** 2, axis=1)
    total_variance = float(np.mean(np.var(delay_vectors, axis=0)))
    residual_scale = max(
        float(np.quantile(residual, 0.75)),
        total_variance * 1.0e-5,
        1.0e-12,
    )
    coordinate_variance = np.maximum(
        np.var(coordinates, axis=0, ddof=1), 1.0e-10
    )
    return LocalExpert(
        ez_kvm=field,
        model=model,
        residual_scale=residual_scale,
        coordinate_variance=coordinate_variance,
        selected_delay=model.delay,
        selected_rank=model.rank,
        selected_ridge_label="truncated_svd_pinv_rcond1e-10",
    )


def delay_vector(history: np.ndarray, delay: int) -> np.ndarray:
    if len(history) < delay:
        raise ValueError(f"Need {delay} history states, got {len(history)}")
    return np.concatenate(
        [history[-1 - lag] for lag in range(delay)], axis=0
    )


def expert_next(expert: LocalExpert, history: np.ndarray) -> np.ndarray:
    vector = delay_vector(history, expert.model.delay)
    coordinate = expert.model.project(vector[None, :])[0]
    following = expert.model.matrix @ coordinate
    reconstructed = expert.model.reconstruct(following[None, :])[0]
    return reconstructed[: expert.model.state_dimensions]


def expert_distance(expert: LocalExpert, history: np.ndarray) -> float:
    vector = delay_vector(history, expert.model.delay)
    coordinate = expert.model.project(vector[None, :])[0]
    reconstructed = expert.model.reconstruct(coordinate[None, :])[0]
    residual = float(np.mean((vector - reconstructed) ** 2))
    residual_distance = residual / expert.residual_scale
    coordinate_distance = float(
        np.mean(coordinate**2 / expert.coordinate_variance)
    )
    return float(
        np.log1p(max(residual_distance, 0.0))
        + 0.10 * np.log1p(max(coordinate_distance, 0.0))
    )


def softmax(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(np.clip(shifted, -700.0, 0.0))
    return weights / np.sum(weights)


def linear_ez_prior(experts: list[LocalExpert], target_ez: float) -> np.ndarray:
    fields = np.asarray([expert.ez_kvm for expert in experts], dtype=np.float64)
    order = np.argsort(fields)
    sorted_fields = fields[order]
    sorted_weights = np.zeros(len(experts), dtype=np.float64)
    if target_ez <= sorted_fields[0]:
        sorted_weights[0] = 1.0
    elif target_ez >= sorted_fields[-1]:
        sorted_weights[-1] = 1.0
    else:
        upper = int(np.searchsorted(sorted_fields, target_ez))
        lower = upper - 1
        span = sorted_fields[upper] - sorted_fields[lower]
        fraction = (target_ez - sorted_fields[lower]) / span
        sorted_weights[lower] = 1.0 - fraction
        sorted_weights[upper] = fraction
    weights = np.empty_like(sorted_weights)
    weights[order] = sorted_weights
    return weights


def rbf_ez_prior(experts: list[LocalExpert], target_ez: float) -> np.ndarray:
    fields = np.asarray([expert.ez_kvm for expert in experts], dtype=np.float64)
    return softmax(
        -0.5 * ((fields - target_ez) / EZ_RBF_BANDWIDTH_KVM) ** 2
    )


def gate_weights(
    experts: list[LocalExpert],
    history: np.ndarray,
    target_ez: float,
    prior_kind: str,
    state_temperature: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if prior_kind == "linear":
        prior = linear_ez_prior(experts, target_ez)
    elif prior_kind == "rbf":
        prior = rbf_ez_prior(experts, target_ez)
    else:
        raise ValueError(f"Unknown prior kind: {prior_kind}")
    distances = np.asarray(
        [expert_distance(expert, history) for expert in experts],
        dtype=np.float64,
    )
    if state_temperature is None:
        return prior, distances
    # A small floor prevents an expert from becoming mathematically
    # unreachable, while the Ez prior still dominates distant conditions.
    log_prior = np.log(np.maximum(prior, 1.0e-3))
    weights = softmax(log_prior - distances / state_temperature)
    return weights, distances


def rollout_parametric(
    experts: list[LocalExpert],
    initial_history: np.ndarray,
    target_ez: float,
    prior_kind: str,
    state_temperature: float | None,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum_delay = max(expert.model.delay for expert in experts)
    history = [state.copy() for state in initial_history[-maximum_delay:]]
    prediction = np.empty((steps, initial_history.shape[1]), dtype=np.float64)
    weights_over_time = np.empty((steps, len(experts)), dtype=np.float64)
    distances_over_time = np.empty((steps, len(experts)), dtype=np.float64)
    for index in range(steps):
        history_array = np.asarray(history, dtype=np.float64)
        weights, distances = gate_weights(
            experts,
            history_array,
            target_ez,
            prior_kind,
            state_temperature,
        )
        candidates = np.stack(
            [expert_next(expert, history_array) for expert in experts], axis=0
        )
        current = history[-1]
        increments = candidates - current[None, :]
        following = current + np.einsum("e,ed->d", weights, increments)
        weights_over_time[index] = weights
        distances_over_time[index] = distances
        if (
            not np.all(np.isfinite(following))
            or np.max(np.abs(following)) > 1.0e8
        ):
            prediction[index:] = np.nan
            weights_over_time[index + 1 :] = np.nan
            distances_over_time[index + 1 :] = np.nan
            break
        prediction[index] = following
        history.append(following)
        if len(history) > maximum_delay:
            history.pop(0)
    return prediction, weights_over_time, distances_over_time


def finite_correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    return block.safe_correlation(truth, prediction)


def temporal_anomaly_correlation(
    truth: np.ndarray, prediction: np.ndarray
) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return finite_correlation(
        truth - np.mean(truth, axis=0, keepdims=True),
        prediction - np.mean(prediction, axis=0, keepdims=True),
    )


def skill_vs_baseline(
    truth: np.ndarray, prediction: np.ndarray, baseline: np.ndarray
) -> float:
    finite = np.isfinite(prediction)
    if not np.all(finite):
        return float("-inf")
    mse = float(np.mean((prediction - truth) ** 2))
    baseline_mse = float(np.mean((baseline - truth) ** 2))
    if baseline_mse <= np.finfo(float).tiny:
        return float("nan")
    return 1.0 - mse / baseline_mse


def evaluate_rollout(
    method: str,
    representation: SharedRepresentation,
    field: int,
    forecast_mask: np.ndarray,
    prediction: np.ndarray,
    persistence_state: np.ndarray,
) -> dict:
    truth_state = representation.states[field][forecast_mask]
    finite_fraction = float(np.mean(np.isfinite(prediction)))
    prediction_groups = representation.scaler.inverse(prediction)
    truth_groups = {
        name: representation.groups[field][name][forecast_mask]
        for name in STATE_GROUPS
    }
    persistence_groups = representation.scaler.inverse(persistence_state)
    state_skill = skill_vs_baseline(
        truth_state, prediction, persistence_state
    )
    radial_anomaly = temporal_anomaly_correlation(
        truth_groups["radial"], prediction_groups["radial"]
    )
    transport_anomaly = temporal_anomaly_correlation(
        truth_groups["transport"], prediction_groups["transport"]
    )
    row = {
        "method": method,
        "target_electric_field_kvm": field,
        "state_dimension": truth_state.shape[1],
        "finite_fraction": finite_fraction,
        "state_mse": float(np.mean((prediction - truth_state) ** 2)),
        "state_correlation": finite_correlation(truth_state, prediction),
        "state_skill_vs_persistence": state_skill,
        "radial_correlation": finite_correlation(
            truth_groups["radial"], prediction_groups["radial"]
        ),
        "radial_temporal_anomaly_correlation": radial_anomaly,
        "radial_skill_vs_persistence": skill_vs_baseline(
            truth_groups["radial"],
            prediction_groups["radial"],
            persistence_groups["radial"],
        ),
        "transport_correlation": finite_correlation(
            truth_groups["transport"], prediction_groups["transport"]
        ),
        "transport_temporal_anomaly_correlation": transport_anomaly,
        "transport_skill_vs_persistence": skill_vs_baseline(
            truth_groups["transport"],
            prediction_groups["transport"],
            persistence_groups["transport"],
        ),
    }
    for band_index, label in enumerate(MODE_BAND_LABELS):
        row[f"{label}_transport_correlation"] = finite_correlation(
            truth_groups["transport"][:, band_index],
            prediction_groups["transport"][:, band_index],
        )
        row[f"{label}_transport_skill"] = skill_vs_baseline(
            truth_groups["transport"][:, band_index],
            prediction_groups["transport"][:, band_index],
            persistence_groups["transport"][:, band_index],
        )
    selection_terms = np.asarray(
        [state_skill, radial_anomaly, transport_anomaly], dtype=np.float64
    )
    row["selection_score"] = (
        float(np.mean(selection_terms[np.isfinite(selection_terms)]))
        if np.any(np.isfinite(selection_terms))
        else float("-inf")
    )
    return row


def validate_e25_interpolation(
    cases: dict[int, RawCase],
    representation: SharedRepresentation,
    selected_by_field: dict[int, dict],
) -> tuple[list[dict], dict[str, dict], dict[str, np.ndarray], LocalExpert]:
    source_fields = representation.fit_fields
    experts = [
        fit_expert(
            field,
            cases[field],
            representation.states[field],
            selected_by_field[field],
            FINAL_FIT_END_US,
        )
        for field in source_fields
    ]
    experts.sort(key=lambda expert: expert.ez_kvm)
    target = cases[VALIDATION_FIELD]
    history_mask = interval_mask(
        target.time_us, FIT_START_US, FINAL_FIT_END_US
    )
    forecast_mask = interval_mask(
        target.time_us,
        FINAL_FIT_END_US,
        FORECAST_END_US,
        include_end=True,
    )
    history = representation.states[VALIDATION_FIELD][history_mask]
    truth = representation.states[VALIDATION_FIELD][forecast_mask]
    persistence = np.repeat(history[-1:, :], len(truth), axis=0)
    series: dict[str, dict] = {
        "truth": {"state": truth},
        "persistence": {"state": persistence},
    }
    metrics = [
        evaluate_rollout(
            "persistence",
            representation,
            VALIDATION_FIELD,
            forecast_mask,
            persistence,
            persistence,
        )
    ]
    for name, prior_kind, temperature in GATE_SPECS:
        prediction, weights, distances = rollout_parametric(
            experts,
            history,
            VALIDATION_FIELD,
            prior_kind,
            temperature,
            len(truth),
        )
        row = evaluate_rollout(
            name,
            representation,
            VALIDATION_FIELD,
            forecast_mask,
            prediction,
            persistence,
        )
        row.update(
            {
                "prior_kind": prior_kind,
                "state_temperature": (
                    temperature if temperature is not None else float("nan")
                ),
                "mean_gate_entropy": float(
                    np.nanmean(
                        -np.sum(
                            weights
                            * np.log(np.maximum(weights, np.finfo(float).tiny)),
                            axis=1,
                        )
                        / np.log(len(experts))
                    )
                ),
                "expert_fields_kvm": ",".join(
                    str(expert.ez_kvm) for expert in experts
                ),
            }
        )
        metrics.append(row)
        series[name] = {
            "state": prediction,
            "weights": weights,
            "distances": distances,
        }

    # This is an upper-bound diagnostic: it uses E25 transitions to fit the
    # local dynamics but retains the E20/E30-fitted common representation.
    oracle_selected, _ = select_expert_hyperparameters(
        VALIDATION_FIELD,
        target,
        representation.states[VALIDATION_FIELD],
        tuple(sorted({row["delay"] for row in selected_by_field.values()})),
        tuple(sorted({row["rank"] for row in selected_by_field.values()})),
    )
    oracle = fit_expert(
        VALIDATION_FIELD,
        target,
        representation.states[VALIDATION_FIELD],
        oracle_selected,
        FINAL_FIT_END_US,
    )
    oracle_prediction = hankel.rollout_hankel(
        oracle.model, history, len(truth)
    )
    metrics.append(
        evaluate_rollout(
            "oracle_local_e25",
            representation,
            VALIDATION_FIELD,
            forecast_mask,
            oracle_prediction,
            persistence,
        )
    )
    series["oracle_local_e25"] = {"state": oracle_prediction}
    arrays = {
        "time_us": target.time_us[forecast_mask],
        "frame": target.frame[forecast_mask],
        "persistence": persistence,
    }
    return metrics, series, arrays, oracle


def best_gate_row(metrics: list[dict]) -> dict:
    candidates = [
        row
        for row in metrics
        if row["method"] not in ("persistence", "oracle_local_e25")
        and np.isfinite(float(row["selection_score"]))
        and float(row["finite_fraction"]) == 1.0
    ]
    if not candidates:
        raise RuntimeError("Every parametric gate failed E25 validation")
    return max(candidates, key=lambda row: float(row["selection_score"]))


def fit_final_experts(
    cases: dict[int, RawCase],
    representation: SharedRepresentation,
    selected_by_field: dict[int, dict],
) -> list[LocalExpert]:
    experts = [
        fit_expert(
            field,
            cases[field],
            representation.states[field],
            selected_by_field[field],
            FINAL_FIT_END_US,
        )
        for field in representation.fit_fields
    ]
    return sorted(experts, key=lambda expert: expert.ez_kvm)


def save_validation_h5(
    path: Path,
    representation: SharedRepresentation,
    series: dict[str, dict],
    arrays: dict[str, np.ndarray],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["target_electric_field_kvm"] = VALIDATION_FIELD
        handle.attrs["state"] = STATE_LABEL
        axes = handle.require_group("axes")
        for name, values in arrays.items():
            axes.create_dataset(name, data=values, compression="gzip")
        for method, payload in series.items():
            group = handle.require_group(method)
            for name, values in payload.items():
                group.create_dataset(name, data=values, compression="gzip")
            if "state" in payload:
                decoded = representation.scaler.inverse(payload["state"])
                for group_name, values in decoded.items():
                    group.create_dataset(
                        group_name, data=values, compression="gzip"
                    )


def save_locked_model(
    path: Path,
    representation: SharedRepresentation,
    experts: list[LocalExpert],
    selected_gate: dict,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "RadAz state-conditioned parametric ROM"
        handle.attrs["format_version"] = 1
        handle.attrs["state"] = STATE_LABEL
        handle.attrs["state_dimension"] = 30
        handle.attrs["training_fields_kvm"] = np.asarray(
            representation.fit_fields, dtype=np.int64
        )
        handle.attrs["target_sweep_field_kvm"] = 22.5
        handle.attrs["target_sweep_data_loaded"] = False
        handle.attrs["selected_gate"] = selected_gate["method"]
        handle.attrs["prior_kind"] = selected_gate["prior_kind"]
        handle.attrs["state_temperature"] = selected_gate.get(
            "state_temperature", np.nan
        )
        representation_group = handle.require_group("representation")
        blocks_group = representation_group.require_group("latent_blocks")
        for name, model in representation.block_models.items():
            group = blocks_group.require_group(name)
            group.attrs["mode_start"] = model.mode_start
            group.attrs["mode_end"] = model.mode_end
            group.attrs["components"] = model.components
            group.attrs["feature_shape"] = model.feature_shape
            group.create_dataset("full_mean", data=model.full_mean)
            group.create_dataset("active", data=model.active.astype(np.uint8))
            group.create_dataset("pca_mean", data=model.pca.mean_)
            group.create_dataset("pca_components", data=model.pca.components_)
            group.create_dataset(
                "explained_variance_ratio",
                data=model.pca.explained_variance_ratio_,
            )
        scaler_group = representation_group.require_group("group_scaler")
        scaler_group.attrs["names"] = np.asarray(
            representation.scaler.names, dtype=h5py.string_dtype("utf-8")
        )
        for name in representation.scaler.names:
            group = scaler_group.require_group(name)
            group.attrs["weight"] = representation.scaler.weights[name]
            group.attrs["slice_start"] = representation.scaler.slices[name].start
            group.attrs["slice_stop"] = representation.scaler.slices[name].stop
            group.create_dataset("mean", data=representation.scaler.means[name])
            group.create_dataset("scale", data=representation.scaler.scales[name])
        experts_group = handle.require_group("experts")
        for expert in experts:
            group = experts_group.require_group(f"E{expert.ez_kvm}kVm")
            group.attrs["electric_field_kvm"] = expert.ez_kvm
            group.attrs["delay"] = expert.model.delay
            group.attrs["rank"] = expert.model.rank
            group.attrs["state_dimensions"] = expert.model.state_dimensions
            group.attrs["residual_scale"] = expert.residual_scale
            group.attrs["spectral_radius"] = float(
                np.max(np.abs(expert.model.eigenvalues))
            )
            group.create_dataset("delay_mean", data=expert.model.delay_mean)
            group.create_dataset("basis", data=expert.model.basis)
            group.create_dataset("matrix", data=expert.model.matrix)
            group.create_dataset("eigenvalues", data=expert.model.eigenvalues)
            group.create_dataset(
                "coordinate_variance", data=expert.coordinate_variance
            )


def plot_validation(
    path: Path,
    representation: SharedRepresentation,
    series: dict[str, dict],
    arrays: dict[str, np.ndarray],
    best_method: str,
) -> None:
    time_us = arrays["time_us"]
    truth = representation.scaler.inverse(series["truth"]["state"])
    persistence = representation.scaler.inverse(
        series["persistence"]["state"]
    )
    best = representation.scaler.inverse(series[best_method]["state"])
    oracle = representation.scaler.inverse(
        series["oracle_local_e25"]["state"]
    )
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    for band, label in enumerate(MODE_BAND_LABELS):
        axis = axes[band]
        axis.plot(time_us, truth["transport"][:, band], color="#111111", label="PIC truth")
        axis.plot(time_us, persistence["transport"][:, band], color="#999999", linestyle=":", label="persistence")
        axis.plot(time_us, best["transport"][:, band], color="#0072B2", label=best_method)
        axis.plot(time_us, oracle["transport"][:, band], color="#D55E00", linestyle="--", label="oracle local E25")
        axis.set_ylabel(f"{label}\ntransport")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    state_truth = series["truth"]["state"]
    for method, color in ((best_method, "#0072B2"), ("oracle_local_e25", "#D55E00"), ("persistence", "#999999")):
        error = np.sqrt(np.mean((series[method]["state"] - state_truth) ** 2, axis=1))
        axes[2].plot(time_us, error, label=method, color=color)
    axes[2].set_ylabel("state RMSE")
    axes[2].set_xlabel("time [us]")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    figure.suptitle("Blind E20/E30 to E25 interpolation validation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_gate(
    path: Path,
    series: dict[str, dict],
    arrays: dict[str, np.ndarray],
    best_method: str,
    expert_fields: tuple[int, ...],
) -> None:
    payload = series[best_method]
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 6.8), sharex=True)
    for index, field in enumerate(expert_fields):
        axes[0].plot(
            arrays["time_us"], payload["weights"][:, index], label=f"E{field} expert"
        )
        axes[1].plot(
            arrays["time_us"], payload["distances"][:, index], label=f"E{field} distance"
        )
    axes[0].set_ylabel("gate weight")
    axes[1].set_ylabel("manifold distance")
    axes[1].set_xlabel("time [us]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"Selected gate on held-out E25: {best_method}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(
    path: Path,
    metrics: list[dict],
    best: dict,
    selected_by_field: dict[int, dict],
    development_status: str,
) -> None:
    persistence = next(row for row in metrics if row["method"] == "persistence")
    oracle = next(row for row in metrics if row["method"] == "oracle_local_e25")
    lines = [
        "# RadAz state-conditioned parametric ROM (data-only baseline)",
        "",
        f"Development status: **{development_status}**",
        "",
        "## Protocol",
        "",
        "- Shared state: `L+R+T` (20 frozen-SimVP latent + 8 radial envelopes + 2 modal transports).",
        "- Development interpolation: fit local experts at E20/E30 and forecast the held-out E25 trajectory over 24--30 us.",
        "- The held-out E25 history before 24 us initializes the delay state, but no E25 transition pair is used by the parametric experts.",
        "- Gate selection uses the E25 development result.  The selected protocol is then refitted on E20/E25/E30 and locked before any E22.5 sweep is loaded.",
        "- Ez=22.5 up/down sweep data used for fitting or selection: **no**.",
        "",
        "## Selected local expert hyperparameters",
        "",
        "| Ez [kV/m] | delay | history [us] | rank | validation skill | spectral radius |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for field, row in sorted(selected_by_field.items()):
        lines.append(
            f"| {field} | {int(row['delay'])} | {float(row['history_us']):.3f} | {int(row['rank'])} | {float(row['validation_skill_vs_persistence']):.4f} | {float(row['spectral_radius']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## E25 interpolation result",
            "",
            "| model | state skill | state corr | radial anomaly corr | transport anomaly corr | selection score |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in (persistence, best, oracle):
        lines.append(
            f"| {row['method']} | {float(row['state_skill_vs_persistence']):.4f} | {float(row['state_correlation']):.4f} | {float(row['radial_temporal_anomaly_correlation']):.4f} | {float(row['transport_temporal_anomaly_correlation']):.4f} | {float(row['selection_score']):.4f} |"
        )
    lines.extend(
        [
            "",
            "The oracle local-E25 model is an upper-bound diagnostic, not a zero-shot model.  The scientifically relevant baseline is the selected E20/E30 parametric interpolation relative to persistence.",
            "",
            "## Locked next test",
            "",
            "Apply the saved E20/E25/E30 model without retuning to both `E20 -> E22.5` and `E25 -> E22.5`.  Compare not only field error but also ECDI/MTSI order parameters, modal transport, the 1.49-us modulation, and the separation between the two histories.",
            "",
            "Physics-informed training is the next ablation.  It must retain this exact data split and gate lock so that any improvement can be attributed to the physics loss rather than target-data tuning.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a common-coordinate state-conditioned parametric RadAz ROM."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fields", default="20,25,30")
    parser.add_argument("--delays", default="20,40,80")
    parser.add_argument("--ranks", default="12,20,30,40")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = tuple(int(value) for value in args.fields.split(","))
    delays = tuple(int(value) for value in args.delays.split(","))
    ranks = tuple(int(value) for value in args.ranks.split(","))
    if fields != DEFAULT_FIELDS:
        raise ValueError(
            f"Protocol lock currently requires fields={DEFAULT_FIELDS}, got {fields}"
        )

    print("[1/6] Loading E20/E25/E30 stationary cases", flush=True)
    cases = {field: load_case(field) for field in fields}

    development_fields = tuple(
        field for field in fields if field != VALIDATION_FIELD
    )
    print(
        "[2/6] Fitting E20/E30-only common L+R+T representation",
        flush=True,
    )
    development_representation = fit_shared_representation(
        cases,
        development_fields,
        FIT_START_US,
        FINAL_FIT_END_US,
    )

    selection_rows: list[dict] = []
    selected_development: dict[int, dict] = {}
    for field in development_fields:
        print(f"[3/6] Selecting local expert at E{field}", flush=True)
        selected, rows = select_expert_hyperparameters(
            field,
            cases[field],
            development_representation.states[field],
            delays,
            ranks,
        )
        selected_development[field] = selected
        selection_rows.extend(
            {"phase": "development_leave_e25", **row} for row in rows
        )

    print("[4/6] Blind E20/E30 -> E25 interpolation validation", flush=True)
    validation_metrics, validation_series, validation_arrays, _ = (
        validate_e25_interpolation(
            cases,
            development_representation,
            selected_development,
        )
    )
    best = best_gate_row(validation_metrics)
    accepted = bool(
        float(best["finite_fraction"]) == 1.0
        and float(best["state_skill_vs_persistence"]) > 0.0
        and float(best["transport_skill_vs_persistence"]) > 0.0
    )
    development_status = (
        "ACCEPTED_DEVELOPMENT" if accepted else "REJECTED_DEVELOPMENT"
    )
    write_csv(output / "validation_e25_metrics.csv", validation_metrics)
    save_validation_h5(
        output / "validation_e25_rollouts.h5",
        development_representation,
        validation_series,
        validation_arrays,
    )
    plot_validation(
        output / "validation_e25_rollout.png",
        development_representation,
        validation_series,
        validation_arrays,
        best["method"],
    )
    plot_gate(
        output / "validation_e25_gate.png",
        validation_series,
        validation_arrays,
        best["method"],
        development_fields,
    )

    print(
        f"[5/6] Refitting locked model on E20/E25/E30; gate={best['method']}",
        flush=True,
    )
    final_representation = fit_shared_representation(
        cases, fields, FIT_START_US, FINAL_FIT_END_US
    )
    selected_final: dict[int, dict] = {}
    for field in fields:
        selected, rows = select_expert_hyperparameters(
            field,
            cases[field],
            final_representation.states[field],
            delays,
            ranks,
        )
        selected_final[field] = selected
        selection_rows.extend({"phase": "final_fit", **row} for row in rows)
    write_csv(output / "expert_selection.csv", selection_rows)
    write_csv(output / "common_latent_pca.csv", final_representation.pca_rows)
    final_experts = fit_final_experts(
        cases, final_representation, selected_final
    )
    model_path = output / "parametric_rom_data_only.h5"
    save_locked_model(model_path, final_representation, final_experts, best)
    with h5py.File(model_path, "r+") as handle:
        handle.attrs["development_status"] = development_status
        handle.attrs["accepted_for_blind_e22p5"] = accepted

    script_path = Path(__file__).resolve()
    protocol = {
        "status": development_status,
        "accepted_for_blind_e22p5": accepted,
        "state": STATE_LABEL,
        "state_dimensions": 30,
        "training_fields_kvm": fields,
        "development_heldout_field_kvm": VALIDATION_FIELD,
        "development_source_fields_kvm": development_fields,
        "selection_intervals_us": {
            "subtrain": [FIT_START_US, SELECTION_SPLIT_US],
            "validation": [SELECTION_SPLIT_US, SELECTION_END_US],
        },
        "final_expert_fit_interval_us": [FIT_START_US, FINAL_FIT_END_US],
        "development_forecast_interval_us": [
            FINAL_FIT_END_US,
            FORECAST_END_US,
        ],
        "selected_gate": best,
        "selected_final_experts": selected_final,
        "target_sweep": {
            "electric_field_kvm": 22.5,
            "directions": ["20_to_22.5", "25_to_22.5"],
            "data_loaded_for_fit": False,
            "data_loaded_for_selection": False,
            "allowed_use": (
                "locked blind evaluation"
                if accepted
                else "negative baseline only; protocol revision required"
            ),
        },
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "script_path": str(script_path),
        "script_sha256": sha256(script_path),
        "source_files": {
            str(field): {
                "features": str(cases[field].feature_path),
                "physical": str(cases[field].physical_path),
            }
            for field in fields
        },
    }
    (output / "model_lock.json").write_text(
        json.dumps(json_safe(protocol), indent=2), encoding="utf-8"
    )
    write_readme(
        output / "README.md",
        validation_metrics,
        best,
        selected_final,
        development_status,
    )
    summary = {
        "status": development_status,
        "accepted_for_blind_e22p5": accepted,
        "selected_gate": best,
        "validation_metrics": validation_metrics,
        "selected_final_experts": selected_final,
        "model_lock": str(output / "model_lock.json"),
        "target_22p5_data_used": False,
    }
    (output / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    print(
        f"[6/6] {development_status}: wrote data-only parametric ROM "
        f"without E22.5 data at {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
