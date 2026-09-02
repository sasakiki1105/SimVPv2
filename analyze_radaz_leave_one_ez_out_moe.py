from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_radaz_leave_one_ez_out as leaveout
import analyze_radaz_leave_one_ez_out_envelope as envelope
import analyze_radaz_physical_carrier_envelope as base


OUTPUT_DIR = (
    base.SIMVP_ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_leave_one_ez_out_moe"
)
GATING_METHODS = (
    "nearest_ez",
    "ez_weighted",
    "state_gated",
    "hybrid_gated",
)
CARRIER_SOURCES = envelope.CARRIER_SOURCES
EZ_BANDWIDTH_KVM = 10.0
STATE_TEMPERATURE = 0.08
STABILITY_LIMIT = 1.002


@dataclass
class LocalExpert:
    electric_field_kvm: int
    mean: np.ndarray
    basis: np.ndarray
    weights: np.ndarray
    rank: int
    ridge: float
    spectral_radius: float


def softmax(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(len(values), 1.0 / len(values))
    shifted = values - np.max(values[finite])
    exponent = np.zeros_like(values, dtype=np.float64)
    exponent[finite] = np.exp(np.clip(shifted[finite], -700.0, 0.0))
    total = float(np.sum(exponent))
    if total <= np.finfo(float).tiny:
        return np.full(len(values), 1.0 / len(values))
    return exponent / total


def fit_expert_from_basis(
    states: np.ndarray,
    electric_field_kvm: int,
    mean: np.ndarray,
    full_basis: np.ndarray,
    rank: int,
    ridge: float,
) -> LocalExpert:
    effective_rank = min(rank, full_basis.shape[1], len(states) - 1)
    if effective_rank < 1:
        raise ValueError("Local expert has zero effective rank")
    basis = full_basis[:, :effective_rank]
    latent = (states - mean) @ basis
    features = np.column_stack(
        (latent[:-1], np.ones(len(latent) - 1, dtype=np.float64))
    )
    targets = latent[1:]
    gram = features.T @ features
    ridge_scale = max(
        float(np.mean(np.diag(gram))), np.finfo(float).tiny
    )
    regularization = ridge * ridge_scale * np.eye(
        gram.shape[0], dtype=np.float64
    )
    regularization[-1, -1] = 0.0
    weights = np.linalg.pinv(
        gram + regularization, rcond=1.0e-10
    ) @ (features.T @ targets)
    eigenvalues = np.linalg.eigvals(weights[:effective_rank])
    radius = float(np.max(np.abs(eigenvalues)))
    return LocalExpert(
        electric_field_kvm=electric_field_kvm,
        mean=mean,
        basis=basis,
        weights=weights,
        rank=effective_rank,
        ridge=ridge,
        spectral_radius=radius,
    )


def fit_local_expert(
    states: np.ndarray,
    electric_field_kvm: int,
    rank: int,
    ridge: float,
) -> LocalExpert:
    mean = np.mean(states, axis=0)
    centered = states - mean
    _, singular_values, vt = np.linalg.svd(
        centered, full_matrices=False
    )
    nonzero = int(np.count_nonzero(singular_values > 1.0e-10))
    if nonzero < 1:
        raise ValueError("Local expert training states have zero rank")
    return fit_expert_from_basis(
        states,
        electric_field_kvm,
        mean,
        vt[:nonzero].T,
        rank,
        ridge,
    )


def step_expert(expert: LocalExpert, state: np.ndarray) -> np.ndarray:
    latent = (state - expert.mean) @ expert.basis
    feature = np.concatenate((latent, np.ones(1, dtype=np.float64)))
    following = feature @ expert.weights
    return expert.mean + following @ expert.basis.T


def rollout_single(
    expert: LocalExpert, initial_state: np.ndarray, steps: int
) -> np.ndarray:
    state = np.asarray(initial_state, dtype=np.float64)
    prediction = np.empty((steps, len(state)), dtype=np.float64)
    for index in range(steps):
        state = step_expert(expert, state)
        if (
            not np.all(np.isfinite(state))
            or np.max(np.abs(state)) > 1.0e8
        ):
            prediction[index:] = np.nan
            break
        prediction[index] = state
    return prediction


def select_local_expert(
    case: leaveout.StateCase,
) -> tuple[dict, list[dict]]:
    subtrain_mask = leaveout.interval_mask(
        case.raw.time_us,
        base.FIT_START_US,
        base.VALIDATION_START_US,
    ) & (case.raw.time_us < base.VALIDATION_START_US - 1.0e-10)
    validation_mask = leaveout.interval_mask(
        case.raw.time_us,
        base.VALIDATION_START_US,
        base.FIT_END_US,
    )
    subtrain = case.states[subtrain_mask]
    validation = case.states[validation_mask]
    initial = subtrain[-1]

    mean = np.mean(subtrain, axis=0)
    centered = subtrain - mean
    _, singular_values, vt = np.linalg.svd(
        centered, full_matrices=False
    )
    nonzero = int(np.count_nonzero(singular_values > 1.0e-10))
    full_basis = vt[:nonzero].T

    trials = []
    best = {"objective": float("inf")}
    for rank in leaveout.RANK_CANDIDATES:
        for ridge in leaveout.RIDGE_CANDIDATES:
            try:
                model = fit_expert_from_basis(
                    subtrain,
                    case.raw.electric_field_kvm,
                    mean,
                    full_basis,
                    rank,
                    ridge,
                )
                prediction = rollout_single(
                    model, initial, len(validation)
                )
                validation_mse = leaveout.prediction_mse(
                    validation, prediction
                )
                objective = validation_mse + max(
                    0.0, model.spectral_radius - STABILITY_LIMIT
                ) * 100.0
                effective_rank = model.rank
                radius = model.spectral_radius
            except (ValueError, np.linalg.LinAlgError):
                validation_mse = float("inf")
                objective = float("inf")
                effective_rank = 0
                radius = float("inf")
            row = {
                "expert_electric_field_kvm": (
                    case.raw.electric_field_kvm
                ),
                "requested_rank": rank,
                "effective_rank": effective_rank,
                "ridge": ridge,
                "validation_mse": validation_mse,
                "spectral_radius": radius,
                "objective": objective,
            }
            trials.append(row)
            if objective < best["objective"]:
                best = row
    if not np.isfinite(best["objective"]):
        raise RuntimeError(
            f"No valid expert for E={case.raw.electric_field_kvm}"
        )
    return best, trials


def ez_prior(
    experts: list[LocalExpert],
    target_electric_field_kvm: int,
    method: str,
) -> np.ndarray:
    fields = np.asarray(
        [expert.electric_field_kvm for expert in experts],
        dtype=np.float64,
    )
    if method == "nearest_ez":
        weights = np.zeros(len(experts), dtype=np.float64)
        weights[int(np.argmin(np.abs(fields - target_electric_field_kvm)))] = (
            1.0
        )
        return weights
    log_weights = -0.5 * (
        (fields - target_electric_field_kvm) / EZ_BANDWIDTH_KVM
    ) ** 2
    return softmax(log_weights)


def state_distances(
    experts: list[LocalExpert], state: np.ndarray
) -> np.ndarray:
    distances = []
    for expert in experts:
        centered = state - expert.mean
        projected = (centered @ expert.basis) @ expert.basis.T
        residual = centered - projected
        denominator = max(
            float(np.mean(centered**2)), np.finfo(float).tiny
        )
        distances.append(float(np.mean(residual**2)) / denominator)
    return np.asarray(distances, dtype=np.float64)


def gating_weights(
    experts: list[LocalExpert],
    state: np.ndarray,
    target_electric_field_kvm: int,
    method: str,
) -> np.ndarray:
    if method == "nearest_ez":
        return ez_prior(experts, target_electric_field_kvm, method)
    if method == "ez_weighted":
        return ez_prior(experts, target_electric_field_kvm, method)
    distances = state_distances(experts, state)
    state_log_weights = -distances / STATE_TEMPERATURE
    if method == "state_gated":
        return softmax(state_log_weights)
    if method == "hybrid_gated":
        prior = ez_prior(
            experts, target_electric_field_kvm, "ez_weighted"
        )
        return softmax(
            np.log(np.maximum(prior, np.finfo(float).tiny))
            + state_log_weights
        )
    raise ValueError(f"Unknown gating method: {method}")


def rollout_moe(
    experts: list[LocalExpert],
    initial_state: np.ndarray,
    target_electric_field_kvm: int,
    method: str,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(initial_state, dtype=np.float64)
    prediction = np.empty((steps, len(state)), dtype=np.float64)
    weights_over_time = np.empty(
        (steps, len(experts)), dtype=np.float64
    )
    for index in range(steps):
        weights = gating_weights(
            experts, state, target_electric_field_kvm, method
        )
        candidates = np.stack(
            [step_expert(expert, state) for expert in experts],
            axis=0,
        )
        state = np.einsum("e,ed->d", weights, candidates)
        weights_over_time[index] = weights
        if (
            not np.all(np.isfinite(state))
            or np.max(np.abs(state)) > 1.0e8
        ):
            prediction[index:] = np.nan
            weights_over_time[index + 1 :] = np.nan
            break
        prediction[index] = state
    return prediction, weights_over_time


def gate_summary(
    experts: list[LocalExpert], weights: np.ndarray
) -> dict:
    finite = np.isfinite(weights).all(axis=1)
    valid = weights[finite]
    fields = [expert.electric_field_kvm for expert in experts]
    if len(valid) == 0:
        result = {
            "gate_entropy_normalized": float("nan"),
            "gate_switch_count": 0,
            "gate_initial_dominant_electric_field_kvm": -1,
            "gate_final_dominant_electric_field_kvm": -1,
        }
        for field in fields:
            result[f"mean_weight_e{field}"] = float("nan")
            result[f"dominant_fraction_e{field}"] = float("nan")
        return result
    entropy = -np.sum(
        valid * np.log(np.maximum(valid, np.finfo(float).tiny)),
        axis=1,
    )
    if len(experts) > 1:
        entropy /= np.log(len(experts))
    dominant = np.argmax(valid, axis=1)
    result = {
        "gate_entropy_normalized": float(np.mean(entropy)),
        "gate_switch_count": int(np.count_nonzero(np.diff(dominant))),
        "gate_initial_dominant_electric_field_kvm": fields[
            int(dominant[0])
        ],
        "gate_final_dominant_electric_field_kvm": fields[
            int(dominant[-1])
        ],
    }
    for index, field in enumerate(fields):
        result[f"mean_weight_e{field}"] = float(
            np.mean(valid[:, index])
        )
        result[f"dominant_fraction_e{field}"] = float(
            np.mean(dominant == index)
        )
    return result


def previous_shared_reference(
    heldout: int,
    variant: str,
    carrier_source: str,
) -> tuple[float, float]:
    path = (
        envelope.OUTPUT_DIR
        / "leave_one_ez_out_envelope_metrics.csv"
    )
    if not path.exists():
        return float("nan"), float("nan")
    matches = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["heldout_electric_field_kvm"]) == heldout
                and row["variant"] == variant
                and row["carrier_source"] == carrier_source
            ):
                matches.append(row)
    if not matches:
        return float("nan"), float("nan")

    def best(key: str) -> float:
        values = []
        for row in matches:
            try:
                value = float(row[key])
            except (KeyError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return max(values) if values else float("nan")

    return best("envelope_state_correlation"), best(
        "raw_fourier_state_correlation"
    )


def evaluate_fold(
    raw_cases: dict[int, leaveout.RawCase],
    heldout: int,
    variant: str,
    final_scales: leaveout.FoldScales,
    final_angles: dict[int, np.ndarray],
    predicted_angle: np.ndarray,
    oracle_angle: np.ndarray,
    selected_by_field: dict[int, dict],
) -> tuple[list[dict], dict, dict]:
    training_fields = tuple(
        value for value in base.ELECTRIC_FIELDS if value != heldout
    )
    training_cases = {}
    experts = []
    for electric_field_kvm in training_fields:
        demodulated = envelope.demodulate_raw(
            raw_cases[electric_field_kvm],
            final_angles[electric_field_kvm],
        )
        case = leaveout.build_state_case(
            demodulated, final_scales, variant
        )
        training_cases[electric_field_kvm] = case
        fit_mask = leaveout.interval_mask(
            case.raw.time_us,
            base.FIT_START_US,
            base.FIT_END_US,
        )
        selected = selected_by_field[electric_field_kvm]
        experts.append(
            fit_local_expert(
                case.states[fit_mask],
                electric_field_kvm,
                int(selected["requested_rank"]),
                float(selected["ridge"]),
            )
        )
    experts.sort(key=lambda item: item.electric_field_kvm)

    metric_rows = []
    series = {}
    weight_data = {}
    for carrier_source, heldout_angle in (
        ("predicted_from_training_ez", predicted_angle),
        ("oracle_diagnostic", oracle_angle),
    ):
        envelope_raw = envelope.demodulate_raw(
            raw_cases[heldout], heldout_angle
        )
        envelope_case = leaveout.build_state_case(
            envelope_raw, final_scales, variant
        )
        raw_case = leaveout.build_state_case(
            raw_cases[heldout], final_scales, variant
        )
        fit_mask = leaveout.interval_mask(
            envelope_case.raw.time_us,
            base.FIT_START_US,
            base.FIT_END_US,
        )
        holdout_mask = leaveout.interval_mask(
            envelope_case.raw.time_us,
            base.FIT_END_US,
            base.HOLDOUT_END_US,
            include_start=False,
        )
        holdout_frames = envelope_case.raw.frame[holdout_mask]
        time_us = envelope_case.raw.time_us[holdout_mask]
        envelope_truth = envelope_case.states[holdout_mask]
        initial = envelope_case.states[np.flatnonzero(fit_mask)[-1]]
        constant_envelope = np.repeat(
            initial[None, :], len(envelope_truth), axis=0
        )
        raw_truth = raw_case.states[holdout_mask]
        carrier_baseline = envelope.remodulate_states(
            envelope_case,
            constant_envelope,
            holdout_frames,
            heldout_angle,
        )
        raw_persistence = envelope.raw_persistence(
            raw_case, fit_mask, len(raw_truth)
        )
        reference_envelope, reference_raw = previous_shared_reference(
            heldout, variant, carrier_source
        )

        for method in GATING_METHODS:
            prediction, weights = rollout_moe(
                experts,
                initial,
                heldout,
                method,
                len(envelope_truth),
            )
            truth_gate_states = np.vstack(
                (initial[None, :], envelope_truth[:-1])
            )
            truth_state_weights = np.stack(
                [
                    gating_weights(
                        experts, state, heldout, method
                    )
                    for state in truth_gate_states
                ],
                axis=0,
            )
            envelope_metrics, _ = leaveout.evaluate_prediction(
                envelope_case,
                envelope_truth,
                prediction,
                constant_envelope,
                time_us,
            )
            raw_prediction = envelope.remodulate_states(
                envelope_case,
                prediction,
                holdout_frames,
                heldout_angle,
            )
            raw_carrier_metrics, raw_series = (
                leaveout.evaluate_prediction(
                    raw_case,
                    raw_truth,
                    raw_prediction,
                    carrier_baseline,
                    time_us,
                )
            )
            raw_persistence_metrics, _ = leaveout.evaluate_prediction(
                raw_case,
                raw_truth,
                raw_prediction,
                raw_persistence,
                time_us,
            )
            gate = gate_summary(experts, weights)
            truth_gate = {
                f"truth_state_{key}": value
                for key, value in gate_summary(
                    experts, truth_state_weights
                ).items()
            }
            metric_rows.append(
                {
                    "heldout_electric_field_kvm": heldout,
                    "training_fields_kvm": ",".join(
                        map(str, training_fields)
                    ),
                    "variant": variant,
                    "gating_method": method,
                    "carrier_source": carrier_source,
                    "state_dimensions": envelope_case.states.shape[1],
                    "expert_fields_kvm": ",".join(
                        str(expert.electric_field_kvm)
                        for expert in experts
                    ),
                    "expert_ranks": ",".join(
                        str(expert.rank) for expert in experts
                    ),
                    "expert_ridges": ",".join(
                        f"{expert.ridge:.1e}" for expert in experts
                    ),
                    "maximum_expert_spectral_radius": max(
                        expert.spectral_radius for expert in experts
                    ),
                    "envelope_state_correlation": envelope_metrics[
                        "fourier_state_correlation"
                    ],
                    "envelope_state_skill_vs_constant_envelope": (
                        envelope_metrics[
                            "fourier_state_skill_vs_persistence"
                        ]
                    ),
                    "raw_fourier_state_correlation": (
                        raw_carrier_metrics[
                            "fourier_state_correlation"
                        ]
                    ),
                    "raw_fourier_skill_vs_carrier_baseline": (
                        raw_carrier_metrics[
                            "fourier_state_skill_vs_persistence"
                        ]
                    ),
                    "raw_fourier_skill_vs_persistence": (
                        raw_persistence_metrics[
                            "fourier_state_skill_vs_persistence"
                        ]
                    ),
                    "cross_phase_mae_rad": raw_carrier_metrics[
                        "cross_phase_mae_rad"
                    ],
                    "transport_correlation": raw_carrier_metrics[
                        "transport_correlation"
                    ],
                    "transport_skill_vs_persistence": (
                        raw_persistence_metrics[
                            "transport_skill_vs_persistence"
                        ]
                    ),
                    "transport_consistency_correlation": (
                        raw_carrier_metrics[
                            "transport_consistency_correlation"
                        ]
                    ),
                    "axial_current_correlation": raw_carrier_metrics[
                        "axial_current_correlation"
                    ],
                    "finite_fraction": raw_carrier_metrics[
                        "finite_fraction"
                    ],
                    "previous_best_shared_envelope_correlation": (
                        reference_envelope
                    ),
                    "previous_best_shared_raw_correlation": reference_raw,
                    **gate,
                    **truth_gate,
                }
            )
            key = f"{carrier_source}/{method}"
            series[key] = {
                **raw_series,
                "expert_fields_kvm": np.asarray(
                    [
                        expert.electric_field_kvm
                        for expert in experts
                    ],
                    dtype=np.int64,
                ),
                "weights": weights,
                "truth_state_weights": truth_state_weights,
            }
            weight_data[key] = {
                "expert_fields_kvm": np.asarray(
                    [
                        expert.electric_field_kvm
                        for expert in experts
                    ],
                    dtype=np.int64,
                ),
                "weights": weights,
            }
    return metric_rows, series, weight_data


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predicted_rows(rows: list[dict], variant: str) -> list[dict]:
    return [
        row
        for row in rows
        if row["variant"] == variant
        and row["carrier_source"] == "predicted_from_training_ez"
    ]


def plot_metrics(rows: list[dict], outdir: Path) -> None:
    colors = {
        "nearest_ez": "#0072B2",
        "ez_weighted": "#E69F00",
        "state_gated": "#009E73",
        "hybrid_gated": "#CC79A7",
    }
    labels = {
        "nearest_ez": "Nearest Ez expert",
        "ez_weighted": "Ez-weighted experts",
        "state_gated": "State-gated experts",
        "hybrid_gated": "Hybrid gate",
    }
    metrics = (
        ("envelope_state_correlation", "Envelope Fourier correlation"),
        ("raw_fourier_state_correlation", "Raw Fourier correlation"),
        ("transport_correlation", "Transport correlation"),
    )
    figure, axes = plt.subplots(
        len(metrics),
        len(leaveout.VARIANTS),
        figsize=(12.4, 10.0),
        sharex=True,
    )
    x = np.asarray(base.ELECTRIC_FIELDS)
    for column, variant in enumerate(leaveout.VARIANTS):
        current = predicted_rows(rows, variant)
        for row_index, (metric, title) in enumerate(metrics):
            axis = axes[row_index, column]
            for method in GATING_METHODS:
                values = [
                    next(
                        float(row[metric])
                        for row in current
                        if row["gating_method"] == method
                        and int(row["heldout_electric_field_kvm"])
                        == heldout
                    )
                    for heldout in base.ELECTRIC_FIELDS
                ]
                axis.plot(
                    x,
                    values,
                    marker="o",
                    linewidth=1.7,
                    color=colors[method],
                    label=labels[method],
                )
            if metric in (
                "envelope_state_correlation",
                "raw_fourier_state_correlation",
            ):
                reference_key = (
                    "previous_best_shared_envelope_correlation"
                    if metric == "envelope_state_correlation"
                    else "previous_best_shared_raw_correlation"
                )
                reference = [
                    next(
                        float(row[reference_key])
                        for row in current
                        if int(row["heldout_electric_field_kvm"])
                        == heldout
                    )
                    for heldout in base.ELECTRIC_FIELDS
                ]
                axis.plot(
                    x,
                    reference,
                    linestyle="--",
                    color="#555555",
                    linewidth=1.5,
                    label="Best previous shared model",
                )
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.set_title(
                f"{title}\n{variant.replace('_', ' ')}", fontsize=10
            )
            axis.set_xticks(x)
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right", fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Held-out Ez [kV/m]")
    figure.tight_layout()
    figure.savefig(
        outdir / "leave_one_ez_out_moe_metrics.png", dpi=180
    )
    plt.close(figure)


def plot_oracle_gap(rows: list[dict], outdir: Path) -> None:
    figure, axes = plt.subplots(
        len(leaveout.VARIANTS),
        1,
        figsize=(9.0, 7.2),
        sharex=True,
    )
    x = np.asarray(base.ELECTRIC_FIELDS)
    for axis, variant in zip(axes, leaveout.VARIANTS):
        for method, color in zip(
            GATING_METHODS,
            ("#0072B2", "#E69F00", "#009E73", "#CC79A7"),
        ):
            predicted = []
            oracle = []
            for heldout in base.ELECTRIC_FIELDS:
                for source, target in (
                    ("predicted_from_training_ez", predicted),
                    ("oracle_diagnostic", oracle),
                ):
                    row = next(
                        item
                        for item in rows
                        if item["variant"] == variant
                        and item["gating_method"] == method
                        and item["carrier_source"] == source
                        and int(
                            item["heldout_electric_field_kvm"]
                        )
                        == heldout
                    )
                    target.append(
                        float(row["raw_fourier_state_correlation"])
                    )
            axis.plot(
                x,
                predicted,
                marker="o",
                color=color,
                label=f"{method}: predicted carrier",
            )
            axis.plot(
                x,
                oracle,
                marker="x",
                linestyle="--",
                color=color,
                label=f"{method}: oracle carrier",
            )
        axis.set_title(variant.replace("_", " "))
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("Held-out Ez [kV/m]")
    figure.supylabel("Raw Fourier correlation")
    figure.tight_layout()
    figure.savefig(
        outdir / "leave_one_ez_out_moe_oracle_carrier_gap.png",
        dpi=180,
    )
    plt.close(figure)


def plot_gate_weights(
    all_series: dict[tuple[int, str], dict],
    outdir: Path,
) -> None:
    figure, axes = plt.subplots(
        len(base.ELECTRIC_FIELDS),
        len(leaveout.VARIANTS),
        figsize=(12.0, 11.0),
        sharex=True,
        sharey=True,
    )
    palette = ("#0072B2", "#E69F00", "#009E73")
    for row_index, heldout in enumerate(base.ELECTRIC_FIELDS):
        for column, variant in enumerate(leaveout.VARIANTS):
            axis = axes[row_index, column]
            payload = all_series[(heldout, variant)][
                "predicted_from_training_ez/hybrid_gated"
            ]
            fields = payload["expert_fields_kvm"]
            time_us = payload["time_us"]
            weights = payload["weights"]
            truth_weights = payload["truth_state_weights"]
            for index, field in enumerate(fields):
                axis.plot(
                    time_us,
                    weights[:, index],
                    color=palette[index],
                    linewidth=1.4,
                    label=f"E{field} rollout",
                )
                axis.plot(
                    time_us,
                    truth_weights[:, index],
                    color=palette[index],
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.8,
                    label=f"E{field} truth-state gate",
                )
            axis.set_title(
                f"held-out E{heldout}, {variant.replace('_', ' ')}",
                fontsize=9,
            )
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right", fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Time [us]")
    for axis in axes[:, 0]:
        axis.set_ylabel("Hybrid gate weight")
    figure.tight_layout()
    figure.savefig(
        outdir / "leave_one_ez_out_moe_hybrid_gate_weights.png",
        dpi=180,
    )
    plt.close(figure)


def save_h5(
    outdir: Path, all_series: dict[tuple[int, str], dict]
) -> None:
    path = outdir / "leave_one_ez_out_moe_rollouts.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["fit_interval_us"] = [
            base.FIT_START_US,
            base.FIT_END_US,
        ]
        handle.attrs["holdout_interval_us"] = [
            base.FIT_END_US,
            base.HOLDOUT_END_US,
        ]
        for (heldout, variant), combinations in all_series.items():
            root = handle.require_group(
                f"heldout_e{heldout}/{variant}"
            )
            for key, payload in combinations.items():
                group = root.require_group(key)
                for name, values in payload.items():
                    group.create_dataset(
                        name, data=values, compression="gzip"
                    )


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def best_rows(rows: list[dict]) -> list[dict]:
    result = []
    for heldout in base.ELECTRIC_FIELDS:
        for variant in leaveout.VARIANTS:
            candidates = [
                row
                for row in rows
                if int(row["heldout_electric_field_kvm"]) == heldout
                and row["variant"] == variant
                and row["carrier_source"]
                == "predicted_from_training_ez"
                and np.isfinite(
                    float(row["raw_fourier_state_correlation"])
                )
            ]
            if candidates:
                result.append(
                    max(
                        candidates,
                        key=lambda row: float(
                            row["raw_fourier_state_correlation"]
                        ),
                    )
                )
    return result


def generate_readme(outdir: Path, rows: list[dict]) -> None:
    lines = [
        "# Leave-one-Ez-out regime-switching / MoE analysis",
        "",
        "## 日本語",
        "",
        "carrier-envelope共通線形モデルが未学習Ezへ転用できなかったため、"
        "学習済みの各Ezに局所線形expertを1個ずつ作り、予測中に混合・切替する"
        "regime-switching（mixture-of-experts; MoE）を検証した。",
        "",
        "- 各foldでは対象Ezを完全に除外した。",
        "- expertのrankとridgeは各学習Ezの20–23 usで同定し、23–24 usで選択した。",
        "- 選択後は20–24 usで再同定し、held-out Ezの24 usの状態だけから"
        "24–30 usを自律予測した。",
        "- primary評価のcarrierは学習Ezから予測した。oracle carrierは原因切り分け専用である。",
        "- gateはnearest Ez、Ez距離混合、状態距離、Ez+状態距離の4種類を比較した。",
        "",
        "## Best primary result in each fold",
        "",
        "| held-out Ez | variant | gate | envelope corr | raw corr | transport corr | previous shared raw corr |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in best_rows(rows):
        lines.append(
            "| {heldout} | {variant} | {gate} | {env:.4f} | "
            "{raw:.4f} | {transport:.4f} | {reference:.4f} |".format(
                heldout=row["heldout_electric_field_kvm"],
                variant=row["variant"],
                gate=row["gating_method"],
                env=float(row["envelope_state_correlation"]),
                raw=float(row["raw_fourier_state_correlation"]),
                transport=float(row["transport_correlation"]),
                reference=float(
                    row["previous_best_shared_raw_correlation"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The best primary raw-Fourier correlation is only about 0.008. "
            "Local experts therefore do not recover the held-out autonomous "
            "dynamics. Ez-weighted mixing is usually better than state gating, "
            "but the improvement over the previous shared model is negligible "
            "and is not consistent across folds.",
            "",
            "The rollout-based state and hybrid gates become almost one-hot "
            "after the first few steps and then remain on one expert. The "
            "truth-state diagnostic is more variable, especially for held-out "
            "E20, showing that the predicted-state gate creates a "
            "self-reinforcing expert lock-in. This diagnostic uses future truth "
            "and is not a zero-shot prediction.",
            "",
            "With the oracle carrier, held-out E40 reaches a raw-Fourier "
            "correlation of about 0.11, while the other folds remain near zero. "
            "Carrier error matters at E40 but does not explain the overall "
            "failure. Transport correlations around 0.10--0.15 at E10/E20 are "
            "not sufficient evidence of closure because the modal state "
            "correlation is essentially zero.",
            "",
            "## Files",
            "",
            "- `leave_one_ez_out_moe_metrics.csv`: 全fold・gate・carrier診断の指標",
            "- `leave_one_ez_out_moe_expert_selection.csv`: 局所expertのrank/ridge選択",
            "- `leave_one_ez_out_moe_metrics.png`: primary評価と以前の共通モデルの比較",
            "- `leave_one_ez_out_moe_oracle_carrier_gap.png`: carrier誤差の寄与",
            "- `leave_one_ez_out_moe_hybrid_gate_weights.png`: hybrid gateの予測状態・真値状態診断",
            "- `leave_one_ez_out_moe_rollouts.h5`: gate重みと輸送時系列",
            "",
            "## English summary",
            "",
            "This analysis tests local carrier-envelope linear experts with "
            "parameter, state, and hybrid gating under strict leave-one-Ez-out "
            "evaluation. The held-out future is not used for model selection.",
        ]
    )
    (outdir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-Ez-out carrier-envelope mixture-of-experts analysis."
        )
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    raw_cases = {
        electric_field_kvm: leaveout.extract_raw_case(
            electric_field_kvm
        )
        for electric_field_kvm in base.ELECTRIC_FIELDS
    }
    for electric_field_kvm in base.ELECTRIC_FIELDS:
        print(f"E{electric_field_kvm}: extracted common-mode fields")

    metric_rows = []
    selection_rows = []
    all_series = {}
    carrier_rows = []
    for heldout in base.ELECTRIC_FIELDS:
        training_fields = tuple(
            value
            for value in base.ELECTRIC_FIELDS
            if value != heldout
        )
        selection_scales = leaveout.compute_fold_scales(
            raw_cases,
            training_fields,
            base.FIT_START_US,
            base.VALIDATION_START_US,
        )
        final_scales = leaveout.compute_fold_scales(
            raw_cases,
            training_fields,
            base.FIT_START_US,
            base.FIT_END_US,
        )
        selection_angles = {
            field: envelope.estimate_carrier_angles(
                raw_cases[field],
                selection_scales,
                base.FIT_START_US,
                base.VALIDATION_START_US,
            )
            for field in training_fields
        }
        final_angles = {
            field: envelope.estimate_carrier_angles(
                raw_cases[field],
                final_scales,
                base.FIT_START_US,
                base.FIT_END_US,
            )
            for field in training_fields
        }
        predicted_angle = envelope.predict_carrier_angles(
            final_angles, heldout
        )
        oracle_angle = envelope.estimate_carrier_angles(
            raw_cases[heldout],
            final_scales,
            base.FIT_START_US,
            base.FIT_END_US,
        )
        carrier_errors = envelope.carrier_error_metrics(
            raw_cases[heldout], predicted_angle, oracle_angle
        )
        carrier_rows.append(
            {
                "heldout_electric_field_kvm": heldout,
                "training_fields_kvm": ",".join(
                    map(str, training_fields)
                ),
                **carrier_errors,
            }
        )

        for variant in leaveout.VARIANTS:
            selection_cases = {}
            for field in training_fields:
                current = envelope.demodulate_raw(
                    raw_cases[field], selection_angles[field]
                )
                selection_cases[field] = leaveout.build_state_case(
                    current, selection_scales, variant
                )
            selected_by_field = {}
            for field in training_fields:
                selected, trials = select_local_expert(
                    selection_cases[field]
                )
                selected_by_field[field] = selected
                selection_rows.extend(
                    {
                        "heldout_electric_field_kvm": heldout,
                        "training_fields_kvm": ",".join(
                            map(str, training_fields)
                        ),
                        "variant": variant,
                        **row,
                    }
                    for row in trials
                )
                print(
                    f"leave E{heldout} {variant}: expert E{field} "
                    f"rank={selected['effective_rank']} "
                    f"ridge={selected['ridge']:.1e} "
                    f"val={selected['validation_mse']:.3e}"
                )

            current_rows, current_series, _ = evaluate_fold(
                raw_cases=raw_cases,
                heldout=heldout,
                variant=variant,
                final_scales=final_scales,
                final_angles=final_angles,
                predicted_angle=predicted_angle,
                oracle_angle=oracle_angle,
                selected_by_field=selected_by_field,
            )
            metric_rows.extend(current_rows)
            all_series[(heldout, variant)] = current_series
            primary = [
                row
                for row in current_rows
                if row["carrier_source"]
                == "predicted_from_training_ez"
            ]
            best = max(
                primary,
                key=lambda row: (
                    float(row["raw_fourier_state_correlation"])
                    if np.isfinite(
                        float(row["raw_fourier_state_correlation"])
                    )
                    else -np.inf
                ),
            )
            print(
                f"E{heldout} {variant}: best={best['gating_method']} "
                f"env={best['envelope_state_correlation']:.3f} "
                f"raw={best['raw_fourier_state_correlation']:.3f} "
                f"transport={best['transport_correlation']:.3f}"
            )

    write_csv(outdir / "leave_one_ez_out_moe_metrics.csv", metric_rows)
    write_csv(
        outdir / "leave_one_ez_out_moe_expert_selection.csv",
        selection_rows,
    )
    write_csv(
        outdir / "leave_one_ez_out_moe_carrier_errors.csv",
        carrier_rows,
    )
    plot_metrics(metric_rows, outdir)
    plot_oracle_gap(metric_rows, outdir)
    plot_gate_weights(all_series, outdir)
    save_h5(outdir, all_series)
    generate_readme(outdir, metric_rows)

    summary = {
        "status": "PASS",
        "definition": {
            "electric_fields_kvm": base.ELECTRIC_FIELDS,
            "variants": leaveout.VARIANTS,
            "gating_methods": GATING_METHODS,
            "ez_bandwidth_kvm": EZ_BANDWIDTH_KVM,
            "state_temperature": STATE_TEMPERATURE,
            "fit_interval_us": [
                base.FIT_START_US,
                base.FIT_END_US,
            ],
            "validation_interval_us": [
                base.VALIDATION_START_US,
                base.FIT_END_US,
            ],
            "holdout_interval_us": [
                base.FIT_END_US,
                base.HOLDOUT_END_US,
            ],
            "heldout_future_used_for_selection": False,
            "primary_carrier_source": "predicted_from_training_ez",
            "oracle_carrier_is_zero_shot": False,
        },
        "carrier_errors": carrier_rows,
        "best_primary_rows": best_rows(metric_rows),
        "metrics": metric_rows,
    }
    (
        outdir / "leave_one_ez_out_moe_summary.json"
    ).write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    print(f"PASS: wrote MoE leave-one-Ez-out analysis to {outdir}")


if __name__ == "__main__":
    main()
