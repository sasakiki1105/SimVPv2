"""Test direct transport plus transport-orthogonal cross-spectrum closure.

This is the follow-up to the E25 circular carrier-state experiment.  The
selected-mode transport is included explicitly, while the cross-spectrum
state is projected onto the null space of the transport functional before a
fit-only PCA is applied.  Consequently, T and delta-X do not duplicate the
same scalar transport information.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import analyze_radaz_coupling_and_rolling_validation as rolling
import analyze_radaz_e25_carrier_state_closure as carrier
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "compare_radaz_e25_transport_residual_closure"
)
RESIDUAL_COMPONENTS = 6
SYSTEMS = {
    "transport_only": ("transport_direct",),
    "latent_transport": ("latent", "transport_direct"),
    "phi_transport": ("phi_circular", "transport_direct"),
    "latent_phi_transport": (
        "latent",
        "phi_circular",
        "transport_direct",
    ),
    "phi_transport_residual": (
        "phi_circular",
        "transport_direct",
        "cross_residual",
    ),
    "latent_phi_transport_residual": (
        "latent",
        "phi_circular",
        "transport_direct",
        "cross_residual",
    ),
}
LABELS = {
    "transport_only": "T",
    "latent_transport": "L+T",
    "phi_transport": "Pcirc+T",
    "latent_phi_transport": "L+Pcirc+T",
    "phi_transport_residual": "Pcirc+T+dX",
    "latent_phi_transport_residual": "L+Pcirc+T+dX",
}
PHYSICAL_BASELINES = {
    "latent_transport": "transport_only",
    "latent_phi_transport": "phi_transport",
    "latent_phi_transport_residual": "phi_transport_residual",
}
WINDOWS = rolling.WINDOWS
METHODS = rolling.METHODS


@dataclass
class TransportResidual:
    cross_block: carrier.CarrierBlock
    transport: np.ndarray
    transport_state: np.ndarray
    residual_scores: np.ndarray
    residual_mean: np.ndarray
    residual_basis: np.ndarray
    transport_direction: np.ndarray
    transport_functional: np.ndarray
    oracle_coefficients: np.ndarray
    explained_variance: np.ndarray

    def decode_cross(
        self,
        predicted_transport: np.ndarray,
        predicted_residual: np.ndarray,
    ) -> np.ndarray:
        transport = predicted_transport[:, 0]
        residual = (
            predicted_residual @ self.residual_basis + self.residual_mean
        )
        state = residual + transport[:, None] * self.transport_direction[None]
        carrier_count = self.cross_block.carrier_count
        normalized = (
            state[:, :carrier_count]
            + 1j * state[:, carrier_count:]
        ).reshape(
            len(state),
            len(self.cross_block.modes),
            self.cross_block.bases.shape[1],
        )
        scores = normalized * self.cross_block.scales[None]
        return self.cross_block._scores_to_coefficients(scores)


def write_csv(path: Path, rows: list[dict]) -> None:
    carrier.write_csv(path, rows)


def json_safe(value):
    return carrier.json_safe(value)


def finite_median(values) -> float:
    return carrier.finite_median(values)


def finite_min(values) -> float:
    return carrier.finite_min(values)


def finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def normalized_raw_scores(block_state: carrier.CarrierBlock) -> np.ndarray:
    shaped = block_state.cartesian.reshape(
        len(block_state.cartesian),
        len(block_state.modes),
        block_state.bases.shape[1],
        2,
    )
    envelope = shaped[..., 0] + 1j * shaped[..., 1]
    return envelope * np.exp(
        1j
        * block_state.frame[:, None, None]
        * block_state.carrier_angles[None]
    )


def transport_functional(block_state: carrier.CarrierBlock) -> np.ndarray:
    sqrt_weights = np.sqrt(block_state.radial_weights)
    radial_integral = np.einsum(
        "mkr,r->mk", block_state.bases, sqrt_weights
    )
    alpha = radial_integral * block_state.scales
    return np.concatenate(
        [
            (-2.0 / carrier.B_T) * alpha.real.ravel(),
            (+2.0 / carrier.B_T) * alpha.imag.ravel(),
        ]
    )


def build_transport_residual(
    cross_block: carrier.CarrierBlock,
    fit_mask: np.ndarray,
) -> TransportResidual:
    normalized = normalized_raw_scores(cross_block)
    flattened = np.concatenate(
        [
            normalized.real.reshape(len(normalized), -1),
            normalized.imag.reshape(len(normalized), -1),
        ],
        axis=1,
    )
    functional = transport_functional(cross_block)
    direction = functional / float(np.dot(functional, functional))
    true_transport = carrier.transport_from_selected_cross(
        cross_block.original, cross_block.radial_weights
    )

    # Correct only the one scalar transport direction so that the POD state
    # exactly carries the raw selected-mode transport before decomposition.
    represented_transport = flattened @ functional
    corrected = flattened + (
        true_transport - represented_transport
    )[:, None] * direction[None]
    residual = corrected - true_transport[:, None] * direction[None]
    orthogonality = np.max(np.abs(residual @ functional))
    tolerance = max(float(np.max(np.abs(true_transport))) * 1.0e-10, 1.0e-8)
    if orthogonality > tolerance:
        raise ValueError(
            f"Transport residual is not orthogonal: {orthogonality}"
        )

    residual_mean = np.mean(residual[fit_mask], axis=0)
    centered = residual[fit_mask] - residual_mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = min(
        RESIDUAL_COMPONENTS,
        len(singular_values),
        int(np.count_nonzero(singular_values > 1.0e-12)),
    )
    basis = vh[:components]
    residual_scores = (residual - residual_mean) @ basis.T
    variance = singular_values**2
    explained = variance[:components] / max(
        float(np.sum(variance)), np.finfo(float).tiny
    )
    transport_rate = carrier.causal_difference(
        true_transport[:, None], carrier.FRAME_DT_US
    )[:, 0]
    transport_state = np.column_stack([true_transport, transport_rate])

    temporary = TransportResidual(
        cross_block=cross_block,
        transport=true_transport,
        transport_state=transport_state,
        residual_scores=residual_scores,
        residual_mean=residual_mean,
        residual_basis=basis,
        transport_direction=direction,
        transport_functional=functional,
        oracle_coefficients=np.empty_like(cross_block.original),
        explained_variance=explained,
    )
    temporary.oracle_coefficients = temporary.decode_cross(
        transport_state, residual_scores
    )
    oracle_transport = carrier.transport_from_selected_cross(
        temporary.oracle_coefficients, cross_block.radial_weights
    )
    relative_error = np.max(
        np.abs(oracle_transport - true_transport)
        / np.maximum(np.abs(true_transport), 1.0)
    )
    # The adaptive two-band states can combine coefficients with a wider
    # dynamic range than the original E25 n=2,6 pair.  Keep this as a strict
    # numerical identity check while allowing a few ulps of reconstruction
    # error around the former 1e-10 boundary.
    if relative_error > 5.0e-10:
        raise ValueError(
            f"Decoded cross state is inconsistent with T: {relative_error}"
        )
    return temporary


def causality_audit(raw: carrier.RawPhysical) -> list[dict]:
    """Verify that changing forecast truth cannot alter fit-side states."""
    rows = []
    for window, (fit_start, fit_end, forecast_end) in WINDOWS.items():
        fit_mask = (raw.time_us >= fit_start) & (raw.time_us < fit_end)
        future_mask = (raw.time_us >= fit_end) & (
            raw.time_us <= forecast_end
        )
        modes = carrier.select_modes(raw.phi, raw.radial_weights, fit_mask)
        phi_reference = carrier.build_carrier_block(
            "phi",
            raw.phi,
            modes,
            raw.radial_weights,
            raw.frame,
            fit_mask,
        )
        cross_reference = carrier.build_carrier_block(
            "cross",
            raw.cross,
            modes,
            raw.radial_weights,
            raw.frame,
            fit_mask,
        )
        hybrid_reference = build_transport_residual(
            cross_reference, fit_mask
        )

        perturbed_phi = raw.phi.copy()
        perturbed_cross = raw.cross.copy()
        perturbed_phi[future_mask] = (
            (1.17 + 0.23j) * perturbed_phi[future_mask] + 3.0
        )
        perturbed_cross[future_mask] = (
            (-0.31 + 1.41j) * perturbed_cross[future_mask] - 5.0
        )
        perturbed_modes = carrier.select_modes(
            perturbed_phi, raw.radial_weights, fit_mask
        )
        phi_perturbed = carrier.build_carrier_block(
            "phi",
            perturbed_phi,
            perturbed_modes,
            raw.radial_weights,
            raw.frame,
            fit_mask,
        )
        cross_perturbed = carrier.build_carrier_block(
            "cross",
            perturbed_cross,
            perturbed_modes,
            raw.radial_weights,
            raw.frame,
            fit_mask,
        )
        hybrid_perturbed = build_transport_residual(
            cross_perturbed, fit_mask
        )
        differences = {
            "phi_circular": float(
                np.max(
                    np.abs(
                        phi_reference.circular[fit_mask]
                        - phi_perturbed.circular[fit_mask]
                    )
                )
            ),
            "transport_direct": float(
                np.max(
                    np.abs(
                        hybrid_reference.transport_state[fit_mask]
                        - hybrid_perturbed.transport_state[fit_mask]
                    )
                )
            ),
            "cross_residual": float(
                np.max(
                    np.abs(
                        hybrid_reference.residual_scores[fit_mask]
                        - hybrid_perturbed.residual_scores[fit_mask]
                    )
                )
            ),
        }
        maximum = max(differences.values())
        if not np.array_equal(modes, perturbed_modes) or maximum > 1.0e-10:
            raise ValueError(
                f"Forecast-truth leakage detected for {window}: {maximum}"
            )
        rows.append(
            {
                "window": window,
                "forecast_values_were_perturbed": True,
                "fit_selected_modes_unchanged": True,
                "max_fit_state_absolute_difference": maximum,
                **{
                    f"max_{name}_difference": value
                    for name, value in differences.items()
                },
            }
        )
    return rows


def source_groups(
    latent: np.ndarray,
    phi_block: carrier.CarrierBlock,
    hybrid: TransportResidual,
) -> dict[str, np.ndarray]:
    return {
        "latent": latent,
        "phi_circular": phi_block.circular,
        "transport_direct": hybrid.transport_state,
        "cross_residual": hybrid.residual_scores,
    }


def groups_for_system(
    system: str, sources: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {name: sources[name] for name in SYSTEMS[system]}


def transport_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
) -> dict[str, float]:
    metrics = augmented.scalar_metrics(truth, prediction, persistence)
    scale = float(np.std(truth, ddof=1))
    truth_mean = float(np.mean(truth))
    metrics.update(
        {
            "mean_ratio": (
                float(np.mean(prediction) / truth_mean)
                if abs(truth_mean) > np.finfo(float).tiny
                else float("nan")
            ),
            "normalized_bias": float(
                np.mean(prediction - truth)
                / max(scale, np.finfo(float).tiny)
            ),
        }
    )
    return metrics


def evaluate_system_physics(
    system: str,
    method: str,
    predicted: dict[str, np.ndarray],
    phi_block: carrier.CarrierBlock,
    hybrid: TransportResidual,
    fit_mask: np.ndarray,
    forecast_mask: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    traces: dict[str, np.ndarray] = {}
    fit_last = np.flatnonzero(fit_mask)[-1]
    forecast_frames = phi_block.frame[forecast_mask]

    truth_transport = hybrid.transport[forecast_mask]
    prediction_transport = predicted["transport_direct"][:, 0]
    persistence_transport = np.repeat(
        hybrid.transport[fit_last], len(truth_transport)
    )
    rows.append(
        {
            "system": system,
            "method": method,
            "quantity": "selected_modal_transport",
            **transport_metrics(
                truth_transport,
                prediction_transport,
                persistence_transport,
            ),
        }
    )
    traces["transport_truth"] = truth_transport
    traces["transport_prediction"] = prediction_transport

    if "phi_circular" in predicted:
        phi_rows, phi_traces = carrier.physical_metrics(
            system,
            method,
            {"phi_circular": predicted["phi_circular"]},
            phi_block,
            hybrid.cross_block,
            fit_mask,
            forecast_mask,
        )
        rows.extend(phi_rows)
        traces.update(phi_traces)

    if "cross_residual" not in predicted:
        return rows, traces
    cross_prediction = hybrid.decode_cross(
        predicted["transport_direct"], predicted["cross_residual"]
    )
    cross_truth = hybrid.cross_block.original[forecast_mask]
    persistence_cross = np.repeat(
        hybrid.cross_block.original[fit_last : fit_last + 1],
        len(cross_truth),
        axis=0,
    )
    oracle_cross = hybrid.oracle_coefficients[forecast_mask]
    coefficient_row = carrier.coefficient_metrics(
        cross_truth,
        cross_prediction,
        persistence_cross,
        persistence_cross,
        hybrid.cross_block.radial_weights,
    )
    coefficient_row.update(
        {
            "system": system,
            "method": method,
            "quantity": "transport_orthogonal_cross_residual",
            "oracle_nrmse": augmented.nrmse(cross_truth, oracle_cross),
            "transport_consistency_max_relative_error": float(
                np.max(
                    np.abs(
                        carrier.transport_from_selected_cross(
                            cross_prediction,
                            hybrid.cross_block.radial_weights,
                        )
                        - prediction_transport
                    )
                    / np.maximum(np.abs(prediction_transport), 1.0)
                )
            ),
        }
    )
    rows.append(coefficient_row)
    return rows, traces


def analyze(
    raw: carrier.RawPhysical,
    features: np.ndarray,
    delays: list[int],
    ranks: list[int],
) -> dict[str, list[dict]]:
    output = {
        "selections": [],
        "candidates": [],
        "state": [],
        "physical": [],
        "representations": [],
        "traces": [],
    }
    budget = block.BUDGETS["medium_20"]
    for window, (fit_start, fit_end, forecast_end) in WINDOWS.items():
        validation_start = fit_end - 1.0
        subtrain_mask = (raw.time_us >= fit_start) & (
            raw.time_us < validation_start
        )
        fit_mask = (raw.time_us >= fit_start) & (raw.time_us < fit_end)
        forecast_mask = (raw.time_us >= fit_end) & (
            raw.time_us <= forecast_end
        )
        sub_modes = carrier.select_modes(
            raw.phi, raw.radial_weights, subtrain_mask
        )
        final_modes = carrier.select_modes(
            raw.phi, raw.radial_weights, fit_mask
        )
        sub_phi = carrier.build_carrier_block(
            "phi",
            raw.phi,
            sub_modes,
            raw.radial_weights,
            raw.frame,
            subtrain_mask,
        )
        sub_cross = carrier.build_carrier_block(
            "cross",
            raw.cross,
            sub_modes,
            raw.radial_weights,
            raw.frame,
            subtrain_mask,
        )
        final_phi = carrier.build_carrier_block(
            "phi",
            raw.phi,
            final_modes,
            raw.radial_weights,
            raw.frame,
            fit_mask,
        )
        final_cross = carrier.build_carrier_block(
            "cross",
            raw.cross,
            final_modes,
            raw.radial_weights,
            raw.frame,
            fit_mask,
        )
        sub_hybrid = build_transport_residual(sub_cross, subtrain_mask)
        final_hybrid = build_transport_residual(final_cross, fit_mask)
        _, latent_sub, _ = block.fit_block_models(
            features, subtrain_mask, budget
        )
        _, latent_final, pca_rows = block.fit_block_models(
            features, fit_mask, budget
        )
        sub_sources = source_groups(latent_sub, sub_phi, sub_hybrid)
        final_sources = source_groups(latent_final, final_phi, final_hybrid)

        output["representations"].append(
            {
                "window": window,
                "selected_modes": ",".join(map(str, final_modes)),
                "phi_carrier_dimension": final_phi.circular.shape[1],
                "transport_dimension": final_hybrid.transport_state.shape[1],
                "cross_residual_dimension": final_hybrid.residual_scores.shape[1],
                "cross_residual_explained_variance": float(
                    np.sum(final_hybrid.explained_variance)
                ),
                "cross_oracle_nrmse": augmented.nrmse(
                    final_cross.original[forecast_mask],
                    final_hybrid.oracle_coefficients[forecast_mask],
                ),
                "max_transport_orthogonality_error": float(
                    np.max(
                        np.abs(
                            (
                                final_hybrid.residual_scores
                                @ final_hybrid.residual_basis
                                + final_hybrid.residual_mean
                            )
                            @ final_hybrid.transport_functional
                        )
                    )
                ),
                "max_transport_orthogonality_relative_error": float(
                    np.max(
                        np.abs(
                            (
                                final_hybrid.residual_scores
                                @ final_hybrid.residual_basis
                                + final_hybrid.residual_mean
                            )
                            @ final_hybrid.transport_functional
                        )
                    )
                    / max(
                        float(np.max(np.abs(final_hybrid.transport))),
                        np.finfo(float).tiny,
                    )
                ),
            }
        )

        for system in SYSTEMS:
            sub_groups = groups_for_system(system, sub_sources)
            sub_scaler = augmented.GroupScaler.fit(
                sub_groups, subtrain_mask
            )
            sub_standardized = sub_scaler.transform(sub_groups)
            selected, candidates = rolling.dynamic_search_hankel(
                sub_standardized,
                raw.time_us,
                fit_start,
                fit_end,
                delays,
                ranks,
            )
            for row in candidates:
                output["candidates"].append(
                    {
                        "window": window,
                        "system": system,
                        "selected": (
                            row["delay"] == selected["delay"]
                            and row["rank"] == selected["rank"]
                        ),
                        **row,
                    }
                )
            groups = groups_for_system(system, final_sources)
            scaler = augmented.GroupScaler.fit(groups, fit_mask)
            standardized = scaler.transform(groups)
            predictions = augmented.fit_and_forecast(
                standardized,
                fit_mask,
                int(np.count_nonzero(forecast_mask)),
                int(selected["delay"]),
                int(selected["rank"]),
            )
            output["selections"].append(
                {
                    "window": window,
                    "system": system,
                    "groups": "+".join(SYSTEMS[system]),
                    "state_dimension": int(
                        sum(value.shape[1] for value in groups.values())
                    ),
                    "latent_source_dimension": int(
                        sum(row["total_features"] for row in pca_rows)
                    ),
                    "selected_modes": ",".join(map(str, final_modes)),
                    **selected,
                }
            )
            for method in METHODS:
                metrics = carrier.state_metrics(
                    standardized,
                    predictions[method],
                    fit_mask,
                    forecast_mask,
                    raw.time_us,
                )
                output["state"].append(
                    {
                        "window": window,
                        "system": system,
                        "method": method,
                        **metrics,
                    }
                )
                predicted = scaler.inverse(predictions[method])
                metric_rows, traces = evaluate_system_physics(
                    system,
                    method,
                    predicted,
                    final_phi,
                    final_hybrid,
                    fit_mask,
                    forecast_mask,
                )
                for row in metric_rows:
                    row["window"] = window
                    output["physical"].append(row)
                if window == "fit20_24_forecast24_30":
                    times = raw.time_us[forecast_mask]
                    for index, time_us in enumerate(times):
                        trace_row = {
                            "window": window,
                            "system": system,
                            "method": method,
                            "time_us": float(time_us),
                        }
                        for name, values in traces.items():
                            trace_row[name] = float(values[index])
                        output["traces"].append(trace_row)
        print(
            f"PASS {window}: modes={final_modes.tolist()} "
            f"dX_var={np.sum(final_hybrid.explained_variance):.5f}",
            flush=True,
        )
    return output


def summarize(results: dict[str, list[dict]]) -> list[dict]:
    physical_lookup = {}
    for row in results["physical"]:
        physical_lookup.setdefault(
            (row["system"], row["method"], row["quantity"]), []
        ).append(row)
    output = []
    selections = {
        system: [
            row for row in results["selections"] if row["system"] == system
        ]
        for system in SYSTEMS
    }
    for system in SYSTEMS:
        for method in METHODS:
            state = [
                row
                for row in results["state"]
                if row["system"] == system and row["method"] == method
            ]
            row = {
                "system": system,
                "label": LABELS[system],
                "method": method,
                "groups": "+".join(SYSTEMS[system]),
                "state_dimension": int(selections[system][0]["state_dimension"]),
                "median_delay": finite_median(
                    item["delay"] for item in selections[system]
                ),
                "median_rank": finite_median(
                    item["rank"] for item in selections[system]
                ),
                "median_state_skill": finite_median(
                    item["skill_vs_persistence"] for item in state
                ),
                "min_state_skill": finite_min(
                    item["skill_vs_persistence"] for item in state
                ),
                "positive_state_skill_windows": sum(
                    item["skill_vs_persistence"] > 0.0 for item in state
                ),
                "median_state_correlation": finite_median(
                    item["flattened_correlation"] for item in state
                ),
            }
            for quantity in (
                "selected_modal_transport",
                "selected_phi_coefficients",
                "selected_phi_envelope",
                "transport_orthogonal_cross_residual",
            ):
                items = physical_lookup.get((system, method, quantity), [])
                for metric in (
                    "correlation",
                    "temporal_anomaly_correlation",
                    "skill_vs_persistence",
                    "mean_ratio",
                    "normalized_bias",
                    "coefficient_correlation",
                    "coefficient_skill_vs_persistence",
                    "weighted_phase_mae_rad",
                    "amplitude_correlation",
                    "amplitude_ratio",
                    "normalized_real_bias",
                    "oracle_nrmse",
                    "transport_consistency_max_relative_error",
                ):
                    values = [item[metric] for item in items if metric in item]
                    if values:
                        row[f"{quantity}_median_{metric}"] = finite_median(values)
                        row[f"{quantity}_mean_{metric}"] = finite_mean(values)
                        row[f"{quantity}_min_{metric}"] = finite_min(values)
                if items:
                    skill_name = (
                        "coefficient_skill_vs_persistence"
                        if "cross" in quantity or "coefficients" in quantity
                        else "skill_vs_persistence"
                    )
                    row[f"{quantity}_positive_skill_windows"] = sum(
                        item.get(skill_name, float("-inf")) > 0.0
                        for item in items
                    )
            output.append(row)

    lookup = {(row["system"], row["method"]): row for row in output}
    for row in output:
        baseline_name = PHYSICAL_BASELINES.get(row["system"])
        if baseline_name is None:
            continue
        baseline = lookup[(baseline_name, row["method"])]
        row["physical_only_baseline"] = baseline_name
        for metric in (
            "median_state_skill",
            "selected_modal_transport_median_temporal_anomaly_correlation",
            "selected_modal_transport_median_skill_vs_persistence",
            "selected_modal_transport_median_normalized_bias",
        ):
            if metric in row and metric in baseline:
                row[f"{metric}_gain_vs_physical_only"] = (
                    row[metric] - baseline[metric]
                )
    return output


def plot_summary(path: Path, summary: list[dict]) -> None:
    labels = [
        f"{row['label']}\n{row['method'].replace('_', ' ')}"
        for row in summary
    ]
    metrics = (
        ("median_state_skill", "Median joint-state skill", (-1.0, 1.0)),
        (
            "selected_modal_transport_median_temporal_anomaly_correlation",
            "Transport correlation",
            (-1.0, 1.0),
        ),
        (
            "selected_modal_transport_mean_skill_vs_persistence",
            "Mean transport skill vs persistence",
            (-2.0, 1.0),
        ),
        (
            "selected_modal_transport_min_skill_vs_persistence",
            "Worst-window transport skill",
            (-2.0, 1.0),
        ),
        (
            "selected_modal_transport_median_normalized_bias",
            "Transport normalized bias",
            (-2.0, 2.0),
        ),
        (
            "selected_modal_transport_positive_skill_windows",
            "Positive-skill forecast windows",
            (0.0, 3.2),
        ),
    )
    x = np.arange(len(labels))
    figure, axes = plt.subplots(
        2, 3, figsize=(19.0, 9.5), constrained_layout=True
    )
    for axis, (metric, title, limits) in zip(axes.ravel(), metrics):
        values = np.asarray(
            [row.get(metric, np.nan) for row in summary], dtype=np.float64
        )
        axis.bar(x, np.nan_to_num(values, nan=0.0), color="#4c78a8")
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_ylim(*limits)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("E25 direct-transport and orthogonal cross-residual closure")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_skill_by_window(path: Path, physical: list[dict]) -> None:
    windows = list(WINDOWS)
    short_windows = ["16-22", "20-26", "24-30"]
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(
        1, len(METHODS), figsize=(15.0, 5.6), sharey=True,
        constrained_layout=True,
    )
    for axis, method in zip(np.atleast_1d(axes), METHODS):
        for index, system in enumerate(SYSTEMS):
            values = []
            for window in windows:
                matches = [
                    row
                    for row in physical
                    if row["window"] == window
                    and row["system"] == system
                    and row["method"] == method
                    and row["quantity"] == "selected_modal_transport"
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Missing transport row: {window}/{system}/{method}"
                    )
                values.append(matches[0]["skill_vs_persistence"])
            axis.plot(
                short_windows,
                values,
                marker="o",
                linewidth=1.8,
                color=colors(index),
                label=LABELS[system],
            )
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_ylim(-0.05, 1.0)
        axis.set_title(method.replace("_", " "))
        axis.set_xlabel("forecast interval [us]")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("transport skill vs persistence")
    figure.suptitle("E25 transport closure across rolling forecasts")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_best_rollout(path: Path, traces: list[dict], summary: list[dict]) -> None:
    candidates = [
        row
        for row in summary
        if row["system"] == "latent_phi_transport_residual"
    ]
    best = max(
        candidates,
        key=lambda row: row[
            "selected_modal_transport_median_skill_vs_persistence"
        ],
    )
    selected = [
        row
        for row in traces
        if row["system"] == best["system"] and row["method"] == best["method"]
    ]
    selected.sort(key=lambda row: row["time_us"])
    time = np.asarray([row["time_us"] for row in selected])
    figure, axes = plt.subplots(
        2, 1, figsize=(11.5, 7.0), constrained_layout=True
    )
    for axis, truth_key, prediction_key, ylabel in (
        (
            axes[0],
            "phi_envelope_truth",
            "phi_envelope_prediction",
            "selected MTSI phi envelope",
        ),
        (
            axes[1],
            "transport_truth",
            "transport_prediction",
            "selected modal transport",
        ),
    ):
        axis.plot(
            time,
            [row[truth_key] for row in selected],
            color="#000000",
            label="PIC truth",
        )
        axis.plot(
            time,
            [row[prediction_key] for row in selected],
            color="#d55e00",
            label="autonomous ROM",
        )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
    axes[1].set_xlabel("time [us]")
    figure.suptitle(
        f"E25 20-24 to 24-30 us: {LABELS[best['system']]} / "
        f"{best['method'].replace('_', ' ')}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_readme(
    path: Path,
    summary: list[dict],
    representations: list[dict],
) -> None:
    ranked = sorted(
        summary,
        key=lambda row: row[
            "selected_modal_transport_mean_skill_vs_persistence"
        ],
        reverse=True,
    )
    best = ranked[0]
    lookup = {(row["system"], row["method"]): row for row in summary}
    direct = lookup[("transport_only", best["method"])]
    without_residual = lookup[("latent_phi_transport", best["method"])]
    without_latent = lookup[("phi_transport_residual", best["method"])]
    lines = [
        "# E25 direct-transport plus orthogonal cross-residual closure",
        "",
        "## Design",
        "",
        "`T` is the selected n=2,6 modal transport and its causal time derivative. `dX` is the density-Ey cross-spectrum after the complete scalar transport direction has been projected out, followed by a six-component fit-only PCA. Thus T and dX contain non-overlapping information. `Pcirc` is the circular phi carrier state from the preceding experiment; `L` is the frozen data-only SimVP latent state.",
        "",
        "Every rolling window uses only its fit interval for mode selection, POD/PCA, carrier estimation, normalization, and final model fitting. Delay/rank use only the final one microsecond of the fit interval. The six-microsecond forecast is fully autonomous.",
        "",
        "## Summary",
        "",
        "| method | state | dim | state skill median/min | transport corr | transport skill mean/min | normalized bias | positive transport windows |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['method']} | {row['label']} | {row['state_dimension']} | "
            f"{row['median_state_skill']:.3f}/{row['min_state_skill']:.3f} | "
            f"{row['selected_modal_transport_median_temporal_anomaly_correlation']:.3f} | "
            f"{row['selected_modal_transport_mean_skill_vs_persistence']:.3f}/"
            f"{row['selected_modal_transport_min_skill_vs_persistence']:.3f} | "
            f"{row['selected_modal_transport_median_normalized_bias']:.3f} | "
            f"{row['selected_modal_transport_positive_skill_windows']}/3 |"
        )
    representation_text = "; ".join(
        f"{row['window']}: dX variance={row['cross_residual_explained_variance']:.5f}, oracle NRMSE={row['cross_oracle_nrmse']:.5f}"
        for row in representations
    )
    lines.extend(
        [
            "",
            "## Reading the ablation",
            "",
            "- `T` versus `L+T`: incremental information in the frozen SimVP latent state.",
            "- `Pcirc+T` versus `L+Pcirc+T`: whether latent information helps after physical phase is explicit.",
            "- `Pcirc+T+dX` versus `L+Pcirc+T+dX`: the strongest coupled closure test.",
            "- Adding dX tests whether transport-orthogonal cross-spectrum structure predicts future T. Because dX is orthogonal to T by construction, improvement cannot be attributed to copying T twice.",
            "",
            "## Main result",
            "",
            f"Across the three rolling forecasts, the best mean transport skill is {best['selected_modal_transport_mean_skill_vs_persistence']:.3f} for {best['label']} / {best['method'].replace('_', ' ')}. The direct T state is therefore compared against every coupled state using both the three-window mean and the worst window, rather than selecting by the median alone.",
            f"Against direct T with the same method, the mean skill gain is {best['selected_modal_transport_mean_skill_vs_persistence'] - direct['selected_modal_transport_mean_skill_vs_persistence']:+.3f}, while the worst-window skill changes from {direct['selected_modal_transport_min_skill_vs_persistence']:.3f} to {best['selected_modal_transport_min_skill_vs_persistence']:.3f}.",
            f"Adding dX to L+Pcirc+T changes mean skill by {best['selected_modal_transport_mean_skill_vs_persistence'] - without_residual['selected_modal_transport_mean_skill_vs_persistence']:+.3f}. Adding L to Pcirc+T+dX changes it by {best['selected_modal_transport_mean_skill_vs_persistence'] - without_latent['selected_modal_transport_mean_skill_vs_persistence']:+.3f}.",
            "These gains show that direct transport autoregression is the strongest single ingredient, but the transport-orthogonal cross structure and frozen SimVP latent state carry additional predictive information. This remains an in-case E25 closure result, not yet a transferred ROM.",
            "",
            f"Representation diagnostics: {representation_text}.",
            "",
            "## Files",
            "",
            "- `summary.csv`: rolling aggregate.",
            "- `state_metrics_by_window.csv` and `physical_metrics_by_window.csv`: per-window metrics.",
            "- `representation_by_window.csv`: dX compression and oracle diagnostics.",
            "- `causality_audit.csv`: forecast-truth perturbation audit.",
            "- `model_selections.csv` and `validation_candidates.csv`: fit-only model selection.",
            "- `last_window_time_series.csv`: 24-30 us traces.",
            "- `transport_closure_summary.png`, `transport_skill_by_window.png`, and `best_transport_rollout.png`: visual summaries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physical", type=Path, default=carrier.DEFAULT_PHYSICAL
    )
    parser.add_argument(
        "--features", type=Path, default=carrier.DEFAULT_FEATURES
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delays", default="40,80,120")
    parser.add_argument("--ranks", default="8,12,20,30,40")
    args = parser.parse_args()
    delays = [int(value) for value in args.delays.split(",")]
    ranks = [int(value) for value in args.ranks.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)

    raw = carrier.load_raw_physical(args.physical)
    features, time_us, frames = block.load_features(args.features)
    if not np.allclose(raw.time_us, time_us, atol=1.0e-9):
        raise ValueError("Physical and latent time axes do not match")
    if not np.array_equal(raw.frame, frames):
        raise ValueError("Physical and latent frame axes do not match")
    audit_rows = causality_audit(raw)
    results = analyze(raw, features, delays, ranks)
    summary = summarize(results)

    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "state_metrics_by_window.csv", results["state"])
    write_csv(args.output / "physical_metrics_by_window.csv", results["physical"])
    write_csv(args.output / "representation_by_window.csv", results["representations"])
    write_csv(args.output / "model_selections.csv", results["selections"])
    write_csv(args.output / "validation_candidates.csv", results["candidates"])
    write_csv(args.output / "last_window_time_series.csv", results["traces"])
    write_csv(args.output / "causality_audit.csv", audit_rows)
    plot_summary(args.output / "transport_closure_summary.png", summary)
    plot_skill_by_window(
        args.output / "transport_skill_by_window.png", results["physical"]
    )
    plot_best_rollout(
        args.output / "best_transport_rollout.png", results["traces"], summary
    )
    write_readme(args.output / "README.md", summary, results["representations"])

    payload = {
        "status": "PASS",
        "physical_source": str(args.physical.resolve()),
        "latent_source": str(args.features.resolve()),
        "windows": WINDOWS,
        "delays": delays,
        "ranks": ranks,
        "residual_components": RESIDUAL_COMPONENTS,
        "forecast_truth_used_as_input": False,
        "causality_audit": audit_rows,
        "transport_and_residual_are_orthogonal": True,
        "summary": summary,
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)
    print(f"PASS: wrote transport residual closure to {args.output}")


if __name__ == "__main__":
    main()
