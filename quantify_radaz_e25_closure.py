"""Quantify approximate reduced-order closure for the E25 RadAz case.

The analysis compares a transparent hierarchy of non-redundant states over
three rolling autonomous forecast windows.  It reuses the frozen SimVP latent
Fourier features and the physical Fourier observables produced by the earlier
analysis; the SimVP checkpoints are not retrained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import analyze_radaz_coupling_and_rolling_validation as rolling
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "workdirs" / "quantify_radaz_e25_rom_closure"

# Cross-spectrum and transport are not placed in the same state because modal
# transport is an algebraic radial integral of the cross-spectrum real part.
SYSTEMS = {
    "latent_only": ("latent",),
    "latent_radial": ("latent", "radial"),
    "latent_radial_transport": ("latent", "radial", "transport"),
    "latent_radial_cross": ("latent", "radial", "cross"),
    "radial_only": ("radial",),
    "radial_transport_only": ("radial", "transport"),
    "radial_cross_only": ("radial", "cross"),
}
CORE_SYSTEMS = (
    "latent_only",
    "latent_radial",
    "latent_radial_transport",
    "latent_radial_cross",
)
PHYSICAL_BASELINES = {
    "latent_radial": "radial_only",
    "latent_radial_transport": "radial_transport_only",
    "latent_radial_cross": "radial_cross_only",
}
METHODS = rolling.METHODS
WINDOWS = rolling.WINDOWS

SYSTEM_LABELS = {
    "latent_only": "L",
    "latent_radial": "L+R",
    "latent_radial_transport": "L+R+T",
    "latent_radial_cross": "L+R+C",
    "radial_only": "R only",
    "radial_transport_only": "R+T only",
    "radial_cross_only": "R+C only",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    augmented.write_csv(path, rows)


def finite_median(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def finite_min(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.min(array)) if array.size else float("nan")


def groups_for_system(
    system: str,
    latent: np.ndarray,
    physical: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    sources = {"latent": latent, **physical}
    return {name: sources[name] for name in SYSTEMS[system]}


def system_dimension(groups: dict[str, np.ndarray]) -> int:
    return int(sum(values.shape[1] for values in groups.values()))


def evaluate_state(
    checkpoint: str,
    window: str,
    system: str,
    method: str,
    standardized: np.ndarray,
    prediction: np.ndarray,
    fit_mask: np.ndarray,
    forecast_mask: np.ndarray,
    time_us: np.ndarray,
) -> dict:
    truth = standardized[forecast_mask]
    fit = standardized[fit_mask]
    persistence = np.repeat(fit[-1:, :], len(truth), axis=0)
    metrics, _ = reduced.evaluate_prediction(
        truth, prediction, persistence, time_us[forecast_mask]
    )
    return {
        "checkpoint": checkpoint,
        "window": window,
        "system": system,
        "method": method,
        **metrics,
    }


def analyze_checkpoint(
    checkpoint: str,
    specification: dict,
    physical: augmented.PhysicalStates,
    physical_flat: dict[str, np.ndarray],
    delays: list[int],
    ranks: list[int],
) -> dict[str, list[dict]]:
    features, time_us, frames = block.load_features(specification["features"])
    if not np.allclose(time_us, physical.time_us, atol=1.0e-9):
        raise ValueError(f"Time mismatch for {checkpoint}")
    if not np.array_equal(frames, physical.frame):
        raise ValueError(f"Frame mismatch for {checkpoint}")

    output: dict[str, list[dict]] = {
        "selections": [],
        "candidates": [],
        "state": [],
        "latent": [],
        "modes": [],
        "auxiliary": [],
    }
    budget_name = specification["budget"]
    budget = block.BUDGETS[budget_name]

    for window, (fit_start, fit_end, forecast_end) in WINDOWS.items():
        validation_start = fit_end - 1.0
        subtrain_mask = (time_us >= fit_start) & (
            time_us < validation_start
        )
        fit_mask = (time_us >= fit_start) & (time_us < fit_end)
        forecast_mask = (time_us >= fit_end) & (
            time_us <= forecast_end
        )

        _, sub_scores, _ = block.fit_block_models(
            features, subtrain_mask, budget
        )
        final_models, final_scores, pca_rows = block.fit_block_models(
            features, fit_mask, budget
        )
        source_dimension = int(sum(row["total_features"] for row in pca_rows))

        for system in SYSTEMS:
            sub_groups = groups_for_system(
                system, sub_scores, physical_flat
            )
            sub_scaler = augmented.GroupScaler.fit(
                sub_groups, subtrain_mask
            )
            sub_standardized = sub_scaler.transform(sub_groups)
            selected, candidates = rolling.dynamic_search_hankel(
                sub_standardized,
                time_us,
                fit_start,
                fit_end,
                delays,
                ranks,
            )
            for row in candidates:
                output["candidates"].append(
                    {
                        "checkpoint": checkpoint,
                        "window": window,
                        "system": system,
                        "selected": (
                            row["delay"] == selected["delay"]
                            and row["rank"] == selected["rank"]
                        ),
                        **row,
                    }
                )

            groups = groups_for_system(system, final_scores, physical_flat)
            scaler = augmented.GroupScaler.fit(groups, fit_mask)
            standardized = scaler.transform(groups)
            predictions = augmented.fit_and_forecast(
                standardized,
                fit_mask,
                int(np.count_nonzero(forecast_mask)),
                int(selected["delay"]),
                int(selected["rank"]),
            )
            dimension = system_dimension(groups)
            output["selections"].append(
                {
                    "checkpoint": checkpoint,
                    "window": window,
                    "system": system,
                    "state_dimension": dimension,
                    "latent_source_dimension": source_dimension,
                    "latent_compression_ratio": (
                        source_dimension / dimension
                        if "latent" in SYSTEMS[system]
                        else float("nan")
                    ),
                    "delay": selected["delay"],
                    "history_us": selected["history_us"],
                    "rank": selected["rank"],
                    "validation_mse": selected["validation_mse"],
                    "validation_skill_vs_persistence": selected[
                        "validation_skill_vs_persistence"
                    ],
                    "validation_correlation": selected[
                        "validation_correlation"
                    ],
                    "spectral_radius": selected["spectral_radius"],
                }
            )

            truth_physical = augmented.subset_physical(
                physical, forecast_mask
            )
            fit_physical = augmented.subset_physical(physical, fit_mask)
            for method in METHODS:
                prediction_standardized = predictions[method]
                prediction_groups = scaler.inverse(
                    prediction_standardized
                )
                output["state"].append(
                    evaluate_state(
                        checkpoint,
                        window,
                        system,
                        method,
                        standardized,
                        prediction_standardized,
                        fit_mask,
                        forecast_mask,
                        time_us,
                    )
                )

                if "latent" in SYSTEMS[system]:
                    latent_row = augmented.latent_metrics(
                        checkpoint,
                        system,
                        method,
                        final_scores[forecast_mask],
                        prediction_groups["latent"],
                        final_scores[fit_mask],
                    )
                    latent_row["window"] = window
                    output["latent"].append(latent_row)
                    mode_rows = augmented.mode_metric_rows(
                        checkpoint,
                        system,
                        method,
                        final_models,
                        features[forecast_mask],
                        prediction_groups["latent"],
                    )
                    for row in mode_rows:
                        row["window"] = window
                        output["modes"].append(row)

                auxiliary_rows, _ = augmented.auxiliary_metric_rows(
                    checkpoint,
                    system,
                    method,
                    prediction_groups,
                    truth_physical,
                    fit_physical,
                )
                for row in auxiliary_rows:
                    row["window"] = window
                    output["auxiliary"].append(row)

        print(
            f"[{checkpoint}] {window}: evaluated {len(SYSTEMS)} states",
            flush=True,
        )
    return output


def rows_for(
    rows: list[dict],
    checkpoint: str,
    system: str,
    method: str,
    **filters,
) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("checkpoint") == checkpoint
        and row.get("system", row.get("variant")) == system
        and row.get("method") == method
        and all(row.get(key) == value for key, value in filters.items())
    ]


def aggregate_results(results: dict[str, list[dict]]) -> list[dict]:
    summaries = []
    selections = results["selections"]
    for checkpoint in augmented.CHECKPOINTS:
        for system in SYSTEMS:
            selected_rows = [
                row
                for row in selections
                if row["checkpoint"] == checkpoint
                and row["system"] == system
            ]
            for method in METHODS:
                state = rows_for(
                    results["state"], checkpoint, system, method
                )
                latent = rows_for(
                    results["latent"], checkpoint, system, method
                )
                modes = rows_for(
                    results["modes"], checkpoint, system, method
                )
                auxiliary = rows_for(
                    results["auxiliary"], checkpoint, system, method
                )
                row = {
                    "checkpoint": checkpoint,
                    "system": system,
                    "method": method,
                    "groups": "+".join(SYSTEMS[system]),
                    "state_dimension": int(
                        finite_median(
                            [item["state_dimension"] for item in selected_rows]
                        )
                    ),
                    "median_delay": finite_median(
                        [item["delay"] for item in selected_rows]
                    ),
                    "median_rank": finite_median(
                        [item["rank"] for item in selected_rows]
                    ),
                    "max_spectral_radius": max(
                        item["spectral_radius"] for item in selected_rows
                    ),
                    "median_state_skill": finite_median(
                        [item["skill_vs_persistence"] for item in state]
                    ),
                    "min_state_skill": finite_min(
                        [item["skill_vs_persistence"] for item in state]
                    ),
                    "median_state_correlation": finite_median(
                        [item["flattened_correlation"] for item in state]
                    ),
                    "min_finite_fraction": finite_min(
                        [item["finite_fraction"] for item in state]
                    ),
                    "positive_state_skill_windows": sum(
                        item["skill_vs_persistence"] > 0.0 for item in state
                    ),
                    "total_windows": len(state),
                }

                if latent:
                    row.update(
                        {
                            "median_latent_skill": finite_median(
                                [
                                    item["skill_vs_persistence"]
                                    for item in latent
                                ]
                            ),
                            "median_latent_correlation": finite_median(
                                [
                                    item["flattened_correlation"]
                                    for item in latent
                                ]
                            ),
                        }
                    )
                for band in augmented.MODE_BANDS:
                    band_rows = [item for item in modes if item["band"] == band]
                    if band_rows:
                        row[f"{band}_median_latent_amplitude_correlation"] = (
                            finite_median(
                                [
                                    item["amplitude_correlation"]
                                    for item in band_rows
                                ]
                            )
                        )

                for quantity in (
                    "radial_envelope",
                    "modal_transport",
                    "cross_spectrum",
                ):
                    for band in augmented.MODE_BANDS:
                        quantity_rows = [
                            item
                            for item in auxiliary
                            if item["quantity"] == quantity
                            and item["band"] == band
                        ]
                        if not quantity_rows:
                            continue
                        prefix = f"{band}_{quantity}"
                        row[f"{prefix}_median_correlation"] = finite_median(
                            [item["correlation"] for item in quantity_rows]
                        )
                        row[f"{prefix}_median_temporal_correlation"] = (
                            finite_median(
                                [
                                    item["temporal_anomaly_correlation"]
                                    for item in quantity_rows
                                ]
                            )
                        )
                        row[f"{prefix}_median_skill"] = finite_median(
                            [
                                item["skill_vs_persistence"]
                                for item in quantity_rows
                            ]
                        )
                        row[f"{prefix}_positive_skill_windows"] = sum(
                            item["skill_vs_persistence"] > 0.0
                            for item in quantity_rows
                        )
                        if quantity == "cross_spectrum":
                            row[f"{prefix}_median_phase_mae_rad"] = (
                                finite_median(
                                    [
                                        item["weighted_phase_mae_rad"]
                                        for item in quantity_rows
                                    ]
                                )
                            )
                summaries.append(row)
    return summaries


def add_physical_baseline_gains(summaries: list[dict]) -> None:
    lookup = {
        (row["checkpoint"], row["method"], row["system"]): row
        for row in summaries
    }
    metric_suffixes = (
        "median_state_skill",
        "median_state_correlation",
        "MTSI_n1_6_radial_envelope_median_temporal_correlation",
        "ECDI_n9_21_radial_envelope_median_temporal_correlation",
        "MTSI_n1_6_modal_transport_median_correlation",
        "ECDI_n9_21_modal_transport_median_correlation",
        "MTSI_n1_6_cross_spectrum_median_temporal_correlation",
        "ECDI_n9_21_cross_spectrum_median_temporal_correlation",
    )
    for row in summaries:
        baseline_name = PHYSICAL_BASELINES.get(row["system"])
        if baseline_name is None:
            continue
        baseline = lookup[(row["checkpoint"], row["method"], baseline_name)]
        row["physical_only_baseline"] = baseline_name
        for metric in metric_suffixes:
            if metric not in row or metric not in baseline:
                continue
            row[f"{metric}_gain_vs_physical_only"] = (
                row[metric] - baseline[metric]
            )


def criterion(
    rows: list[dict],
    summary: dict,
    name: str,
    description: str,
    applicable: bool,
    passed: bool,
    value: float | int | None = None,
    threshold: str = "",
) -> None:
    rows.append(
        {
            "checkpoint": summary["checkpoint"],
            "system": summary["system"],
            "method": summary["method"],
            "criterion": name,
            "description": description,
            "applicable": applicable,
            "passed": bool(passed) if applicable else "",
            "value": value if applicable else "",
            "threshold": threshold if applicable else "",
        }
    )


def closure_criteria(summaries: list[dict]) -> tuple[list[dict], list[dict]]:
    criteria_rows = []
    scored = []
    total_criteria = 9
    for source in summaries:
        summary = dict(source)
        groups = set(SYSTEMS[summary["system"]])
        local = []
        criterion(
            local,
            summary,
            "state_skill_all_windows",
            "Joint-state skill is positive in every rolling window.",
            True,
            summary["min_state_skill"] > 0.0,
            summary["min_state_skill"],
            "> 0",
        )
        criterion(
            local,
            summary,
            "state_correlation",
            "Median joint-state trajectory correlation is at least 0.5.",
            True,
            summary["median_state_correlation"] >= 0.5,
            summary["median_state_correlation"],
            ">= 0.5",
        )
        criterion(
            local,
            summary,
            "finite_rollout",
            "Every autonomous rollout remains finite.",
            True,
            summary["min_finite_fraction"] >= 1.0,
            summary["min_finite_fraction"],
            "= 1",
        )

        latent_applicable = "latent" in groups
        criterion(
            local,
            summary,
            "latent_trajectory",
            "Median latent trajectory correlation is at least 0.5.",
            latent_applicable,
            summary.get("median_latent_correlation", -math.inf) >= 0.5,
            summary.get("median_latent_correlation"),
            ">= 0.5",
        )
        for band in augmented.MODE_BANDS:
            key = f"{band}_median_latent_amplitude_correlation"
            criterion(
                local,
                summary,
                f"latent_amplitude_{band}",
                f"Median {band} latent-mode amplitude correlation is at least 0.5.",
                latent_applicable,
                summary.get(key, -math.inf) >= 0.5,
                summary.get(key),
                ">= 0.5",
            )

        radial_applicable = "radial" in groups
        radial_values = []
        radial_skills = []
        for band in augmented.MODE_BANDS:
            radial_values.append(
                summary.get(
                    f"{band}_radial_envelope_median_temporal_correlation",
                    float("nan"),
                )
            )
            radial_skills.append(
                summary.get(
                    f"{band}_radial_envelope_median_skill", float("nan")
                )
            )
        radial_value = finite_min(radial_values)
        radial_skill = finite_min(radial_skills)
        criterion(
            local,
            summary,
            "radial_envelope",
            "Both radial-envelope bands have median temporal correlation >= 0.5 and positive skill.",
            radial_applicable,
            radial_value >= 0.5 and radial_skill > 0.0,
            radial_value,
            "min corr >= 0.5 and min skill > 0",
        )

        transport_applicable = "transport" in groups or "cross" in groups
        transport_correlations = []
        transport_skills = []
        transport_positive = []
        for band in augmented.MODE_BANDS:
            prefix = f"{band}_modal_transport"
            transport_correlations.append(
                summary.get(f"{prefix}_median_correlation", float("nan"))
            )
            transport_skills.append(
                summary.get(f"{prefix}_median_skill", float("nan"))
            )
            transport_positive.append(
                summary.get(f"{prefix}_positive_skill_windows", 0)
            )
        transport_value = finite_min(transport_correlations)
        transport_skill = finite_min(transport_skills)
        criterion(
            local,
            summary,
            "modal_transport",
            "Both transport bands have median correlation >= 0.5, positive skill, and positive skill in at least 2/3 windows.",
            transport_applicable,
            (
                transport_value >= 0.5
                and transport_skill > 0.0
                and min(transport_positive, default=0) >= 2
            ),
            transport_value,
            "min corr >= 0.5, min skill > 0, >= 2/3 windows",
        )

        cross_applicable = "cross" in groups
        cross_correlations = []
        cross_skills = []
        phase_errors = []
        for band in augmented.MODE_BANDS:
            prefix = f"{band}_cross_spectrum"
            cross_correlations.append(
                summary.get(
                    f"{prefix}_median_temporal_correlation", float("nan")
                )
            )
            cross_skills.append(
                summary.get(f"{prefix}_median_skill", float("nan"))
            )
            phase_errors.append(
                summary.get(f"{prefix}_median_phase_mae_rad", float("nan"))
            )
        cross_value = finite_min(cross_correlations)
        cross_skill = finite_min(cross_skills)
        phase_value = (
            float(np.nanmax(phase_errors))
            if np.any(np.isfinite(phase_errors))
            else float("nan")
        )
        criterion(
            local,
            summary,
            "complex_cross_spectrum",
            "Both cross-spectrum bands have temporal correlation >= 0.5, positive skill, and phase MAE <= pi/4.",
            cross_applicable,
            (
                cross_value >= 0.5
                and cross_skill > 0.0
                and phase_value <= math.pi / 4.0
            ),
            cross_value,
            "min corr >= 0.5, min skill > 0, max phase MAE <= pi/4",
        )

        applicable = [row for row in local if row["applicable"]]
        passed = [row for row in applicable if row["passed"] is True]
        applicable_fraction = len(passed) / len(applicable)
        coverage_fraction = len(applicable) / total_criteria
        overall_score = len(passed) / total_criteria
        if overall_score >= 0.8 and summary["min_state_skill"] > 0.0:
            label = "strong_local_evidence"
        elif overall_score >= 0.6 and summary["min_state_skill"] > 0.0:
            label = "partial_local_closure"
        elif overall_score >= 0.4:
            label = "weak_partial_evidence"
        else:
            label = "closure_not_supported"
        summary.update(
            {
                "criteria_passed": len(passed),
                "criteria_applicable": len(applicable),
                "applicable_pass_fraction": applicable_fraction,
                "physical_coverage_fraction": coverage_fraction,
                "overall_closure_score": overall_score,
                "closure_evidence_fraction": overall_score,
                "closure_label": label,
            }
        )
        scored.append(summary)
        criteria_rows.extend(local)
    return scored, criteria_rows


def plot_criteria(path: Path, criteria: list[dict]) -> None:
    names = [
        "state_skill_all_windows",
        "state_correlation",
        "finite_rollout",
        "latent_trajectory",
        "latent_amplitude_MTSI_n1_6",
        "latent_amplitude_ECDI_n9_21",
        "radial_envelope",
        "modal_transport",
        "complex_cross_spectrum",
    ]
    rows = [
        row
        for row in criteria
        if row["system"] in CORE_SYSTEMS
    ]
    keys = []
    for checkpoint in augmented.CHECKPOINTS:
        for method in METHODS:
            for system in CORE_SYSTEMS:
                keys.append((checkpoint, method, system))
    matrix = np.full((len(keys), len(names)), np.nan)
    lookup = {
        (row["checkpoint"], row["method"], row["system"], row["criterion"]): row
        for row in rows
    }
    for i, (checkpoint, method, system) in enumerate(keys):
        for j, name in enumerate(names):
            item = lookup.get((checkpoint, method, system, name))
            if item is not None and item["applicable"]:
                matrix[i, j] = 1.0 if item["passed"] else 0.0

    figure, axis = plt.subplots(figsize=(13.5, 8.5), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(["#d55e00", "#009e73"])
    cmap.set_bad("#d9d9d9")
    axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(
        [
            "state skill\nall windows",
            "state\ncorr",
            "finite",
            "latent\ncorr",
            "MTSI\nlatent amp",
            "ECDI\nlatent amp",
            "radial\nenvelope",
            "modal\ntransport",
            "complex\ncross-spectrum",
        ],
        rotation=30,
        ha="right",
    )
    axis.set_yticks(range(len(keys)))
    axis.set_yticklabels(
        [
            f"{checkpoint} | {method.replace('_', ' ')} | {SYSTEM_LABELS[system]}"
            for checkpoint, method, system in keys
        ]
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "N/A" if not np.isfinite(value) else ("PASS" if value else "FAIL")
            axis.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if np.isfinite(value) else "#555555",
            )
    axis.set_title("E25 approximate-closure criteria across three rolling windows")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_robustness(path: Path, state_rows: list[dict]) -> None:
    labels = list(WINDOWS)
    x = np.arange(len(labels))
    figure, axes = plt.subplots(
        2, 2, figsize=(13.5, 9.0), sharex=True, constrained_layout=True
    )
    colors = {
        "latent_only": "#000000",
        "latent_radial": "#0072b2",
        "latent_radial_transport": "#d55e00",
        "latent_radial_cross": "#009e73",
    }
    for row_index, checkpoint in enumerate(augmented.CHECKPOINTS):
        for column_index, method in enumerate(METHODS):
            axis = axes[row_index, column_index]
            for system in CORE_SYSTEMS:
                selected = [
                    row
                    for row in state_rows
                    if row["checkpoint"] == checkpoint
                    and row["system"] == system
                    and row["method"] == method
                ]
                by_window = {row["window"]: row for row in selected}
                values = [
                    by_window[label]["skill_vs_persistence"]
                    for label in labels
                ]
                display_values = np.maximum(values, -1.0)
                axis.plot(
                    x,
                    display_values,
                    marker="o",
                    linewidth=1.8,
                    label=SYSTEM_LABELS[system],
                    color=colors[system],
                )
                for position, (raw, shown) in enumerate(
                    zip(values, display_values)
                ):
                    if raw < -1.0:
                        axis.annotate(
                            f"{raw:.1f}",
                            (position, shown),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha="center",
                            fontsize=7,
                            color=colors[system],
                        )
            axis.axhline(0.0, color="#777777", linewidth=1.0)
            axis.set_ylim(-1.05, 1.0)
            axis.set_title(f"{checkpoint} | {method.replace('_', ' ')}")
            axis.set_ylabel("joint-state skill vs persistence")
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right", framealpha=0.9)
    for axis in axes[-1]:
        axis.set_xticks(x)
        axis.set_xticklabels(["12-16→22", "16-20→26", "20-24→30"])
        axis.set_xlabel("fit interval → forecast end [us]")
    figure.suptitle(
        "E25 closure robustness (skills below -1 are clipped and annotated)"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_readme(path: Path, scored: list[dict]) -> None:
    core = [row for row in scored if row["system"] in CORE_SYSTEMS]
    ranked = sorted(
        core,
        key=lambda row: (
            row["overall_closure_score"],
            row["median_state_skill"],
        ),
        reverse=True,
    )
    lines = [
        "# E25 ROM closure quantification",
        "",
        "## 目的",
        "",
        "25 kV/mケースの低次元状態が、未学習の6 usを真値再入力なしで自律予測できるかを、3つのrolling時間窓で定量化した。SimVP本体の再学習は行っていない。",
        "",
        "## 状態",
        "",
        "- `L`: 未プールSimVP潜在Fourier係数のblockwise PCA状態。",
        "- `L+R`: Lと4 radial帯域 x 2 mode帯のphi envelope。",
        "- `L+R+T`: L+RとMTSI/ECDI modal transport。",
        "- `L+R+C`: L+Rと4 radial帯域 x 2 mode帯の複素density--Ey cross-spectrum。",
        "- `R only`, `R+T only`, `R+C only`: 潜在状態の増分情報を調べる物理observableのみの対照。",
        "",
        "modal transportはcross-spectrum実部のradial積分なので、CとTは同じ状態に重複投入していない。",
        "",
        "## 評価条件",
        "",
        "- rolling窓: 12--16→22 us、16--20→26 us、20--24→30 us。",
        "- 各窓の最後の1 usだけでdelay/rankを選択し、その後6 usを自律予測。",
        "- 主比較: persistenceに対するskill、軌道相関、MTSI/ECDI包絡、radial envelope、cross-spectrum、cross-phase、modal transport。",
        "- 各groupを成分別z-scoreした後、1/sqrt(group dimension)で重み付けした。",
        "",
        "## 閉包判定",
        "",
        "`applicable_pass_fraction`は、その状態が実際に持つobservableに対するPASS率である。これだけではobservableの少ない状態が有利になるため、主順位には全9条件を分母にした`overall_closure_score`を使う。予測対象に含めない物理量もカバレッジ不足として総合点に反映する。これは定理的な閉包証明ではなく、候補を同じ物差しで比較する診断スコアである。閾値と各PASS/FAILは`closure_criteria.csv`に残している。",
        "",
        "| checkpoint | method | state | dim | overall score | applicable pass | coverage | state skill median/min | state corr | label |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['checkpoint']} | {row['method']} | {SYSTEM_LABELS[row['system']]} | "
            f"{row['state_dimension']} | {row['overall_closure_score']:.3f} | "
            f"{row['applicable_pass_fraction']:.3f} "
            f"({row['criteria_passed']}/{row['criteria_applicable']}) | "
            f"{row['physical_coverage_fraction']:.3f} | "
            f"{row['median_state_skill']:.3f}/{row['min_state_skill']:.3f} | "
            f"{row['median_state_correlation']:.3f} | {row['closure_label']} |"
        )
    best = ranked[0]
    baseline_name = best.get("physical_only_baseline")
    baseline_lines = []
    if baseline_name is not None:
        baseline_lines = [
            "",
            "## 潜在状態の増分情報",
            "",
            f"最上位状態の物理量のみ対照は `{SYSTEM_LABELS[baseline_name]}` である。独立にvalidation選択した対照と比べ、joint-state skill中央値の増分は {best.get('median_state_skill_gain_vs_physical_only', float('nan')):.3f} だった。",
            f"modal transport相関の増分はMTSIで {best.get('MTSI_n1_6_modal_transport_median_correlation_gain_vs_physical_only', float('nan')):.3f}、ECDIで {best.get('ECDI_n9_21_modal_transport_median_correlation_gain_vs_physical_only', float('nan')):.3f} だった。正なら、物理observable自身の自己予測だけでなくSimVP潜在状態が追加情報を持つ。",
        ]
    lines.extend(
        [
            "",
            "## 読み方",
            "",
            f"診断上の最上位は `{best['checkpoint']} / {best['method']} / {SYSTEM_LABELS[best['system']]}` で、overall closure scoreは {best['overall_closure_score']:.3f}、適用項目内PASS率は {best['applicable_pass_fraction']:.3f} だった。",
            "",
            "重要なのは最高点だけでなく、`closure_criteria_matrix.png`でどの物理量がFAILしたかを見ることである。全窓でstate skillが正でも、輸送やcross-phaseがFAILなら完全な物理閉包とは呼ばない。また、同一E25軌道の時間窓をずらした評価なので、ここでのstrongはE25内のlocal closureを意味し、異なるEzへのパラメータ汎化を意味しない。",
            *baseline_lines,
            "",
            "## 出力",
            "",
            "- `closure_summary.csv`: 状態・checkpoint・手法ごとの3窓集約。",
            "- `closure_criteria.csv`: 閾値を含む全PASS/FAIL。",
            "- `state_metrics_by_window.csv`: 各窓のjoint-state自律予測指標。",
            "- `latent_metrics_by_window.csv`, `latent_mode_metrics_by_window.csv`: 潜在軌道とmode包絡。",
            "- `auxiliary_metrics_by_window.csv`: radial/cross/transport指標。",
            "- `model_selections.csv`: 各窓のdelay、rank、状態次元、spectral radius。",
            "- `closure_criteria_matrix.png`: 閉包条件の一覧。",
            "- `closure_robustness_by_window.png`: 時間窓ごとのjoint-state skill。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physical", type=Path, default=augmented.DEFAULT_PHYSICAL
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delays", default="20,40,60,80,100")
    parser.add_argument("--ranks", default="8,12,20,30,40")
    args = parser.parse_args()
    delays = [int(value) for value in args.delays.split(",")]
    ranks = [int(value) for value in args.ranks.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)

    physical = augmented.load_physical_states(args.physical)
    physical_flat = augmented.flatten_physical(physical)
    all_results = {
        "selections": [],
        "candidates": [],
        "state": [],
        "latent": [],
        "modes": [],
        "auxiliary": [],
    }
    for checkpoint, specification in augmented.CHECKPOINTS.items():
        result = analyze_checkpoint(
            checkpoint,
            specification,
            physical,
            physical_flat,
            delays,
            ranks,
        )
        for key in all_results:
            all_results[key].extend(result[key])

    summaries = aggregate_results(all_results)
    add_physical_baseline_gains(summaries)
    scored, criteria = closure_criteria(summaries)
    write_csv(args.output / "closure_summary.csv", scored)
    write_csv(args.output / "closure_criteria.csv", criteria)
    write_csv(
        args.output / "state_metrics_by_window.csv", all_results["state"]
    )
    write_csv(
        args.output / "latent_metrics_by_window.csv", all_results["latent"]
    )
    write_csv(
        args.output / "latent_mode_metrics_by_window.csv",
        all_results["modes"],
    )
    write_csv(
        args.output / "auxiliary_metrics_by_window.csv",
        all_results["auxiliary"],
    )
    write_csv(
        args.output / "model_selections.csv", all_results["selections"]
    )
    write_csv(
        args.output / "validation_candidates.csv",
        all_results["candidates"],
    )
    plot_criteria(args.output / "closure_criteria_matrix.png", criteria)
    plot_robustness(
        args.output / "closure_robustness_by_window.png",
        all_results["state"],
    )
    write_readme(args.output / "README.md", scored)
    summary = {
        "physical_source": str(args.physical),
        "windows": WINDOWS,
        "systems": SYSTEMS,
        "methods": METHODS,
        "delays": delays,
        "ranks": ranks,
        "criterion_note": (
            "closure_evidence_fraction is a transparent diagnostic score, "
            "not a proof of mathematical closure"
        ),
        "forecast_truth_used_as_input": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(augmented.json_safe(summary), indent=2),
        encoding="utf-8",
    )
    print(f"PASS: wrote E25 closure analysis to {args.output}", flush=True)


if __name__ == "__main__":
    main()
