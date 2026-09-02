"""Fit one E25 transport ROM and forecast 20--30 us without resetting it."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_blockwise_fourier_latent_dynamics as block
import analyze_radaz_e25_carrier_state_closure as carrier
import analyze_radaz_e25_transport_residual_closure as closure
import analyze_radaz_hankel_havok as hankel
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "workdirs" / "compare_radaz_e25_fixed_transport_closure"
TRAIN_START_US = 12.0
TRAIN_END_US = 18.0
VALIDATION_END_US = 20.0
FINAL_FIT_END_US = 20.0
FORECAST_END_US = 30.0
SEGMENTS = {
    "full20_30": (20.0, 30.0),
    "early20_24": (20.0, 24.0),
    "late24_30_no_reset": (24.0, 30.0),
}
PRIMARY_SYSTEM = "latent_phi_transport_residual"
PRIMARY_METHOD = "hankel_dmd"
EXPLORATORY_SYSTEM = "latent_phi_transport"
METHODS = closure.METHODS
SYSTEMS = closure.SYSTEMS
LABELS = closure.LABELS


def write_csv(path: Path, rows: list[dict]) -> None:
    closure.write_csv(path, rows)


def search_fixed_hankel(
    standardized: np.ndarray,
    time_us: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    delays: list[int],
    ranks: list[int],
) -> tuple[dict, list[dict]]:
    """Select delay/rank using only 12--18 -> 18--20 us."""
    train = standardized[train_mask]
    validation = standardized[validation_mask]
    persistence = np.repeat(train[-1:], len(validation), axis=0)
    validation_time = time_us[validation_mask]
    dt_us = float(np.median(np.diff(time_us)))
    rows = []
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
                model = block.make_rank_model(
                    train,
                    delay,
                    rank,
                    delay_vectors,
                    delay_mean,
                    right,
                    singular_values,
                )
                prediction = hankel.rollout_hankel(
                    model, train, len(validation)
                )
                metrics, _ = reduced.evaluate_prediction(
                    validation,
                    prediction,
                    persistence,
                    validation_time,
                )
                mse = float(metrics["standardized_mse"])
                radius = float(np.max(np.abs(model.eigenvalues)))
            except (ValueError, np.linalg.LinAlgError):
                mse = float("inf")
                radius = float("nan")
                metrics = {}
            rows.append(
                {
                    "delay": delay,
                    "history_us": delay * dt_us,
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
    finite = [row for row in rows if np.isfinite(row["validation_mse"])]
    if not finite:
        raise RuntimeError("Every fixed-ROM validation candidate failed")
    selected = min(
        finite,
        key=lambda row: (
            row["validation_mse"],
            row["delay"],
            row["rank"],
        ),
    )
    return selected, rows


def build_sources(
    raw: carrier.RawPhysical,
    features: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    carrier.CarrierBlock,
    closure.TransportResidual,
    list[dict],
]:
    modes = carrier.select_modes(raw.phi, raw.radial_weights, fit_mask)
    phi_block = carrier.build_carrier_block(
        "phi",
        raw.phi,
        modes,
        raw.radial_weights,
        raw.frame,
        fit_mask,
    )
    cross_block = carrier.build_carrier_block(
        "cross",
        raw.cross,
        modes,
        raw.radial_weights,
        raw.frame,
        fit_mask,
    )
    hybrid = closure.build_transport_residual(cross_block, fit_mask)
    _, latent, pca_rows = block.fit_block_models(
        features, fit_mask, block.BUDGETS["medium_20"]
    )
    return (
        closure.source_groups(latent, phi_block, hybrid),
        phi_block,
        hybrid,
        pca_rows,
    )


def mse_skill(
    truth: np.ndarray, prediction: np.ndarray, baseline: np.ndarray
) -> float:
    mse = float(np.mean((prediction - truth) ** 2))
    baseline_mse = float(np.mean((baseline - truth) ** 2))
    if not np.isfinite(mse) or baseline_mse <= np.finfo(float).tiny:
        return float("nan")
    return 1.0 - mse / baseline_mse


def add_fixed_transport_metrics(
    row: dict,
    truth: np.ndarray,
    prediction: np.ndarray,
    fit_transport: np.ndarray,
) -> None:
    history_mean = np.repeat(float(np.mean(fit_transport)), len(truth))
    persistence = np.repeat(float(fit_transport[-1]), len(truth))
    truth_std = float(np.std(truth, ddof=1))
    row.update(
        {
            "skill_vs_history_mean": mse_skill(
                truth, prediction, history_mean
            ),
            "prediction_std_over_truth_std": float(
                np.std(prediction, ddof=1)
                / max(truth_std, np.finfo(float).tiny)
            ),
            "persistence_value": float(persistence[0]),
            "history_mean_value": float(history_mean[0]),
        }
    )


def fixed_causality_audit(
    raw: carrier.RawPhysical,
) -> dict[str, float | bool]:
    fit_mask = (raw.time_us >= TRAIN_START_US) & (
        raw.time_us < FINAL_FIT_END_US
    )
    future_mask = (raw.time_us >= FINAL_FIT_END_US) & (
        raw.time_us <= FORECAST_END_US
    )
    modes = carrier.select_modes(raw.phi, raw.radial_weights, fit_mask)
    phi_reference = carrier.build_carrier_block(
        "phi", raw.phi, modes, raw.radial_weights, raw.frame, fit_mask
    )
    cross_reference = carrier.build_carrier_block(
        "cross", raw.cross, modes, raw.radial_weights, raw.frame, fit_mask
    )
    hybrid_reference = closure.build_transport_residual(
        cross_reference, fit_mask
    )

    phi_changed = raw.phi.copy()
    cross_changed = raw.cross.copy()
    phi_changed[future_mask] = (1.23 - 0.47j) * phi_changed[future_mask] + 2.0
    cross_changed[future_mask] = (-0.8 + 1.1j) * cross_changed[future_mask] - 7.0
    changed_modes = carrier.select_modes(
        phi_changed, raw.radial_weights, fit_mask
    )
    phi_perturbed = carrier.build_carrier_block(
        "phi",
        phi_changed,
        changed_modes,
        raw.radial_weights,
        raw.frame,
        fit_mask,
    )
    cross_perturbed = carrier.build_carrier_block(
        "cross",
        cross_changed,
        changed_modes,
        raw.radial_weights,
        raw.frame,
        fit_mask,
    )
    hybrid_perturbed = closure.build_transport_residual(
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
    passed = bool(np.array_equal(modes, changed_modes) and maximum <= 1.0e-10)
    if not passed:
        raise ValueError(f"Fixed-ROM causality audit failed: {maximum}")
    return {
        "forecast_truth_perturbed": True,
        "selected_modes_unchanged": True,
        "max_fit_state_absolute_difference": maximum,
        **{f"max_{name}_difference": value for name, value in differences.items()},
    }


def analyze(
    raw: carrier.RawPhysical,
    features: np.ndarray,
    delays: list[int],
    ranks: list[int],
) -> dict[str, list[dict] | dict]:
    train_mask = (raw.time_us >= TRAIN_START_US) & (
        raw.time_us < TRAIN_END_US
    )
    validation_mask = (raw.time_us >= TRAIN_END_US) & (
        raw.time_us < VALIDATION_END_US
    )
    final_fit_mask = (raw.time_us >= TRAIN_START_US) & (
        raw.time_us < FINAL_FIT_END_US
    )
    full_forecast_mask = (raw.time_us >= FINAL_FIT_END_US) & (
        raw.time_us <= FORECAST_END_US
    )

    validation_sources, _, _, _ = build_sources(raw, features, train_mask)
    final_sources, phi_block, hybrid, pca_rows = build_sources(
        raw, features, final_fit_mask
    )
    output: dict[str, list[dict] | dict] = {
        "selections": [],
        "candidates": [],
        "state": [],
        "physical": [],
        "traces": [],
        "representation": [],
    }
    forecast_count = int(np.count_nonzero(full_forecast_mask))
    forecast_times = raw.time_us[full_forecast_mask]

    for system in SYSTEMS:
        validation_groups = closure.groups_for_system(
            system, validation_sources
        )
        validation_scaler = augmented.GroupScaler.fit(
            validation_groups, train_mask
        )
        validation_standardized = validation_scaler.transform(
            validation_groups
        )
        selected, candidates = search_fixed_hankel(
            validation_standardized,
            raw.time_us,
            train_mask,
            validation_mask,
            delays,
            ranks,
        )
        for candidate in candidates:
            output["candidates"].append(
                {
                    "system": system,
                    "selected": bool(
                        candidate["delay"] == selected["delay"]
                        and candidate["rank"] == selected["rank"]
                    ),
                    **candidate,
                }
            )

        groups = closure.groups_for_system(system, final_sources)
        scaler = augmented.GroupScaler.fit(groups, final_fit_mask)
        standardized = scaler.transform(groups)
        predictions = augmented.fit_and_forecast(
            standardized,
            final_fit_mask,
            forecast_count,
            int(selected["delay"]),
            int(selected["rank"]),
        )
        output["selections"].append(
            {
                "system": system,
                "label": LABELS[system],
                "groups": "+".join(SYSTEMS[system]),
                "state_dimension": int(
                    sum(value.shape[1] for value in groups.values())
                ),
                "latent_source_dimension": int(
                    sum(row["total_features"] for row in pca_rows)
                ),
                "selected_modes": ",".join(map(str, phi_block.modes)),
                **selected,
            }
        )

        for method in METHODS:
            prediction_standardized = predictions[method]
            prediction_groups = scaler.inverse(prediction_standardized)
            for segment, (segment_start, segment_end) in SEGMENTS.items():
                global_segment_mask = (
                    (raw.time_us >= segment_start)
                    & (raw.time_us <= segment_end)
                    & full_forecast_mask
                )
                local_segment_mask = (forecast_times >= segment_start) & (
                    forecast_times <= segment_end
                )
                state_row = carrier.state_metrics(
                    standardized,
                    prediction_standardized[local_segment_mask],
                    final_fit_mask,
                    global_segment_mask,
                    raw.time_us,
                )
                output["state"].append(
                    {
                        "segment": segment,
                        "system": system,
                        "method": method,
                        **state_row,
                    }
                )
                segment_prediction = {
                    name: values[local_segment_mask]
                    for name, values in prediction_groups.items()
                }
                physical_rows, _ = closure.evaluate_system_physics(
                    system,
                    method,
                    segment_prediction,
                    phi_block,
                    hybrid,
                    final_fit_mask,
                    global_segment_mask,
                )
                for row in physical_rows:
                    row["segment"] = segment
                    if row["quantity"] == "selected_modal_transport":
                        add_fixed_transport_metrics(
                            row,
                            hybrid.transport[global_segment_mask],
                            segment_prediction["transport_direct"][:, 0],
                            hybrid.transport[final_fit_mask],
                        )
                    output["physical"].append(row)

            if (
                system in (PRIMARY_SYSTEM, EXPLORATORY_SYSTEM)
                and method == PRIMARY_METHOD
            ):
                truth_transport = hybrid.transport[full_forecast_mask]
                predicted_transport = prediction_groups["transport_direct"][:, 0]
                phi_prediction = phi_block.decode_circular(
                    prediction_groups["phi_circular"],
                    raw.frame[full_forecast_mask],
                )
                truth_phi_envelope = carrier.phi_envelope(
                    phi_block.original[full_forecast_mask],
                    phi_block.radial_weights,
                )
                predicted_phi_envelope = carrier.phi_envelope(
                    phi_prediction, phi_block.radial_weights
                )
                fit_transport = hybrid.transport[final_fit_mask]
                for index, time_us in enumerate(forecast_times):
                    output["traces"].append(
                        {
                            "system": system,
                            "method": method,
                            "time_us": float(time_us),
                            "transport_truth": float(truth_transport[index]),
                            "transport_prediction": float(
                                predicted_transport[index]
                            ),
                            "transport_persistence": float(fit_transport[-1]),
                            "transport_history_mean": float(
                                np.mean(fit_transport)
                            ),
                            "phi_envelope_truth": float(
                                truth_phi_envelope[index]
                            ),
                            "phi_envelope_prediction": float(
                                predicted_phi_envelope[index]
                            ),
                        }
                    )

    residual_reconstruction = (
        hybrid.residual_scores @ hybrid.residual_basis
        + hybrid.residual_mean
    )
    orthogonality = float(
        np.max(np.abs(residual_reconstruction @ hybrid.transport_functional))
        / max(
            float(np.max(np.abs(hybrid.transport))),
            np.finfo(float).tiny,
        )
    )
    output["representation"].append(
        {
            "fit_interval_us": "12-20",
            "selected_modes": ",".join(map(str, phi_block.modes)),
            "state_dimension_primary": int(
                sum(
                    final_sources[name].shape[1]
                    for name in SYSTEMS[PRIMARY_SYSTEM]
                )
            ),
            "cross_residual_components": hybrid.residual_scores.shape[1],
            "cross_residual_explained_variance": float(
                np.sum(hybrid.explained_variance)
            ),
            "transport_orthogonality_relative_error": orthogonality,
        }
    )
    output["audit"] = fixed_causality_audit(raw)
    return output


def make_summary(results: dict[str, list[dict] | dict]) -> list[dict]:
    physical = results["physical"]
    selections = {row["system"]: row for row in results["selections"]}
    rows = []
    for system in SYSTEMS:
        for method in METHODS:
            row = {
                "system": system,
                "label": LABELS[system],
                "method": method,
                "state_dimension": selections[system]["state_dimension"],
                "delay": selections[system]["delay"],
                "rank": selections[system]["rank"],
            }
            for segment in SEGMENTS:
                matches = [
                    item
                    for item in physical
                    if item["segment"] == segment
                    and item["system"] == system
                    and item["method"] == method
                    and item["quantity"] == "selected_modal_transport"
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Missing fixed transport result: {segment}/{system}/{method}"
                    )
                item = matches[0]
                for metric in (
                    "temporal_anomaly_correlation",
                    "nrmse",
                    "skill_vs_persistence",
                    "skill_vs_history_mean",
                    "prediction_std_over_truth_std",
                    "normalized_bias",
                ):
                    row[f"{segment}_{metric}"] = item[metric]
            rows.append(row)
    return rows


def plot_skill(path: Path, physical: list[dict]) -> None:
    segments = list(SEGMENTS)
    short = ["20-30", "20-24", "24-30\n(no reset)"]
    metric_rows = (
        ("skill_vs_persistence", "skill vs last-state persistence"),
        ("skill_vs_history_mean", "skill vs 12-20 us history mean"),
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(15.5, 10.0), sharex=True, constrained_layout=True
    )
    colors = plt.get_cmap("tab10")
    plotted_values = []
    for column, method in enumerate(METHODS):
        for row_index, (metric, ylabel) in enumerate(metric_rows):
            axis = axes[row_index, column]
            for system_index, system in enumerate(SYSTEMS):
                values = []
                for segment in segments:
                    match = next(
                        item
                        for item in physical
                        if item["segment"] == segment
                        and item["system"] == system
                        and item["method"] == method
                        and item["quantity"] == "selected_modal_transport"
                    )
                    values.append(match[metric])
                    plotted_values.append(match[metric])
                axis.plot(
                    short,
                    values,
                    marker="o",
                    linewidth=1.7,
                    color=colors(system_index),
                    label=LABELS[system],
                )
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.set_title(method.replace("_", " "))
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right", fontsize=8)
    finite = np.asarray([v for v in plotted_values if np.isfinite(v)])
    lower = min(-0.1, float(np.min(finite)) - 0.1)
    lower = max(lower, -5.0)
    for axis in axes.ravel():
        axis.set_ylim(lower, 1.05)
    figure.suptitle("E25 single fixed ROM: autonomous 20-30 us transport")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_fixed_rollout(
    path: Path,
    traces: list[dict],
    system: str,
    title_prefix: str,
) -> None:
    traces = [row for row in traces if row["system"] == system]
    time = np.asarray([row["time_us"] for row in traces])
    figure, axes = plt.subplots(
        2, 1, figsize=(12.0, 7.5), constrained_layout=True
    )
    axes[0].plot(
        time,
        [row["phi_envelope_truth"] for row in traces],
        color="#000000",
        label="PIC truth",
    )
    axes[0].plot(
        time,
        [row["phi_envelope_prediction"] for row in traces],
        color="#d55e00",
        label="fixed ROM",
    )
    axes[0].set_ylabel("selected MTSI phi envelope")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        time,
        [row["transport_truth"] for row in traces],
        color="#000000",
        label="PIC truth",
    )
    axes[1].plot(
        time,
        [row["transport_prediction"] for row in traces],
        color="#d55e00",
        label="fixed ROM",
    )
    axes[1].plot(
        time,
        [row["transport_persistence"] for row in traces],
        color="#0072b2",
        linestyle="--",
        label="persistence from 20 us",
    )
    axes[1].plot(
        time,
        [row["transport_history_mean"] for row in traces],
        color="#009e73",
        linestyle=":",
        label="12-20 us history mean",
    )
    axes[1].axvline(24.0, color="#777777", linewidth=0.8)
    axes[1].set_xlabel("time [us]")
    axes[1].set_ylabel("selected modal transport")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.25)
    figure.suptitle(f"E25 fixed {title_prefix} Hankel DMD: 10 us rollout")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_readme(
    path: Path,
    summary: list[dict],
    selections: list[dict],
    representation: list[dict],
    audit: dict,
    physical: list[dict],
) -> None:
    lookup = {(row["system"], row["method"]): row for row in summary}
    primary = lookup[(PRIMARY_SYSTEM, PRIMARY_METHOD)]
    direct = lookup[("transport_only", PRIMARY_METHOD)]
    exploratory = lookup[(EXPLORATORY_SYSTEM, PRIMARY_METHOD)]
    phi_rows = {
        row["system"]: row
        for row in physical
        if row["segment"] == "full20_30"
        and row["method"] == PRIMARY_METHOD
        and row["quantity"] == "selected_phi_coefficients"
        and row["system"] in (PRIMARY_SYSTEM, EXPLORATORY_SYSTEM)
    }
    lines = [
        "# E25 single fixed transport ROM closure test",
        "",
        "## Design / 設計",
        "",
        "Delay and rank are selected with 12-18 -> 18-20 us only. The physical/PCA representation, scaler, and transition operator are then fitted once on 12-20 us. The resulting model forecasts 20-30 us continuously; it is not reset or refitted at 24 us.",
        "",
        "delay/rankは12--18 -> 18--20 usだけで選択した。その後、物理/PCA表現、scaler、遷移operatorを12--20 usで一度だけ固定し、20--30 usを10 us連続で自律予測した。24 usで真値を再入力せず、再同定もしない。",
        "",
        "The primary state was fixed before this run as L+Pcirc+T+dX with Hankel DMD. Other states are reported as ablations, not selected from the test result.",
        "",
        "## Primary result / 主要結果",
        "",
        "| interval | transport corr | skill vs persistence | skill vs history mean | std ratio | normalized bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for segment, label in (
        ("full20_30", "20-30 us"),
        ("early20_24", "20-24 us"),
        ("late24_30_no_reset", "24-30 us, no reset"),
    ):
        lines.append(
            f"| {label} | {primary[f'{segment}_temporal_anomaly_correlation']:.3f} | "
            f"{primary[f'{segment}_skill_vs_persistence']:.3f} | "
            f"{primary[f'{segment}_skill_vs_history_mean']:.3f} | "
            f"{primary[f'{segment}_prediction_std_over_truth_std']:.3f} | "
            f"{primary[f'{segment}_normalized_bias']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"For the full 20-30 us rollout, direct T has persistence/history-mean skills {direct['full20_30_skill_vs_persistence']:.3f}/{direct['full20_30_skill_vs_history_mean']:.3f}; the coupled primary state has {primary['full20_30_skill_vs_persistence']:.3f}/{primary['full20_30_skill_vs_history_mean']:.3f}.",
            "",
            "A positive history-mean skill is required before this is treated as nontrivial time evolution rather than regression to a constant mean.",
            "",
            "## Exploratory ablation result / 探索的比較",
            "",
            f"After inspecting all test ablations, the simpler L+Pcirc+T Hankel DMD is best: full-interval transport correlation/skills versus persistence/history mean are {exploratory['full20_30_temporal_anomaly_correlation']:.3f}/{exploratory['full20_30_skill_vs_persistence']:.3f}/{exploratory['full20_30_skill_vs_history_mean']:.3f}, and its no-reset 24-30 us history-mean skill is {exploratory['late24_30_no_reset_skill_vs_history_mean']:.3f}.",
            f"Its phi coefficient correlation and phase MAE are {float(phi_rows[EXPLORATORY_SYSTEM]['coefficient_correlation']):.3f} and {float(phi_rows[EXPLORATORY_SYSTEM]['weighted_phase_mae_rad']):.3f} rad, compared with {float(phi_rows[PRIMARY_SYSTEM]['coefficient_correlation']):.3f} and {float(phi_rows[PRIMARY_SYSTEM]['weighted_phase_mae_rad']):.3f} rad for L+Pcirc+T+dX.",
            "",
            "This suggests that dX helped short rolling fits but is not required once a longer 12-20 us history is used to identify one stationary operator. The simpler state is an exploratory best model because this choice was made after reading the 20-30 us test; it needs a new untouched case or interval for confirmatory evaluation.",
            "",
            "全test ablationを見た後の探索的最良は、より単純なL+Pcirc+Tである。dXは短いrolling fitの弱い区間を補ったが、12--20 usの長い履歴から一つのoperatorを同定する場合には不要だった可能性が高い。ただしこれはtest結果を見て選んだため、確認的結論には新しい未使用caseまたは区間が必要である。",
            "",
            "## Scope / 主張範囲",
            "",
            "This is an operator-level holdout: forecast truth is absent from representation fitting, scaling, hyperparameter selection, and operator fitting. It is not a pristine research-level holdout because the state design was motivated by earlier E25 rolling results, and the frozen SimVP feature extractor was already trained for E25.",
            "",
            "これはoperator水準のholdoutである。予測真値は表現、正規化、hyperparameter選択、operator同定に使っていない。ただし状態設計自体は以前のE25 rolling結果を見て決めており、SimVP特徴抽出器もE25用学習済みなので、研究全体として完全未観測のconfirmatory holdoutではない。",
            "",
            f"Forecast perturbation audit maximum fit-state difference: {audit['max_fit_state_absolute_difference']:.3e}.",
            f"Transport-orthogonality relative error: {representation[0]['transport_orthogonality_relative_error']:.3e}.",
            "",
            "## Fixed model selections",
            "",
            "| state | delay | rank | validation skill |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in selections:
        lines.append(
            f"| {row['label']} | {row['delay']} | {row['rank']} | "
            f"{row['validation_skill_vs_persistence']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: fixed-model metrics by state and method.",
            "- `physical_metrics_by_segment.csv`: physical metrics for all intervals.",
            "- `state_metrics_by_segment.csv`: joint-state metrics.",
            "- `model_selection.csv` and `validation_candidates.csv`: validation-only choices.",
            "- `primary_time_series.csv` and `exploratory_best_time_series.csv`: uninterrupted 20-30 us rollouts.",
            "- `fixed_transport_skill.png`, `primary_fixed_rollout.png`, and `exploratory_best_fixed_rollout.png`: visual summaries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", type=Path, default=carrier.DEFAULT_PHYSICAL)
    parser.add_argument("--features", type=Path, default=carrier.DEFAULT_FEATURES)
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

    results = analyze(raw, features, delays, ranks)
    summary = make_summary(results)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "state_metrics_by_segment.csv", results["state"])
    write_csv(
        args.output / "physical_metrics_by_segment.csv", results["physical"]
    )
    write_csv(args.output / "model_selection.csv", results["selections"])
    write_csv(
        args.output / "validation_candidates.csv", results["candidates"]
    )
    write_csv(args.output / "representation.csv", results["representation"])
    write_csv(
        args.output / "primary_time_series.csv",
        [row for row in results["traces"] if row["system"] == PRIMARY_SYSTEM],
    )
    write_csv(
        args.output / "exploratory_best_time_series.csv",
        [
            row
            for row in results["traces"]
            if row["system"] == EXPLORATORY_SYSTEM
        ],
    )
    write_csv(args.output / "causality_audit.csv", [results["audit"]])
    plot_skill(
        args.output / "fixed_transport_skill.png", results["physical"]
    )
    plot_fixed_rollout(
        args.output / "primary_fixed_rollout.png",
        results["traces"],
        PRIMARY_SYSTEM,
        "L+Pcirc+T+dX",
    )
    plot_fixed_rollout(
        args.output / "exploratory_best_fixed_rollout.png",
        results["traces"],
        EXPLORATORY_SYSTEM,
        "L+Pcirc+T",
    )
    write_readme(
        args.output / "README.md",
        summary,
        results["selections"],
        results["representation"],
        results["audit"],
        results["physical"],
    )
    payload = {
        "status": "PASS",
        "train_interval_us": [TRAIN_START_US, TRAIN_END_US],
        "validation_interval_us": [TRAIN_END_US, VALIDATION_END_US],
        "final_fit_interval_us": [TRAIN_START_US, FINAL_FIT_END_US],
        "single_uninterrupted_forecast_interval_us": [
            FINAL_FIT_END_US,
            FORECAST_END_US,
        ],
        "reset_or_refit_during_forecast": False,
        "forecast_truth_used_as_input": False,
        "primary_system": PRIMARY_SYSTEM,
        "primary_method": PRIMARY_METHOD,
        "audit": results["audit"],
        "summary": summary,
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(closure.json_safe(payload), handle, indent=2, ensure_ascii=False)
    print(f"PASS: wrote fixed E25 closure test to {args.output}")


if __name__ == "__main__":
    main()
