"""Test whether SimVP latent states add predictive information to transport.

For three rolling time splits, this script compares physical-only dynamics,
independent latent/physical dynamics, and a jointly identified coupled model.
The same test is run for explicit modal transport and complex cross spectra,
using both data-only and spectral-loss SimVP checkpoints.
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
import analyze_radaz_hankel_havok as hankel
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "compare_radaz_e25_physical_only_vs_coupled_rolling"
)
WINDOWS = {
    "fit12_16_forecast16_22": (12.0, 16.0, 22.0),
    "fit16_20_forecast20_26": (16.0, 20.0, 26.0),
    "fit20_24_forecast24_30": (20.0, 24.0, 30.0),
}
SYSTEMS = {
    "latent_only": ("latent",),
    "transport_only": ("transport",),
    "cross_only": ("cross",),
    "latent_transport_coupled": ("latent", "transport"),
    "latent_cross_coupled": ("latent", "cross"),
}
FAMILIES = {
    "transport": {
        "physical_system": "transport_only",
        "coupled_system": "latent_transport_coupled",
    },
    "cross": {
        "physical_system": "cross_only",
        "coupled_system": "latent_cross_coupled",
    },
}
METHODS = ("hankel_dmd", "havok_zero_forcing")


def dynamic_search_hankel(
    standardized: np.ndarray,
    time_us: np.ndarray,
    fit_start: float,
    fit_end: float,
    delays: list[int],
    ranks: list[int],
) -> tuple[dict, list[dict]]:
    validation_start = fit_end - 1.0
    subtrain_mask = (time_us >= fit_start) & (time_us < validation_start)
    validation_mask = (time_us >= validation_start) & (time_us < fit_end)
    subtrain = standardized[subtrain_mask]
    validation = standardized[validation_mask]
    persistence = np.repeat(subtrain[-1:, :], len(validation), axis=0)
    validation_time = time_us[validation_mask]
    dt_us = float(np.median(np.diff(time_us)))
    rows = []
    for delay in delays:
        delay_vectors = hankel.make_delay_vectors(subtrain, delay)
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
                    subtrain,
                    delay,
                    rank,
                    delay_vectors,
                    delay_mean,
                    right,
                    singular_values,
                )
                prediction = hankel.rollout_hankel(
                    model, subtrain, len(validation)
                )
                metrics, _ = reduced.evaluate_prediction(
                    validation, prediction, persistence, validation_time
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
        raise RuntimeError("Every rolling Hankel candidate failed")
    selected = min(
        finite,
        key=lambda row: (
            row["validation_mse"],
            row["delay"],
            row["rank"],
        ),
    )
    return selected, rows


def system_groups(
    system: str,
    latent: np.ndarray,
    physical: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    sources = {"latent": latent, **physical}
    return {name: sources[name] for name in SYSTEMS[system]}


def fit_systems(
    latent_subtrain: np.ndarray,
    latent_final: np.ndarray,
    physical_flat: dict[str, np.ndarray],
    time_us: np.ndarray,
    fit_start: float,
    fit_end: float,
    forecast_end: float,
    delays: list[int],
    ranks: list[int],
) -> tuple[dict, list[dict]]:
    validation_start = fit_end - 1.0
    subtrain_mask = (time_us >= fit_start) & (time_us < validation_start)
    fit_mask = (time_us >= fit_start) & (time_us < fit_end)
    forecast_mask = (time_us >= fit_end) & (time_us <= forecast_end)
    results = {}
    candidate_rows = []
    for system in SYSTEMS:
        sub_groups = system_groups(
            system, latent_subtrain, physical_flat
        )
        sub_scaler = augmented.GroupScaler.fit(sub_groups, subtrain_mask)
        sub_standardized = sub_scaler.transform(sub_groups)
        selected, candidates = dynamic_search_hankel(
            sub_standardized,
            time_us,
            fit_start,
            fit_end,
            delays,
            ranks,
        )
        for row in candidates:
            candidate_rows.append(
                {
                    "system": system,
                    "selected": (
                        row["delay"] == selected["delay"]
                        and row["rank"] == selected["rank"]
                    ),
                    **row,
                }
            )
        groups = system_groups(system, latent_final, physical_flat)
        scaler = augmented.GroupScaler.fit(groups, fit_mask)
        standardized = scaler.transform(groups)
        forecasts = augmented.fit_and_forecast(
            standardized,
            fit_mask,
            int(np.count_nonzero(forecast_mask)),
            int(selected["delay"]),
            int(selected["rank"]),
        )
        results[system] = {
            "selected": selected,
            "scaler": scaler,
            "groups": groups,
            "predictions": {
                method: scaler.inverse(prediction)
                for method, prediction in forecasts.items()
            },
        }
    for family, specification in FAMILIES.items():
        physical_system = specification["physical_system"]
        coupled_system = specification["coupled_system"]
        groups = system_groups(
            physical_system, latent_final, physical_flat
        )
        scaler = augmented.GroupScaler.fit(groups, fit_mask)
        standardized = scaler.transform(groups)
        coupled_selection = results[coupled_system]["selected"]
        forecasts = augmented.fit_and_forecast(
            standardized,
            fit_mask,
            int(np.count_nonzero(forecast_mask)),
            int(coupled_selection["delay"]),
            int(coupled_selection["rank"]),
        )
        results[f"{family}_physical_matched"] = {
            "selected": coupled_selection,
            "scaler": scaler,
            "groups": groups,
            "matched_to": coupled_system,
            "predictions": {
                method: scaler.inverse(prediction)
                for method, prediction in forecasts.items()
            },
        }
    return results, candidate_rows


def transport_prediction(
    family: str,
    prediction: dict[str, np.ndarray],
    radial_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    if family == "transport":
        return prediction["transport"], None
    cross = augmented.unflatten_cross(prediction["cross"])
    return augmented.transport_from_cross(cross, radial_weights), cross


def strategy_predictions(
    systems: dict,
    family: str,
    method: str,
) -> dict[str, dict[str, np.ndarray]]:
    specification = FAMILIES[family]
    latent = systems["latent_only"]["predictions"][method]["latent"]
    physical = systems[specification["physical_system"]]["predictions"][
        method
    ]
    coupled = systems[specification["coupled_system"]]["predictions"][
        method
    ]
    matched = systems[f"{family}_physical_matched"]["predictions"][method]
    independent = {"latent": latent, **physical}
    return {
        "physical_only": physical,
        "physical_matched": matched,
        "independent": independent,
        "coupled": coupled,
    }


def evaluate_physics(
    checkpoint: str,
    window: str,
    family: str,
    strategy: str,
    method: str,
    prediction: dict[str, np.ndarray],
    physical: augmented.PhysicalStates,
    fit_mask: np.ndarray,
    forecast_mask: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    truth_transport = physical.transport[forecast_mask]
    predicted_transport, predicted_cross = transport_prediction(
        family, prediction, physical.macro_weights
    )
    rows = []
    traces = {}
    for band_index, band in enumerate(augmented.MODE_BANDS):
        target = truth_transport[:, band_index]
        estimate = predicted_transport[:, band_index]
        persistence = np.repeat(
            physical.transport[fit_mask][-1, band_index], len(target)
        )
        metrics = augmented.scalar_metrics(target, estimate, persistence)
        row = {
            "checkpoint": checkpoint,
            "window": window,
            "family": family,
            "strategy": strategy,
            "method": method,
            "band": band,
            **{f"transport_{key}": value for key, value in metrics.items()},
        }
        if predicted_cross is not None:
            target_cross = physical.cross[forecast_mask, :, band_index]
            estimate_cross = predicted_cross[:, :, band_index]
            persistence_cross = np.repeat(
                physical.cross[fit_mask][-1:, :, band_index],
                len(target_cross),
                axis=0,
            )
            cross_metrics = augmented.scalar_metrics(
                target_cross, estimate_cross, persistence_cross
            )
            row.update(
                {
                    f"cross_{key}": value
                    for key, value in cross_metrics.items()
                }
            )
            row["cross_weighted_phase_mae_rad"] = (
                augmented.weighted_phase_mae(
                    target_cross[:, :, None],
                    estimate_cross[:, :, None],
                    physical.macro_weights,
                )
            )
        rows.append(row)
        traces[band] = estimate
    return rows, traces


def evaluate_joint(
    checkpoint: str,
    window: str,
    family: str,
    strategy: str,
    method: str,
    prediction: dict[str, np.ndarray],
    latent: np.ndarray,
    physical_flat: dict[str, np.ndarray],
    fit_mask: np.ndarray,
    forecast_mask: np.ndarray,
    time_us: np.ndarray,
) -> dict:
    physical_name = "transport" if family == "transport" else "cross"
    truth_groups = {
        "latent": latent,
        physical_name: physical_flat[physical_name],
    }
    scaler = augmented.GroupScaler.fit(truth_groups, fit_mask)
    truth = scaler.transform(truth_groups)[forecast_mask]
    predicted = scaler.transform(prediction)
    fit = scaler.transform(truth_groups)[fit_mask]
    persistence = np.repeat(fit[-1:, :], len(truth), axis=0)
    metrics, _ = reduced.evaluate_prediction(
        truth, predicted, persistence, time_us[forecast_mask]
    )
    return {
        "checkpoint": checkpoint,
        "window": window,
        "family": family,
        "strategy": strategy,
        "method": method,
        **metrics,
    }


def add_context(rows: list[dict], **context) -> list[dict]:
    return [{**context, **row} for row in rows]


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
    result = {
        "candidates": [],
        "physics": [],
        "latent": [],
        "modes": [],
        "joint": [],
        "rollouts": [],
        "selections": [],
    }
    for window, (fit_start, fit_end, forecast_end) in WINDOWS.items():
        validation_start = fit_end - 1.0
        subtrain_mask = (time_us >= fit_start) & (
            time_us < validation_start
        )
        fit_mask = (time_us >= fit_start) & (time_us < fit_end)
        forecast_mask = (time_us >= fit_end) & (
            time_us <= forecast_end
        )
        budget_name = specification["budget"]
        budget = block.BUDGETS[budget_name]
        _, sub_scores, _ = block.fit_block_models(
            features, subtrain_mask, budget
        )
        final_models, final_scores, _ = block.fit_block_models(
            features, fit_mask, budget
        )
        systems, candidates = fit_systems(
            sub_scores,
            final_scores,
            physical_flat,
            time_us,
            fit_start,
            fit_end,
            forecast_end,
            delays,
            ranks,
        )
        result["candidates"].extend(
            add_context(
                candidates,
                checkpoint=checkpoint,
                window=window,
                budget=budget_name,
            )
        )
        for system, values in systems.items():
            result["selections"].append(
                {
                    "checkpoint": checkpoint,
                    "window": window,
                    "system": system,
                    "delay": values["selected"]["delay"],
                    "rank": values["selected"]["rank"],
                    "history_us": values["selected"]["history_us"],
                }
            )

        forecast_time = time_us[forecast_mask]
        forecast_frames = frames[forecast_mask]
        for method in METHODS:
            latent_only = systems["latent_only"]["predictions"][method][
                "latent"
            ]
            latent_row = augmented.latent_metrics(
                checkpoint,
                "latent_only",
                method,
                final_scores[forecast_mask],
                latent_only,
                final_scores[fit_mask],
            )
            latent_row.update({"window": window, "family": "baseline"})
            result["latent"].append(latent_row)
            result["modes"].extend(
                add_context(
                    augmented.mode_metric_rows(
                        checkpoint,
                        "latent_only",
                        method,
                        final_models,
                        features[forecast_mask],
                        latent_only,
                    ),
                    window=window,
                    family="baseline",
                )
            )

            for family in FAMILIES:
                predictions = strategy_predictions(systems, family, method)
                for strategy, prediction in predictions.items():
                    physics_rows, traces = evaluate_physics(
                        checkpoint,
                        window,
                        family,
                        strategy,
                        method,
                        prediction,
                        physical,
                        fit_mask,
                        forecast_mask,
                    )
                    result["physics"].extend(physics_rows)
                    for index, time_value in enumerate(forecast_time):
                        row = {
                            "checkpoint": checkpoint,
                            "window": window,
                            "family": family,
                            "strategy": strategy,
                            "method": method,
                            "time_us": float(time_value),
                            "frame": int(forecast_frames[index]),
                        }
                        for band_index, band in enumerate(
                            augmented.MODE_BANDS
                        ):
                            row[f"truth_transport_{band}"] = float(
                                physical.transport[
                                    forecast_mask, band_index
                                ][index]
                            )
                            row[f"pred_transport_{band}"] = float(
                                traces[band][index]
                            )
                        result["rollouts"].append(row)

                    if strategy in {"physical_only", "physical_matched"}:
                        continue
                    latent_prediction = prediction["latent"]
                    latent_metric = augmented.latent_metrics(
                        checkpoint,
                        strategy,
                        method,
                        final_scores[forecast_mask],
                        latent_prediction,
                        final_scores[fit_mask],
                    )
                    latent_metric.update(
                        {"window": window, "family": family}
                    )
                    result["latent"].append(latent_metric)
                    result["modes"].extend(
                        add_context(
                            augmented.mode_metric_rows(
                                checkpoint,
                                strategy,
                                method,
                                final_models,
                                features[forecast_mask],
                                latent_prediction,
                            ),
                            window=window,
                            family=family,
                        )
                    )
                    result["joint"].append(
                        evaluate_joint(
                            checkpoint,
                            window,
                            family,
                            strategy,
                            method,
                            prediction,
                            final_scores,
                            physical_flat,
                            fit_mask,
                            forecast_mask,
                            time_us,
                        )
                    )
        print(
            f"[{checkpoint}] {window}: "
            + ", ".join(
                f"{name}=q{values['selected']['delay']}/r{values['selected']['rank']}"
                for name, values in systems.items()
            ),
            flush=True,
        )
    return result


def coupling_gain_rows(physics_rows: list[dict]) -> list[dict]:
    rows = []
    keys = ("checkpoint", "window", "family", "method", "band")
    groups = {}
    for row in physics_rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, {})[row["strategy"]] = row
    for key, strategies in groups.items():
        if "physical_only" not in strategies or "coupled" not in strategies:
            continue
        physical = strategies["physical_only"]
        matched = strategies.get("physical_matched")
        coupled = strategies["coupled"]
        row = {name: value for name, value in zip(keys, key)}
        row.update(
            {
                "correlation_gain": coupled["transport_correlation"]
                - physical["transport_correlation"],
                "skill_gain": coupled["transport_skill_vs_persistence"]
                - physical["transport_skill_vs_persistence"],
                "nrmse_reduction": physical["transport_nrmse"]
                - coupled["transport_nrmse"],
                "matched_correlation_gain": (
                    coupled["transport_correlation"]
                    - matched["transport_correlation"]
                    if matched is not None
                    else float("nan")
                ),
                "matched_skill_gain": (
                    coupled["transport_skill_vs_persistence"]
                    - matched["transport_skill_vs_persistence"]
                    if matched is not None
                    else float("nan")
                ),
            }
        )
        rows.append(row)
    return rows


def plot_family_metric(
    path: Path,
    rows: list[dict],
    family: str,
    metric: str,
    ylabel: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    labels = list(WINDOWS)
    x = np.arange(len(labels), dtype=float)
    styles = {
        ("physical_only", "hankel_dmd"): ("#56b4e9", "--"),
        ("physical_only", "havok_zero_forcing"): ("#009e73", "--"),
        ("coupled", "hankel_dmd"): ("#0072b2", "-"),
        ("coupled", "havok_zero_forcing"): ("#d55e00", "-"),
    }
    for row_index, checkpoint in enumerate(augmented.CHECKPOINTS):
        for column_index, band in enumerate(augmented.MODE_BANDS):
            axis = axes[row_index, column_index]
            for (strategy, method), (color, linestyle) in styles.items():
                values = []
                for window in labels:
                    match = [
                        row
                        for row in rows
                        if row["checkpoint"] == checkpoint
                        and row["window"] == window
                        and row["family"] == family
                        and row["strategy"] == strategy
                        and row["method"] == method
                        and row["band"] == band
                    ][0]
                    values.append(match[metric])
                axis.plot(
                    x,
                    values,
                    marker="o",
                    color=color,
                    linestyle=linestyle,
                    label=f"{strategy} / {method}",
                )
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.set_title(f"{checkpoint} / {band}")
            axis.set_ylabel(ylabel)
            axis.set_xticks(x)
            axis.set_xticklabels(
                [label.replace("fit", "").replace("_forecast", " -> ") for label in labels],
                rotation=20,
                ha="right",
            )
            axis.set_xlim(-0.3, len(labels) + 1.2)
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right", fontsize=8)
    figure.suptitle(f"{family}: physical-only versus coupled dynamics")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_coupling_gain(
    path: Path,
    rows: list[dict],
    metric: str = "correlation_gain",
    title: str = "Incremental transport information from SimVP latent state",
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    labels = list(WINDOWS)
    x = np.arange(len(labels), dtype=float)
    series = [
        ("transport", "hankel_dmd", "#0072b2"),
        ("transport", "havok_zero_forcing", "#56b4e9"),
        ("cross", "hankel_dmd", "#d55e00"),
        ("cross", "havok_zero_forcing", "#009e73"),
    ]
    for row_index, checkpoint in enumerate(augmented.CHECKPOINTS):
        for column_index, band in enumerate(augmented.MODE_BANDS):
            axis = axes[row_index, column_index]
            for index, (family, method, color) in enumerate(series):
                values = []
                for window in labels:
                    match = [
                        row
                        for row in rows
                        if row["checkpoint"] == checkpoint
                        and row["window"] == window
                        and row["family"] == family
                        and row["method"] == method
                        and row["band"] == band
                    ][0]
                    values.append(match[metric])
                axis.bar(
                    x + (index - 1.5) * 0.18,
                    values,
                    width=0.18,
                    color=color,
                    label=f"{family} / {method}",
                )
            axis.axhline(0.0, color="#111111", linewidth=1.0)
            axis.set_title(f"{checkpoint} / {band}")
            axis.set_ylabel("coupled - physical baseline correlation")
            axis.set_xticks(x)
            axis.set_xticklabels(
                [label.replace("fit", "").replace("_forecast", " -> ") for label in labels],
                rotation=20,
                ha="right",
            )
            axis.set_xlim(-0.5, len(labels) + 1.2)
            axis.grid(axis="y", alpha=0.25)
            axis.legend(loc="lower right", fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_representative_rollouts(path: Path, rows: list[dict]) -> None:
    window = "fit20_24_forecast24_30"
    family = "transport"
    method = "hankel_dmd"
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for row_index, checkpoint in enumerate(augmented.CHECKPOINTS):
        for column_index, band in enumerate(augmented.MODE_BANDS):
            axis = axes[row_index, column_index]
            for strategy, color in (
                ("physical_only", "#56b4e9"),
                ("coupled", "#0072b2"),
            ):
                selected = [
                    row
                    for row in rows
                    if row["checkpoint"] == checkpoint
                    and row["window"] == window
                    and row["family"] == family
                    and row["strategy"] == strategy
                    and row["method"] == method
                ]
                times = np.asarray([row["time_us"] for row in selected])
                prediction = np.asarray(
                    [row[f"pred_transport_{band}"] for row in selected]
                )
                axis.plot(times, prediction, color=color, label=strategy)
            truth = np.asarray(
                [row[f"truth_transport_{band}"] for row in selected]
            )
            axis.plot(times, truth, color="#111111", linewidth=2, label="truth")
            axis.set_title(f"{checkpoint} / {band}")
            axis.set_xlabel("time [us]")
            axis.set_ylabel("modal transport [m^-2 s^-1]")
            axis.set_xlim(24.0, 31.3)
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right")
    figure.suptitle("Transport-only versus latent+transport coupled rollout")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summary_table(gains: list[dict]) -> list[dict]:
    rows = []
    for checkpoint in augmented.CHECKPOINTS:
        for family in FAMILIES:
            for method in METHODS:
                for band in augmented.MODE_BANDS:
                    selected = [
                        row
                        for row in gains
                        if row["checkpoint"] == checkpoint
                        and row["family"] == family
                        and row["method"] == method
                        and row["band"] == band
                    ]
                    values = np.asarray(
                        [row["correlation_gain"] for row in selected]
                    )
                    matched_values = np.asarray(
                        [row["matched_correlation_gain"] for row in selected]
                    )
                    rows.append(
                        {
                            "checkpoint": checkpoint,
                            "family": family,
                            "method": method,
                            "band": band,
                            "mean_correlation_gain": float(np.mean(values)),
                            "median_correlation_gain": float(np.median(values)),
                            "positive_windows": int(np.count_nonzero(values > 0)),
                            "total_windows": len(values),
                            "min_gain": float(np.min(values)),
                            "max_gain": float(np.max(values)),
                            "mean_matched_correlation_gain": float(
                                np.mean(matched_values)
                            ),
                            "positive_matched_windows": int(
                                np.count_nonzero(matched_values > 0)
                            ),
                        }
                    )
    return rows


def write_readme(path: Path, summary_rows: list[dict]) -> None:
    lines = [
        "# E25 physical-only vs coupled rolling validation",
        "",
        "## 目的",
        "",
        "前解析で得た輸送予測が、transport/cross-spectrum自身の自己予測だけなのか、SimVP潜在状態が追加の予測情報を与えたのかを切り分ける。",
        "",
        "## モデル",
        "",
        "- `physical_only`: transportまたはcross-spectrumだけを自律予測。",
        "- `independent`: latentと物理状態を別々のHankel/HAVOKで予測して結合。物理予測はphysical-onlyと同一であり、結合のない基準。",
        "- `physical_matched`: 物理状態だけをcoupledと同じdelay/rankで予測する。hyperparameter差をlatent効果と誤認しないための診断。",
        "- `coupled`: latentと物理状態を同じ状態ベクトルへ入れ、交差結合を許して自律予測。",
        "",
        "内部選択は各4 us同定窓の最後1 us、予測は直後の6 usで行った。未来区間の真値は入力していない。",
        "",
        "## Coupling gainの3窓平均",
        "",
        "`correlation gain = coupled transport correlation - physical-only transport correlation`である。正ならlatent追加が輸送予測を改善した。",
        "",
        "| checkpoint | family | method | band | mean gain | positive windows | matched mean | matched positive | range |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['checkpoint']} | {row['family']} | {row['method']} | {row['band']} | "
            f"{row['mean_correlation_gain']:.4f} | {row['positive_windows']}/{row['total_windows']} | "
            f"{row['mean_matched_correlation_gain']:.4f} | {row['positive_matched_windows']}/{row['total_windows']} | "
            f"{row['min_gain']:.4f} to {row['max_gain']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 出力",
            "",
            "- `rolling_physics_metrics.csv`: 各窓の輸送・cross-spectrum指標。",
            "- `rolling_coupling_gain.csv`: coupledからphysical-onlyを引いた増分。",
            "- `rolling_coupling_summary.csv`: 3窓平均と正の窓数。",
            "- `rolling_latent_metrics.csv`: latent軌道指標。",
            "- `rolling_latent_mode_metrics.csv`: MTSI/ECDI潜在振幅指標。",
            "- `rolling_joint_metrics.csv`: independent/coupled結合状態全体の指標。",
            "- `transport_family_correlation.png`, `cross_family_correlation.png`: 時間窓別比較。",
            "- `coupling_gain_by_window.png`: latent追加による輸送相関の増減。",
            "- `coupling_gain_vs_matched_by_window.png`: coupledと同じdelay/rankを使った物理量のみモデルに対する増分。",
            "- `representative_transport_rollouts.png`: 最終窓の代表時系列。",
            "",
            "## 解釈上の注意",
            "",
            "coupledがphysical-onlyを複数窓で一貫して上回った場合、latentが輸送の将来に対する増分予測情報を持つ。ただし、これは介入を伴う物理的因果の証明ではなく、out-of-sample予測上の結合寄与である。",
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
    args.output.mkdir(parents=True, exist_ok=True)
    delays = [int(value) for value in args.delays.split(",")]
    ranks = [int(value) for value in args.ranks.split(",")]

    physical = augmented.load_physical_states(args.physical)
    physical_flat = augmented.flatten_physical(physical)
    combined = {
        "candidates": [],
        "physics": [],
        "latent": [],
        "modes": [],
        "joint": [],
        "rollouts": [],
        "selections": [],
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
        for key in combined:
            combined[key].extend(result[key])

    gains = coupling_gain_rows(combined["physics"])
    summaries = summary_table(gains)
    augmented.write_csv(
        args.output / "rolling_validation_candidates.csv",
        combined["candidates"],
    )
    augmented.write_csv(
        args.output / "rolling_model_selections.csv",
        combined["selections"],
    )
    augmented.write_csv(
        args.output / "rolling_physics_metrics.csv", combined["physics"]
    )
    augmented.write_csv(
        args.output / "rolling_latent_metrics.csv", combined["latent"]
    )
    augmented.write_csv(
        args.output / "rolling_latent_mode_metrics.csv", combined["modes"]
    )
    augmented.write_csv(
        args.output / "rolling_joint_metrics.csv", combined["joint"]
    )
    augmented.write_csv(
        args.output / "rolling_coupling_gain.csv", gains
    )
    augmented.write_csv(
        args.output / "rolling_coupling_summary.csv", summaries
    )
    augmented.write_csv(
        args.output / "rolling_transport_rollouts.csv", combined["rollouts"]
    )

    plot_family_metric(
        args.output / "transport_family_correlation.png",
        combined["physics"],
        "transport",
        "transport_correlation",
        "transport correlation",
    )
    plot_family_metric(
        args.output / "cross_family_correlation.png",
        combined["physics"],
        "cross",
        "transport_correlation",
        "transport correlation derived from cross-spectrum",
    )
    plot_coupling_gain(args.output / "coupling_gain_by_window.png", gains)
    plot_coupling_gain(
        args.output / "coupling_gain_vs_matched_by_window.png",
        gains,
        metric="matched_correlation_gain",
        title="Latent coupling gain after matching delay and rank",
    )
    plot_representative_rollouts(
        args.output / "representative_transport_rollouts.png",
        combined["rollouts"],
    )
    write_readme(args.output / "README.md", summaries)
    metadata = {
        "physical_source": str(args.physical),
        "windows": WINDOWS,
        "systems": SYSTEMS,
        "families": FAMILIES,
        "delays": delays,
        "ranks": ranks,
        "group_scaling": "per-component z-score, then 1/sqrt(group dimension)",
        "forecast_truth_used_as_input": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(augmented.json_safe(metadata), indent=2), encoding="utf-8"
    )
    print(f"Saved rolling coupling analysis to {args.output}", flush=True)


if __name__ == "__main__":
    main()
