"""Diagnose affine state mismatch in RadAz ROM transfer.

The source ROM and its delay/rank choices are frozen from the preceding
E25/E30-to-E40 transfer experiment.  Target statistics are estimated only
from 20--24 us, before the autonomous 24--30 us forecast.  Component-wise
affine calibration is applied to the target history and inverted on output;
the source dynamics are never refit on target data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_hankel_havok as hankel
import evaluate_radaz_rom_transfer as transfer


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "workdirs" / "compare_radaz_rom_transfer_e25_e30_e40"
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "compare_radaz_rom_transfer_affine_calibration_e25_e30_e40"
)

CALIBRATIONS = {
    "none": (),
    "cross_affine": ("cross",),
    "latent_affine": ("latent",),
    "latent_cross_affine": ("latent", "cross"),
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def selections_from_rows(path: Path) -> dict[str, dict]:
    rows = read_csv(path)
    return {row["system"]: row for row in rows}


def calibrated_target_scaler(
    source_scaler: augmented.GroupScaler,
    target_groups: dict[str, np.ndarray],
    target: transfer.CaseData,
    calibrated_groups: tuple[str, ...],
) -> tuple[augmented.GroupScaler, list[dict]]:
    """Return a target-history scaler while retaining source group weights."""
    history_mask = transfer.time_mask(
        target, transfer.FIT_START_US, transfer.FORECAST_START_US
    )
    means = {name: value.copy() for name, value in source_scaler.means.items()}
    scales = {
        name: value.copy() for name, value in source_scaler.scales.items()
    }
    rows: list[dict] = []
    for name in source_scaler.names:
        values = target_groups[name][history_mask]
        target_mean = np.mean(values, axis=0)
        target_scale = np.std(values, axis=0, ddof=1)
        target_scale = np.where(target_scale > 1.0e-12, target_scale, 1.0)
        source_mean = source_scaler.means[name]
        source_scale = source_scaler.scales[name]
        use_target = name in calibrated_groups
        if use_target:
            means[name] = target_mean
            scales[name] = target_scale
        rows.append(
            {
                "group": name,
                "calibrated": use_target,
                "components": values.shape[1],
                "mean_shift_source_sigma_rms": float(
                    np.sqrt(np.mean(((target_mean - source_mean) / source_scale) ** 2))
                ),
                "median_target_over_source_scale": float(
                    np.median(target_scale / source_scale)
                ),
                "mean_abs_log_scale_ratio": float(
                    np.mean(np.abs(np.log(target_scale / source_scale)))
                ),
            }
        )
    scaler = augmented.GroupScaler(
        source_scaler.names,
        source_scaler.slices,
        means,
        scales,
        source_scaler.weights,
    )
    return scaler, rows


def predict_calibrated(
    system: str,
    delay: int,
    rank: int,
    sources: list[transfer.CaseData],
    target: transfer.CaseData,
    source_scores: dict[str, np.ndarray],
    target_scores: np.ndarray,
    calibration: str,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:
    source_groups = {
        case.name: transfer.groups_for_case(
            case, source_scores[case.name], system
        )
        for case in sources
    }
    target_groups = transfer.groups_for_case(target, target_scores, system)
    source_scaler = transfer.fit_source_scaler(
        source_groups,
        sources,
        transfer.FIT_START_US,
        transfer.FORECAST_START_US,
    )
    source_states = [
        source_scaler.transform(source_groups[case.name])[
            transfer.time_mask(
                case, transfer.FIT_START_US, transfer.FORECAST_START_US
            )
        ]
        for case in sources
    ]
    model = transfer.make_multitrajectory_hankel(source_states, delay, rank)
    havok_model = transfer.fit_multitrajectory_havok(model, source_states)

    applicable = tuple(
        name for name in CALIBRATIONS[calibration] if name in source_scaler.names
    )
    target_scaler, diagnostics = calibrated_target_scaler(
        source_scaler, target_groups, target, applicable
    )
    history_mask = target.time_us < transfer.FORECAST_START_US
    target_states = target_scaler.transform(target_groups)
    history = target_states[history_mask]
    forecast_mask = transfer.time_mask(
        target,
        transfer.FORECAST_START_US,
        transfer.FORECAST_END_US,
        inclusive=True,
    )
    steps = int(np.count_nonzero(forecast_mask))
    standardized = {
        "hankel_dmd": hankel.rollout_hankel(model, history, steps),
        "havok_zero_forcing": hankel.rollout_havok_zero_forcing(
            model, havok_model, history, steps
        ),
    }
    predictions = {
        method: target_scaler.inverse(values)
        for method, values in standardized.items()
    }
    return predictions, diagnostics


def run_transfer(
    sources: list[transfer.CaseData],
    target: transfer.CaseData,
    selections: dict[str, dict],
) -> dict[str, list[dict]]:
    source_label = "+".join(case.name for case in sources)
    pca_models, _ = transfer.fit_shared_block_pca(
        sources, transfer.FIT_START_US, transfer.FORECAST_START_US
    )
    source_scores = {
        case.name: transfer.transform_block_pca(pca_models, case.features)
        for case in sources
    }
    target_scores = transfer.transform_block_pca(pca_models, target.features)
    metrics: list[dict] = []
    rollouts: list[dict] = []
    diagnostics: list[dict] = []
    variants = (
        ("cross_only_matched", "cross_only", ("none", "cross_affine")),
        (
            "latent_cross_coupled",
            "latent_cross",
            tuple(CALIBRATIONS),
        ),
    )
    for system_label, system, calibration_names in variants:
        selection = selections["latent_cross"]
        delay = int(selection["delay"])
        rank = int(selection["rank"])
        for calibration in calibration_names:
            predictions, calibration_rows = predict_calibrated(
                system,
                delay,
                rank,
                sources,
                target,
                source_scores,
                target_scores,
                calibration,
            )
            for row in calibration_rows:
                diagnostics.append(
                    {
                        "source": source_label,
                        "target": target.name,
                        "system": system_label,
                        "calibration": calibration,
                        **row,
                    }
                )
            for method, prediction in predictions.items():
                metric_rows, rollout_rows = transfer.evaluate_predictions(
                    source_label,
                    target,
                    system_label,
                    method,
                    delay,
                    rank,
                    prediction,
                )
                calibration_mask = transfer.time_mask(
                    target,
                    transfer.FIT_START_US,
                    transfer.FORECAST_START_US,
                )
                history_mean = np.mean(
                    target.physical.transport[calibration_mask], axis=0
                )
                for row in metric_rows:
                    row["calibration"] = calibration
                    row["target_calibration_us"] = "20--24"
                    row["target_forecast_truth_used"] = False
                for row in rollout_rows:
                    row["calibration"] = calibration
                    band_index = list(transfer.MODE_BANDS).index(row["band"])
                    row["history_mean_transport"] = float(
                        history_mean[band_index]
                    )
                metrics.extend(metric_rows)
                rollouts.extend(rollout_rows)
    return {
        "metrics": metrics,
        "rollouts": rollouts,
        "diagnostics": diagnostics,
    }


def metric_lookup(
    rows: list[dict], source: str, target: str, calibration: str, band: str
) -> dict:
    matches = [
        row
        for row in rows
        if row["source"] == source
        and row["target"] == target
        and row["system"] == "latent_cross_coupled"
        and row["method"] == "havok_zero_forcing"
        and row["calibration"] == calibration
        and row["band"] == band
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one metric row for {source}, {target}, "
            f"{calibration}, {band}; got {len(matches)}"
        )
    return matches[0]


def add_gain_rows(
    rows: list[dict], rollout_rows: list[dict]
) -> list[dict]:
    gains: list[dict] = []
    pairs = sorted({(row["source"], row["target"]) for row in rows})
    for source, target in pairs:
        for band in transfer.MODE_BANDS:
            base = metric_lookup(rows, source, target, "none", band)
            for calibration in CALIBRATIONS:
                current = metric_lookup(rows, source, target, calibration, band)
                rollout = [
                    row
                    for row in rollout_rows
                    if row["source"] == source
                    and row["target"] == target
                    and row["system"] == "latent_cross_coupled"
                    and row["method"] == "havok_zero_forcing"
                    and row["calibration"] == calibration
                    and row["band"] == band
                ]
                truth = np.asarray(
                    [float(row["truth_transport"]) for row in rollout]
                )
                prediction = np.asarray(
                    [float(row["predicted_transport"]) for row in rollout]
                )
                history_mean = np.asarray(
                    [float(row["history_mean_transport"]) for row in rollout]
                )
                model_mse = float(np.mean((prediction - truth) ** 2))
                history_mean_mse = float(
                    np.mean((history_mean - truth) ** 2)
                )
                truth_std = float(np.std(truth, ddof=1))
                prediction_std = float(np.std(prediction, ddof=1))
                gains.append(
                    {
                        "source": source,
                        "target": target,
                        "band": band,
                        "calibration": calibration,
                        "correlation": float(
                            current["transport_correlation"]
                        ),
                        "correlation_gain_vs_none": float(
                            current["transport_correlation"]
                        )
                        - float(base["transport_correlation"]),
                        "nrmse": float(current["transport_nrmse"]),
                        "nrmse_ratio_vs_none": float(
                            current["transport_nrmse"]
                        )
                        / float(base["transport_nrmse"]),
                        "skill_vs_persistence": float(
                            current["transport_skill_vs_persistence"]
                        ),
                        "skill_vs_history_mean": (
                            1.0 - model_mse / history_mean_mse
                            if history_mean_mse > 0.0
                            else float("-inf")
                        ),
                        "history_mean_nrmse": float(
                            np.sqrt(history_mean_mse)
                            / max(truth_std, np.finfo(float).tiny)
                        ),
                        "prediction_over_truth_temporal_std": (
                            prediction_std
                            / max(truth_std, np.finfo(float).tiny)
                        ),
                    }
                )
    return gains


def plot_gain(path: Path, rows: list[dict], target: str) -> None:
    selected = [row for row in rows if row["target"] == target]
    sources = sorted({row["source"] for row in selected})
    calibrations = list(CALIBRATIONS)
    colors = {
        "none": "#666666",
        "cross_affine": "#0072b2",
        "latent_affine": "#e69f00",
        "latent_cross_affine": "#009e73",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), sharex="col")
    x = np.arange(len(sources), dtype=np.float64)
    width = 0.19
    for row_index, band in enumerate(transfer.MODE_BANDS):
        for calibration_index, calibration in enumerate(calibrations):
            values = []
            nrmse = []
            for source in sources:
                match = [
                    row
                    for row in selected
                    if row["source"] == source
                    and row["band"] == band
                    and row["calibration"] == calibration
                ][0]
                values.append(match["correlation"])
                nrmse.append(match["nrmse"])
            offset = (calibration_index - 1.5) * width
            axes[row_index, 0].bar(
                x + offset,
                values,
                width,
                color=colors[calibration],
                label=calibration,
            )
            axes[row_index, 1].bar(
                x + offset,
                nrmse,
                width,
                color=colors[calibration],
                label=calibration,
            )
        axes[row_index, 0].axhline(0.0, color="#222222", linewidth=1.0)
        axes[row_index, 0].set_ylabel(f"{band}\ncorrelation")
        axes[row_index, 1].set_ylabel(f"{band}\nNRMSE")
        for axis in axes[row_index]:
            axis.set_xticks(x, sources)
            axis.grid(axis="y", alpha=0.25)
            axis.legend(loc="upper right", fontsize=8)
    figure.suptitle(
        f"{target}: target-history affine calibration of latent+cross ROM"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_rollout(path: Path, rows: list[dict], target: str) -> None:
    selected = [
        row
        for row in rows
        if row["target"] == target
        and row["system"] == "latent_cross_coupled"
        and row["method"] == "havok_zero_forcing"
    ]
    sources = sorted({row["source"] for row in selected})
    figure, axes = plt.subplots(
        len(sources), 2, figsize=(13.0, 3.4 * len(sources)), sharex=True
    )
    axes = np.asarray(axes).reshape(len(sources), 2)
    colors = {
        "none": "#666666",
        "cross_affine": "#0072b2",
        "latent_affine": "#e69f00",
        "latent_cross_affine": "#009e73",
    }
    for row_index, source in enumerate(sources):
        for column, band in enumerate(transfer.MODE_BANDS):
            axis = axes[row_index, column]
            base = [
                row
                for row in selected
                if row["source"] == source
                and row["band"] == band
                and row["calibration"] == "none"
            ]
            axis.plot(
                [float(row["time_us"]) for row in base],
                [float(row["truth_transport"]) for row in base],
                color="#111111",
                linewidth=1.8,
                label="PIC truth",
            )
            axis.plot(
                [float(row["time_us"]) for row in base],
                [float(row["persistence_transport"]) for row in base],
                color="#999999",
                linestyle=":",
                linewidth=1.2,
                label="persistence",
            )
            for calibration in CALIBRATIONS:
                values = [
                    row
                    for row in selected
                    if row["source"] == source
                    and row["band"] == band
                    and row["calibration"] == calibration
                ]
                axis.plot(
                    [float(row["time_us"]) for row in values],
                    [float(row["predicted_transport"]) for row in values],
                    color=colors[calibration],
                    linewidth=1.0,
                    label=calibration,
                )
            axis.set_title(f"{source} -> {target}, {band}")
            axis.set_xlabel("target time (us)")
            axis.set_ylabel("modal transport")
            axis.grid(alpha=0.22)
            axis.legend(loc="lower right", fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def format_table(rows: list[dict], target: str) -> list[str]:
    lines = [
        "| source | band | calibration | corr | NRMSE | NRMSE/none | persistence skill | history-mean skill |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["target"] != target:
            continue
        lines.append(
            f"| {row['source']} | {row['band']} | {row['calibration']} | "
            f"{row['correlation']:.3f} | {row['nrmse']:.3f} | "
            f"{row['nrmse_ratio_vs_none']:.3f} | "
            f"{row['skill_vs_persistence']:.3f} | "
            f"{row['skill_vs_history_mean']:.3f} |"
        )
    return lines


def write_readme(output: Path, gains: list[dict]) -> None:
    lines = [
        "# Target-history affine calibration of RadAz ROM transfer",
        "",
        "## Purpose",
        "",
        "This experiment tests whether the failed electric-field transfer was mainly a component-wise offset/scale mismatch. The source PCA, source scaler, delay/rank, Hankel DMD, and HAVOK dynamics remain source-frozen. Only target data from 20--24 us estimates affine mean and standard deviation; 24--30 us remains an untouched autonomous forecast interval.",
        "",
        "The four predefined variants are `none`, `cross_affine`, `latent_affine`, and `latent_cross_affine`. A large improvement from `cross_affine` would support a scale-mismatch explanation. Failure after `latent_cross_affine` indicates that changing the first two moments is insufficient and the dynamics themselves are condition dependent.",
        "",
        "## Development: E25 to E30",
        "",
        *format_table(gains, "E30"),
        "",
        "## Final diagnostic: E40",
        "",
        *format_table(gains, "E40"),
        "",
        "## Interpretation",
        "",
        "See the generated metrics and plots. This is a few-shot target-history calibration diagnostic, not strict zero-shot transfer, because target statistics from 20--24 us are used.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = BASELINE / "physical_cache"

    cases = {
        name: transfer.load_case(name, cache) for name in ("E25", "E30")
    }
    development_selections = selections_from_rows(
        BASELINE / "development_e25_to_e30_selections.csv"
    )
    development = run_transfer(
        [cases["E25"]], cases["E30"], development_selections
    )
    for key, rows in development.items():
        transfer.write_csv(output / f"development_e25_to_e30_{key}.csv", rows)

    lock = json.loads(
        (BASELINE / "final_e40_protocol_lock_before_target_load.json").read_text(
            encoding="utf-8"
        )
    )
    locked = lock["selections"]
    cases["E40"] = transfer.load_case("E40", cache)
    final_combined = {"metrics": [], "rollouts": [], "diagnostics": []}
    for names in (("E25",), ("E30",), ("E25", "E30")):
        label = "+".join(names)
        result = run_transfer(
            [cases[name] for name in names], cases["E40"], locked[label]
        )
        for key in final_combined:
            final_combined[key].extend(result[key])
    for key, rows in final_combined.items():
        transfer.write_csv(output / f"final_e40_{key}.csv", rows)

    all_metrics = development["metrics"] + final_combined["metrics"]
    all_rollouts = development["rollouts"] + final_combined["rollouts"]
    gains = add_gain_rows(all_metrics, all_rollouts)
    transfer.write_csv(output / "affine_calibration_comparison.csv", gains)
    plot_gain(output / "development_e30_affine_metrics.png", gains, "E30")
    plot_gain(output / "final_e40_affine_metrics.png", gains, "E40")
    plot_rollout(
        output / "development_e30_affine_rollouts.png", all_rollouts, "E30"
    )
    plot_rollout(
        output / "final_e40_affine_rollouts.png", all_rollouts, "E40"
    )
    summary = {
        "target_calibration_us": [
            transfer.FIT_START_US,
            transfer.FORECAST_START_US,
        ],
        "target_forecast_us": [
            transfer.FORECAST_START_US,
            transfer.FORECAST_END_US,
        ],
        "target_forecast_truth_used": False,
        "source_dynamics_refit_on_target": False,
        "source_selections_reused": True,
        "calibrations_predefined": CALIBRATIONS,
    }
    (output / "summary.json").write_text(
        json.dumps(transfer.json_safe(summary), indent=2), encoding="utf-8"
    )
    write_readme(output, gains)
    print(f"[DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
