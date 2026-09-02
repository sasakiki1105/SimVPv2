#!/usr/bin/env python3
"""Rolling-window ROM closure diagnosis for the B25 magnetic-sweep case.

Each protocol uses a four-microsecond identification window followed by a
six-microsecond autonomous holdout. Windows advance by two microseconds from
20--24 -> 24--30 us through 30--34 -> 34--40 us. Hyperparameters are selected
only from the last one microsecond of each identification window.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_radaz_hankel_havok as hankel
import analyze_radaz_latent_features as latent
import analyze_radaz_magnetic_sweep_rom as sweep
import analyze_radaz_reduced_dynamics as reduced


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT.parent
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b25_rolling_window_rom_0to40us"
)
DT_US = 0.015
WINDOWS = (
    (20.0, 24.0, 30.0),
    (22.0, 26.0, 32.0),
    (24.0, 28.0, 34.0),
    (26.0, 30.0, 36.0),
    (28.0, 32.0, 38.0),
    (30.0, 34.0, 40.0),
)
METHODS = ("standard_dmd", "hankel_dmd", "havok_zero_forcing")


@dataclass
class WindowResult:
    metrics: list[dict]
    diagnostic: dict
    time_us: np.ndarray
    truth: np.ndarray
    persistence: np.ndarray
    predictions: dict[str, np.ndarray]


def window_id(fit_start: float, fit_end: float, test_end: float) -> str:
    return f"fit{fit_start:g}-{fit_end:g}_test{fit_end:g}-{test_end:g}us"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_skill(mse: float, baseline_mse: float) -> float:
    if not np.isfinite(mse) or baseline_mse <= 1.0e-20:
        return float("-inf")
    return float(1.0 - mse / baseline_mse)


def evaluate_array(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    time_us: np.ndarray,
) -> tuple[dict, np.ndarray]:
    metrics, per_time = reduced.evaluate_prediction(
        truth, prediction, persistence, time_us
    )
    mean_mse = float(np.mean(truth * truth))
    metrics["skill_vs_training_mean"] = finite_skill(
        float(metrics["standardized_mse"]), mean_mse
    )
    return metrics, per_time


def validation_models(
    standardized: np.ndarray,
    time_us: np.ndarray,
    fit_start: float,
    fit_end: float,
    delays: list[int],
    ranks: list[int],
) -> dict[str, dict]:
    validation_start = fit_end - 1.0
    subtrain_mask = (time_us >= fit_start) & (time_us < validation_start)
    validation_mask = (time_us >= validation_start) & (time_us < fit_end)
    subtrain = standardized[subtrain_mask]
    validation = standardized[validation_mask]
    persistence = np.repeat(subtrain[-1][None, :], len(validation), axis=0)

    matrix, eigenvalues = reduced.fit_dmd(subtrain)
    prediction = reduced.rollout_linear(matrix, subtrain[-1], len(validation))
    standard_metrics, _ = evaluate_array(
        validation, prediction, persistence, time_us[validation_mask]
    )
    selected = {
        "standard_dmd": {
            "delay": 1,
            "rank": subtrain.shape[1],
            "validation_mse": float(standard_metrics["standardized_mse"]),
            "validation_skill_vs_persistence": float(
                standard_metrics["skill_vs_persistence"]
            ),
            "validation_correlation": float(
                standard_metrics["flattened_correlation"]
            ),
            "validation_spectral_radius": float(np.max(np.abs(eigenvalues))),
        }
    }

    trials = {"hankel_dmd": [], "havok_zero_forcing": []}
    for delay in delays:
        for rank in ranks:
            try:
                model = hankel.fit_hankel_dmd(subtrain, delay, rank)
                hankel_prediction = hankel.rollout_hankel(
                    model, subtrain, len(validation)
                )
                hankel_metrics, _ = evaluate_array(
                    validation,
                    hankel_prediction,
                    persistence,
                    time_us[validation_mask],
                )
                havok_model, _ = hankel.fit_havok(model, subtrain)
                havok_prediction = hankel.rollout_havok_zero_forcing(
                    model, havok_model, subtrain, len(validation)
                )
                havok_metrics, _ = evaluate_array(
                    validation,
                    havok_prediction,
                    persistence,
                    time_us[validation_mask],
                )
                trial_values = (
                    (
                        "hankel_dmd",
                        hankel_metrics,
                        float(np.max(np.abs(model.eigenvalues))),
                    ),
                    (
                        "havok_zero_forcing",
                        havok_metrics,
                        float(np.max(np.abs(np.linalg.eigvals(havok_model.matrix)))),
                    ),
                )
                for method, metrics, radius in trial_values:
                    trials[method].append(
                        {
                            "delay": delay,
                            "rank": rank,
                            "validation_mse": float(metrics["standardized_mse"]),
                            "validation_skill_vs_persistence": float(
                                metrics["skill_vs_persistence"]
                            ),
                            "validation_correlation": float(
                                metrics["flattened_correlation"]
                            ),
                            "validation_spectral_radius": radius,
                        }
                    )
            except (ValueError, np.linalg.LinAlgError):
                continue

    for method in ("hankel_dmd", "havok_zero_forcing"):
        finite = [
            trial
            for trial in trials[method]
            if np.isfinite(trial["validation_mse"])
        ]
        if not finite:
            raise RuntimeError(f"No valid {method} candidate")
        selected[method] = min(
            finite,
            key=lambda trial: (
                trial["validation_mse"],
                trial["delay"],
                trial["rank"],
            ),
        )
    return selected


def recurrence_diagnostic(
    standardized: np.ndarray,
    time_us: np.ndarray,
    fit_start: float,
    test_end: float,
) -> dict:
    interval = standardized[
        (time_us >= fit_start) & (time_us <= test_end + 1.0e-9)
    ]
    lag_min = max(2, int(round(0.30 / DT_US)))
    lag_max = min(int(round(3.00 / DT_US)), len(interval) // 2)
    candidates = []
    for lag in range(lag_min, lag_max + 1):
        left = interval[:-lag]
        right = interval[lag:]
        candidates.append(
            (
                float(np.sqrt(np.mean((left - right) ** 2))),
                float(np.corrcoef(left.ravel(), right.ravel())[0, 1]),
                lag,
            )
        )
    rmse, correlation, lag = min(candidates, key=lambda item: item[0])
    return {
        "best_recurrence_period_us": lag * DT_US,
        "best_recurrence_rmse": rmse,
        "best_recurrence_correlation": correlation,
    }


def evaluate_window(
    series: sweep.StateSeries,
    fit_start: float,
    fit_end: float,
    test_end: float,
    pca_components: int,
    delays: list[int],
    ranks: list[int],
) -> tuple[WindowResult, PCA]:
    fit_mask = (series.time_us >= fit_start) & (series.time_us < fit_end)
    test_mask = (
        (series.time_us >= fit_end)
        & (series.time_us <= test_end + 1.0e-9)
    )
    count = min(
        pca_components,
        int(np.count_nonzero(fit_mask)) - 1,
        series.features.shape[1],
    )
    pca = PCA(n_components=count, svd_solver="randomized", random_state=42)
    pca.fit(series.features[fit_mask])
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n95 = (
        int(np.searchsorted(cumulative, 0.95) + 1)
        if cumulative[-1] >= 0.95
        else len(cumulative)
    )
    components = min(n95, 20)
    scores = pca.transform(series.features)[:, :components].astype(np.float64)
    standardizer = reduced.fit_standardizer(scores[fit_mask])
    standardized = standardizer.transform(scores)
    fit = standardized[fit_mask]
    truth = standardized[test_mask]
    forecast_time = series.time_us[test_mask]
    persistence = np.repeat(fit[-1][None, :], len(truth), axis=0)
    selection = validation_models(
        standardized,
        series.time_us,
        fit_start,
        fit_end,
        delays,
        ranks,
    )

    predictions = {}
    spectral_radius = {}
    matrix, eigenvalues = reduced.fit_dmd(fit)
    predictions["standard_dmd"] = reduced.rollout_linear(
        matrix, fit[-1], len(truth)
    )
    spectral_radius["standard_dmd"] = float(np.max(np.abs(eigenvalues)))

    hankel_config = selection["hankel_dmd"]
    hankel_model = hankel.fit_hankel_dmd(
        fit, int(hankel_config["delay"]), int(hankel_config["rank"])
    )
    predictions["hankel_dmd"] = hankel.rollout_hankel(
        hankel_model, fit, len(truth)
    )
    spectral_radius["hankel_dmd"] = float(
        np.max(np.abs(hankel_model.eigenvalues))
    )

    havok_config = selection["havok_zero_forcing"]
    havok_hankel = hankel.fit_hankel_dmd(
        fit, int(havok_config["delay"]), int(havok_config["rank"])
    )
    havok_model, _ = hankel.fit_havok(havok_hankel, fit)
    predictions["havok_zero_forcing"] = hankel.rollout_havok_zero_forcing(
        havok_hankel, havok_model, fit, len(truth)
    )
    spectral_radius["havok_zero_forcing"] = float(
        np.max(np.abs(np.linalg.eigvals(havok_model.matrix)))
    )

    metrics_rows = []
    validation_winner = min(
        METHODS, key=lambda method: selection[method]["validation_mse"]
    )
    for method in METHODS:
        metrics, per_time = evaluate_array(
            truth, predictions[method], persistence, forecast_time
        )
        contiguous = 0
        for value in np.isfinite(per_time) & (per_time < 1.0):
            if not value:
                break
            contiguous += 1
        metrics_rows.append(
            {
                "representation": series.representation,
                "window": window_id(fit_start, fit_end, test_end),
                "fit_start_us": fit_start,
                "fit_end_us": fit_end,
                "test_end_us": test_end,
                "pca_components_95": n95,
                "rom_components": components,
                "method": method,
                "selected_by_validation": method == validation_winner,
                "delay": int(selection[method]["delay"]),
                "rank": int(selection[method]["rank"]),
                "validation_mse": float(selection[method]["validation_mse"]),
                "validation_skill_vs_persistence": float(
                    selection[method]["validation_skill_vs_persistence"]
                ),
                "standardized_rmse": float(metrics["standardized_rmse"]),
                "skill_vs_persistence": float(metrics["skill_vs_persistence"]),
                "skill_vs_training_mean": float(
                    metrics["skill_vs_training_mean"]
                ),
                "correlation": float(metrics["flattened_correlation"]),
                "spectral_radius": spectral_radius[method],
                "contiguous_horizon_rmse_lt_1_us": contiguous * DT_US,
            }
        )

    test = standardized[test_mask]
    diagnostic = {
        "representation": series.representation,
        "window": window_id(fit_start, fit_end, test_end),
        "fit_start_us": fit_start,
        "fit_end_us": fit_end,
        "test_end_us": test_end,
        "pca_components_95": n95,
        "rom_components": components,
        "variance_pc1": float(pca.explained_variance_ratio_[0]),
        "holdout_mean_shift_rms_fit_std": float(
            np.sqrt(np.mean(np.mean(test, axis=0) ** 2))
        ),
        "holdout_scale_rms_fit_std": float(np.sqrt(np.mean(np.var(test, axis=0)))),
        "fit_frame_delta_rms": float(np.sqrt(np.mean(np.diff(fit, axis=0) ** 2))),
        "holdout_frame_delta_rms": float(
            np.sqrt(np.mean(np.diff(test, axis=0) ** 2))
        ),
        **recurrence_diagnostic(
            standardized, series.time_us, fit_start, test_end
        ),
    }
    return (
        WindowResult(
            metrics_rows,
            diagnostic,
            forecast_time,
            truth,
            persistence,
            predictions,
        ),
        pca,
    )


def plot_metrics(rows: list[dict], path: Path) -> None:
    representations = ("physical_fourier", "encoder", "translator")
    markers = {"standard_dmd": "o", "hankel_dmd": "s", "havok_zero_forcing": "^"}
    colors = {"standard_dmd": "#0072B2", "hankel_dmd": "#D55E00", "havok_zero_forcing": "#009E73"}
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    for row_index, representation in enumerate(representations):
        subset_rep = [row for row in rows if row["representation"] == representation]
        for method in METHODS:
            subset = sorted(
                [row for row in subset_rep if row["method"] == method],
                key=lambda row: row["fit_end_us"],
            )
            x = [row["fit_end_us"] for row in subset]
            axes[row_index, 0].plot(
                x,
                np.clip([row["skill_vs_persistence"] for row in subset], -1, 1),
                marker=markers[method],
                color=colors[method],
                label=method,
            )
            axes[row_index, 1].plot(
                x,
                [row["correlation"] for row in subset],
                marker=markers[method],
                color=colors[method],
                label=method,
            )
        axes[row_index, 0].axhline(0, color="black", linewidth=0.8)
        axes[row_index, 1].axhline(0, color="black", linewidth=0.8)
        axes[row_index, 0].set_ylabel(f"{representation}\nskill vs persistence")
        axes[row_index, 1].set_ylabel(f"{representation}\ntrajectory correlation")
        axes[row_index, 0].set_ylim(-1.08, 1.08)
        axes[row_index, 1].set_ylim(-1.08, 1.08)
        for axis in axes[row_index]:
            axis.grid(alpha=0.25)
            axis.set_xlabel("forecast start (us)")
    axes[0, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle("B25 rolling ROM: 4 us fit followed by 6 us autonomous holdout\n(skill display clipped at -1; exact values in CSV)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_diagnostics(rows: list[dict], path: Path) -> None:
    representations = ("physical_fourier", "encoder", "translator")
    colors = {"physical_fourier": "#0072B2", "encoder": "#D55E00", "translator": "#009E73"}
    quantities = (
        ("pca_components_95", "PCs for 95% variance"),
        ("holdout_mean_shift_rms_fit_std", "Holdout mean shift (fit std)"),
        ("holdout_scale_rms_fit_std", "Holdout scale (fit std)"),
        ("best_recurrence_correlation", "Best nontrivial recurrence corr."),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True)
    for axis, (key, label) in zip(axes.ravel(), quantities):
        for representation in representations:
            subset = sorted(
                [row for row in rows if row["representation"] == representation],
                key=lambda row: row["fit_end_us"],
            )
            axis.plot(
                [row["fit_end_us"] for row in subset],
                [row[key] for row in subset],
                marker="o",
                color=colors[representation],
                label=representation,
            )
        axis.set_ylabel(label)
        axis.set_xlabel("forecast start (us)")
        axis.grid(alpha=0.25)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].axhline(1, color="black", linewidth=0.8)
    axes[0, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle("B25 rolling dimensionality, drift, and recurrence")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selected_pc1(
    results: dict[tuple[str, str], WindowResult],
    path: Path,
) -> None:
    representations = ("physical_fourier", "encoder", "translator")
    selected_windows = (WINDOWS[0], WINDOWS[-1])
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)
    for row_index, representation in enumerate(representations):
        for column, window in enumerate(selected_windows):
            fit_start, fit_end, test_end = window
            identifier = window_id(*window)
            result = results[(representation, identifier)]
            selected = next(
                row["method"]
                for row in result.metrics
                if row["selected_by_validation"]
            )
            axis = axes[row_index, column]
            axis.plot(result.time_us, result.truth[:, 0], color="black", label="truth")
            axis.plot(
                result.time_us,
                result.persistence[:, 0],
                color="#999999",
                linestyle=":",
                label="persistence",
            )
            axis.plot(
                result.time_us,
                result.predictions[selected][:, 0],
                color="#D55E00",
                label=selected,
            )
            axis.set_title(f"{representation} | {fit_start:g}-{fit_end:g} -> {fit_end:g}-{test_end:g} us")
            axis.set_xlabel("time (us)")
            axis.set_ylabel("standardized local PC1")
            axis.grid(alpha=0.25)
    axes[0, 1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(output: Path, metrics: list[dict], diagnostics: list[dict]) -> None:
    lines = []
    for representation in ("physical_fourier", "encoder", "translator"):
        for window in WINDOWS:
            identifier = window_id(*window)
            row = next(
                item
                for item in metrics
                if item["representation"] == representation
                and item["window"] == identifier
                and item["selected_by_validation"]
            )
            diagnostic = next(
                item
                for item in diagnostics
                if item["representation"] == representation
                and item["window"] == identifier
            )
            lines.append(
                f"| {representation} | {window[0]:g}-{window[1]:g} -> {window[1]:g}-{window[2]:g} | "
                f"{row['method']} | {row['skill_vs_persistence']:.4f} | "
                f"{row['skill_vs_training_mean']:.4f} | {row['correlation']:.4f} | "
                f"{diagnostic['pca_components_95']} | "
                f"{diagnostic['holdout_scale_rms_fit_std']:.3f} |"
            )
    late = {
        representation: next(
            row
            for row in metrics
            if row["representation"] == representation
            and row["window"] == window_id(*WINDOWS[-1])
            and row["selected_by_validation"]
        )
        for representation in ("physical_fourier", "encoder", "translator")
    }
    text = f"""# B25 rolling-window ROM closure diagnosis

Case: Bx=25 mT, Ez=10 kV/m, output interval 15 ns.

The original 2667-frame PIC HDF5 covers 0--39.99 us. The B20-trained SimVPv2
is frozen and reused without retraining. Target data use the fixed B20
training-only normalization with no clipping.

## Prospective rolling protocol

- four-microsecond identification window
- final one microsecond of that window for delay/rank and method selection
- six-microsecond strict autonomous holdout
- windows advance by two microseconds
- representations: physical radial-band Fourier, pooled encoder, pooled translator
- methods: standard DMD, Hankel DMD, HAVOK with zero future forcing

## Validation-selected result per window

| representation | fit -> holdout (us) | method | skill vs persistence | skill vs mean | correlation | PCs95 | holdout scale |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Method selection uses only the final one microsecond of each fit window. The
six-microsecond holdout is not used for method, delay, rank, PCA dimension, or
normalization selection. Exact results for every method remain in
`rolling_rom_metrics.csv`.

## Interpretation

The catastrophic 20--24 -> 24--30 us failure is not permanent. For the late
30--34 -> 34--40 us window, the validation-selected physical and encoder ROMs
recover positive skill against persistence. The encoder also has positive
skill against the training mean and a trajectory correlation of
{late['encoder']['correlation']:.3f}. The same three signs are already present
for the 28--32 -> 32--38 us encoder ROM.

This is partial rather than complete closure. The late encoder correlation is
moderate, its contiguous RMSE<1 horizon is only
{late['encoder']['contiguous_horizon_rmse_lt_1_us']:.3f} us, and the selected
translator still loses to the training mean. PCA dimensions and subspaces also
change between adjacent windows. B25 is therefore best described as a
time-dependent or intermittent closure regime: it becomes more predictable
after the early expansion, but no single stationary low-dimensional operator
describes the full 20--40 us interval.

## Files

- `rolling_rom_metrics.csv`
- `rolling_state_diagnostics.csv`
- `rolling_rollouts.h5`
- `rolling_rom_skill_and_correlation.png`
- `rolling_state_diagnostics.png`
- `early_vs_late_selected_pc1.png`
- `analysis_summary.json`
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pca-components", type=int, default=40)
    parser.add_argument("--delays", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 15, 20, 30])
    parser.add_argument("--overwrite-prepared", action="store_true")
    parser.add_argument("--overwrite-latent", action="store_true")
    parser.add_argument("--overwrite-physical", action="store_true")
    parser.add_argument("--skip-latent-extraction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sweep.FORECAST_END_US = 40.0
    normalized = sweep.prepare_normalized_input(
        25,
        args.output / "B25_b20norm_0to40us.h5",
        sweep.b20_input_path(),
        args.overwrite_prepared,
    )
    latent_path = args.output / "B25_latent_0to40us.h5"
    if args.skip_latent_extraction:
        if not latent_path.is_file():
            raise FileNotFoundError(latent_path)
    else:
        sweep.extract_latent_case(
            25,
            normalized,
            latent_path,
            latent.resolve_device(args.device),
            args.batch_size,
            args.overwrite_latent,
        )
    physical_path = sweep.physical_fourier_features(
        25,
        args.output / "B25_physical_fourier_0to40us.h5",
        8,
        48,
        args.overwrite_physical,
    )

    states = {
        "physical_fourier": sweep.load_physical(physical_path, 25),
        "encoder": sweep.load_latent(latent_path, 25, "encoder"),
        "translator": sweep.load_latent(latent_path, 25, "translator"),
    }
    metrics = []
    diagnostics = []
    results = {}
    previous_pca = {}
    for representation, series in states.items():
        for window in WINDOWS:
            fit_start, fit_end, test_end = window
            identifier = window_id(*window)
            print(f"[ROLL] {representation} {identifier}", flush=True)
            result, pca = evaluate_window(
                series,
                fit_start,
                fit_end,
                test_end,
                args.pca_components,
                args.delays,
                args.ranks,
            )
            if representation in previous_pca:
                prior = previous_pca[representation]
                dimensions = min(
                    result.diagnostic["rom_components"],
                    prior[1],
                )
                from scipy.linalg import subspace_angles

                angles = np.degrees(
                    subspace_angles(
                        prior[0].components_[:dimensions].T,
                        pca.components_[:dimensions].T,
                    )
                )
                result.diagnostic["previous_window_mean_subspace_angle_deg"] = float(
                    np.mean(angles)
                )
                result.diagnostic["previous_window_max_subspace_angle_deg"] = float(
                    np.max(angles)
                )
            else:
                result.diagnostic["previous_window_mean_subspace_angle_deg"] = float("nan")
                result.diagnostic["previous_window_max_subspace_angle_deg"] = float("nan")
            previous_pca[representation] = (
                pca,
                result.diagnostic["rom_components"],
            )
            metrics.extend(result.metrics)
            diagnostics.append(result.diagnostic)
            results[(representation, identifier)] = result

    write_csv(args.output / "rolling_rom_metrics.csv", metrics)
    write_csv(args.output / "rolling_state_diagnostics.csv", diagnostics)
    with h5py.File(args.output / "rolling_rollouts.h5", "w") as output:
        for (representation, identifier), result in results.items():
            group = output.require_group(f"{representation}/{identifier}")
            group.create_dataset("time_us", data=result.time_us)
            group.create_dataset("truth", data=result.truth, compression="gzip", compression_opts=4)
            group.create_dataset("persistence", data=result.persistence, compression="gzip", compression_opts=4)
            for method, prediction in result.predictions.items():
                group.create_dataset(method, data=prediction, compression="gzip", compression_opts=4)

    plot_metrics(metrics, args.output / "rolling_rom_skill_and_correlation.png")
    plot_diagnostics(diagnostics, args.output / "rolling_state_diagnostics.png")
    plot_selected_pc1(results, args.output / "early_vs_late_selected_pc1.png")
    write_readme(args.output, metrics, diagnostics)
    summary = {
        "status": "PASS",
        "case": sweep.case_name(25),
        "source_frames": 2667,
        "source_end_us": 39.99,
        "normalization": "fixed B20 training-only, no clipping",
        "windows": [list(window) for window in WINDOWS],
        "metrics": sweep.json_safe(metrics),
        "diagnostics": sweep.json_safe(diagnostics),
    }
    (args.output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] output={args.output}", flush=True)


if __name__ == "__main__":
    main()
