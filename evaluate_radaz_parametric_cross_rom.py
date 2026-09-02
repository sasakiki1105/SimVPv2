"""Fit an Ez-conditioned physical cross-spectrum ROM and hold out E40.

The state is the 16-dimensional real/imaginary radial cross spectrum shared
by every PIC case.  Source trajectories are normalized case-wise over the
fit interval.  Their component means and log standard deviations, and the
Hankel-coordinate transition operator, are modeled as affine functions of
Ez.  E40 is not loaded until all source-only choices and models are locked.
"""

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

import analyze_radaz_augmented_physical_state_dynamics as augmented
import analyze_radaz_fourier_latent_to_physical_modes as physical_modes
import analyze_radaz_hankel_havok as hankel
import evaluate_radaz_rom_transfer as transfer


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "workdirs" / "compare_radaz_parametric_cross_rom_to_e40"
)
BASELINE = ROOT / "workdirs" / "compare_radaz_rom_transfer_e25_e30_e40"

FIT_START_US = 20.0
VALIDATION_START_US = 23.0
FORECAST_START_US = 24.0
FORECAST_END_US = 30.0
DELAY = 40
RANK = 30
RIDGES = (1.0e-2, 1.0, 100.0, 1.0e4, 1.0e6, 1.0e8, 1.0e10)
SPECTRAL_RADIUS_LIMIT = 0.999
SOURCE_FIELDS = (10, 20, 25, 30)
REGIMES = {
    "all_E10_E20_E25_E30": (10, 20, 25, 30),
    "high_E20_E25_E30": (20, 25, 30),
}

FEATURE_PATHS = {
    10: ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez10kvm_fourier_latent_dynamics"
    / "fourier_latent_features.h5",
    20: ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_electric_field_sweep_forcing"
    / "cases"
    / "E20kVm"
    / "fourier_latent_features.h5",
    25: ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_raw"
    / "fourier_latent_features.h5",
    30: ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_electric_field_sweep_forcing"
    / "cases"
    / "E30kVm"
    / "fourier_latent_features.h5",
    40: ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_electric_field_sweep_forcing"
    / "cases"
    / "E40kVm"
    / "fourier_latent_features.h5",
}


@dataclass
class PhysicalCase:
    ez_kvm: int
    time_us: np.ndarray
    frame: np.ndarray
    physical: augmented.PhysicalStates
    cross_flat: np.ndarray
    physical_path: Path


@dataclass
class AffineScaleLaw:
    electric_center: float
    electric_scale: float
    mean_coefficients: np.ndarray
    log_scale_coefficients: np.ndarray

    def predict(self, ez_kvm: float) -> tuple[np.ndarray, np.ndarray]:
        parameter = (ez_kvm - self.electric_center) / self.electric_scale
        features = np.asarray([1.0, parameter], dtype=np.float64)
        mean = features @ self.mean_coefficients
        scale = np.exp(features @ self.log_scale_coefficients)
        return mean, np.maximum(scale, 1.0e-12)


@dataclass
class ParametricHankel:
    delay: int
    rank: int
    dimensions: int
    delay_mean: np.ndarray
    basis: np.ndarray
    parametric_coefficients: np.ndarray
    pooled_coefficients: np.ndarray
    electric_center: float
    electric_scale: float

    def parameter(self, ez_kvm: float) -> float:
        return (ez_kvm - self.electric_center) / self.electric_scale

    def project(self, delay_vector: np.ndarray) -> np.ndarray:
        return (delay_vector - self.delay_mean) @ self.basis

    def reconstruct(self, coordinate: np.ndarray) -> np.ndarray:
        return coordinate @ self.basis.T + self.delay_mean


def write_csv(path: Path, rows: list[dict]) -> None:
    transfer.write_csv(path, rows)


def case_directory(ez_kvm: int) -> Path:
    name = f"2D_RadAz_Xe1p_Bx20mT_Ez{ez_kvm}kVm_dt15ps_out15ns"
    return transfer.RESULTS_ROOT / name / name


def feature_time(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        frames = np.asarray(handle["translator_frame"], dtype=np.int64)
        time_us = (
            np.asarray(handle["translator_time_s"], dtype=np.float64) * 1.0e6
        )
    return frames, time_us


def load_case(ez_kvm: int, cache: Path) -> PhysicalCase:
    frames, time_us = feature_time(FEATURE_PATHS[ez_kvm])
    if ez_kvm == 25:
        physical_path = (
            ROOT
            / "workdirs"
            / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_to_physical_modes"
            / "physical_fourier_targets.h5"
        )
    elif ez_kvm in (30, 40):
        physical_path = BASELINE / "physical_cache" / f"E{ez_kvm}_physical_fourier_targets.h5"
    else:
        physical_path = cache / f"E{ez_kvm}_physical_fourier_targets.h5"
        source = case_directory(ez_kvm) / "analysis_fields_uncompressed.h5"
        physical_modes.extract_physical_fourier(
            source,
            physical_path,
            frames,
            time_us,
            bands=8,
            maximum_mode=21,
        )
    physical = augmented.load_physical_states(physical_path)
    if not np.array_equal(frames, physical.frame):
        raise ValueError(f"E{ez_kvm}: frame mismatch")
    if not np.allclose(time_us, physical.time_us, atol=1.0e-10, rtol=0.0):
        raise ValueError(f"E{ez_kvm}: time mismatch")
    if len(time_us) < 1000 or time_us[-1] < 29.8:
        raise ValueError(f"E{ez_kvm}: incomplete time series")
    return PhysicalCase(
        ez_kvm,
        time_us,
        frames,
        physical,
        augmented.flatten_physical(physical)["cross"],
        physical_path,
    )


def mask(case: PhysicalCase, start: float, stop: float, inclusive=False):
    upper = case.time_us <= stop if inclusive else case.time_us < stop
    return (case.time_us >= start) & upper


def local_statistics(
    case: PhysicalCase, start: float, stop: float
) -> tuple[np.ndarray, np.ndarray]:
    values = case.cross_flat[mask(case, start, stop)]
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    return mean, np.where(scale > 1.0e-12, scale, 1.0)


def fit_scale_law(
    cases: list[PhysicalCase], start: float, stop: float
) -> tuple[AffineScaleLaw, dict[int, tuple[np.ndarray, np.ndarray]]]:
    electric = np.asarray([case.ez_kvm for case in cases], dtype=np.float64)
    center = float(np.mean(electric))
    scale = float(np.std(electric, ddof=1)) if len(electric) > 1 else 1.0
    parameter = (electric - center) / scale
    design = np.column_stack([np.ones(len(cases)), parameter])
    statistics = {
        case.ez_kvm: local_statistics(case, start, stop) for case in cases
    }
    means = np.stack([statistics[case.ez_kvm][0] for case in cases])
    scales = np.stack([statistics[case.ez_kvm][1] for case in cases])
    law = AffineScaleLaw(
        center,
        scale,
        np.linalg.lstsq(design, means, rcond=1.0e-10)[0],
        np.linalg.lstsq(design, np.log(scales), rcond=1.0e-10)[0],
    )
    return law, statistics


def standardized_interval(
    case: PhysicalCase,
    mean: np.ndarray,
    scale: np.ndarray,
    start: float,
    stop: float,
) -> np.ndarray:
    return (case.cross_flat[mask(case, start, stop)] - mean) / scale


def ridge_fit(
    features: np.ndarray,
    targets: np.ndarray,
    ridge: float,
    unpenalized: int,
):
    penalty = np.eye(features.shape[1], dtype=np.float64) * ridge
    penalty[-unpenalized:, -unpenalized:] = 0.0
    return np.linalg.solve(
        features.T @ features + penalty,
        features.T @ targets,
    )


def fit_parametric_hankel(
    trajectories: list[np.ndarray],
    electric_fields: list[float],
    delay: int,
    rank: int,
    ridge: float,
) -> ParametricHankel:
    delay_sets = [hankel.make_delay_vectors(values, delay) for values in trajectories]
    combined = np.concatenate(delay_sets, axis=0)
    delay_mean = np.mean(combined, axis=0)
    centered = combined - delay_mean
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    if rank > right.shape[0]:
        raise ValueError(f"rank={rank} exceeds available rank={right.shape[0]}")
    basis = right[:rank].T
    coordinate_sets = [(values - delay_mean) @ basis for values in delay_sets]
    electric = np.asarray(electric_fields, dtype=np.float64)
    electric_center = float(np.mean(electric))
    electric_scale = float(np.std(electric, ddof=1))
    parameter = (electric - electric_center) / electric_scale
    features = []
    pooled_features = []
    targets = []
    for coordinates, value in zip(coordinate_sets, parameter):
        current = coordinates[:-1]
        features.append(
            np.column_stack(
                [
                    current,
                    value * current,
                    np.ones(len(current)),
                    np.full(len(current), value),
                ]
            )
        )
        pooled_features.append(
            np.column_stack([current, np.ones(len(current))])
        )
        targets.append(coordinates[1:])
    target = np.concatenate(targets, axis=0)
    return ParametricHankel(
        delay,
        rank,
        trajectories[0].shape[1],
        delay_mean,
        basis,
        ridge_fit(
            np.concatenate(features, axis=0), target, ridge, unpenalized=2
        ),
        ridge_fit(
            np.concatenate(pooled_features, axis=0),
            target,
            ridge,
            unpenalized=1,
        ),
        electric_center,
        electric_scale,
    )


def rollout(
    model: ParametricHankel,
    history: np.ndarray,
    steps: int,
    ez_kvm: float,
    method: str,
) -> np.ndarray:
    delay_vector = hankel.make_delay_vectors(
        history[-model.delay :], model.delay
    )[0]
    coordinate = model.project(delay_vector)
    parameter = model.parameter(ez_kvm)
    if method == "parametric_affine_operator":
        matrix = (
            model.parametric_coefficients[: model.rank]
            + parameter
            * model.parametric_coefficients[model.rank : 2 * model.rank]
        )
        intercept = (
            model.parametric_coefficients[-2]
            + parameter * model.parametric_coefficients[-1]
        )
    elif method == "pooled_operator":
        matrix = model.pooled_coefficients[: model.rank]
        intercept = model.pooled_coefficients[-1]
    else:
        raise ValueError(method)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if spectral_radius > SPECTRAL_RADIUS_LIMIT:
        matrix = matrix * (SPECTRAL_RADIUS_LIMIT / spectral_radius)
    forecast = np.empty((steps, model.dimensions), dtype=np.float64)
    for index in range(steps):
        coordinate = coordinate @ matrix + intercept
        if not np.all(np.isfinite(coordinate)) or np.max(np.abs(coordinate)) > 1.0e8:
            forecast[index:] = np.nan
            break
        forecast[index] = model.reconstruct(coordinate)[: model.dimensions]
    return forecast


def transport_metrics(
    case: PhysicalCase,
    predicted_cross_flat: np.ndarray,
    forecast_start: float,
    forecast_stop: float,
) -> tuple[list[dict], list[dict]]:
    forecast_mask = mask(case, forecast_start, forecast_stop, inclusive=True)
    history_mask = mask(case, FIT_START_US, forecast_start)
    truth = case.physical.transport[forecast_mask]
    cross = augmented.unflatten_cross(predicted_cross_flat)
    prediction = augmented.transport_from_cross(
        cross, case.physical.macro_weights
    )
    persistence = np.repeat(
        case.physical.transport[case.time_us < forecast_start][-1:],
        len(truth),
        axis=0,
    )
    history_mean = np.repeat(
        np.mean(case.physical.transport[history_mask], axis=0, keepdims=True),
        len(truth),
        axis=0,
    )
    metrics = []
    rollouts = []
    for band_index, band in enumerate(transfer.MODE_BANDS):
        values = augmented.scalar_metrics(
            truth[:, band_index],
            prediction[:, band_index],
            persistence[:, band_index],
        )
        model_mse = float(
            np.mean((prediction[:, band_index] - truth[:, band_index]) ** 2)
        )
        mean_mse = float(
            np.mean((history_mean[:, band_index] - truth[:, band_index]) ** 2)
        )
        metrics.append(
            {
                "band": band,
                **values,
                "skill_vs_history_mean": (
                    1.0 - model_mse / mean_mse if mean_mse > 0.0 else float("-inf")
                ),
                "prediction_over_truth_temporal_std": float(
                    np.std(prediction[:, band_index], ddof=1)
                    / max(np.std(truth[:, band_index], ddof=1), np.finfo(float).tiny)
                ),
            }
        )
        for index, time_value in enumerate(case.time_us[forecast_mask]):
            rollouts.append(
                {
                    "band": band,
                    "time_us": float(time_value),
                    "truth_transport": float(truth[index, band_index]),
                    "predicted_transport": float(prediction[index, band_index]),
                    "persistence_transport": float(persistence[index, band_index]),
                    "history_mean_transport": float(history_mean[index, band_index]),
                }
            )
    return metrics, rollouts


def fit_regime(
    cases: list[PhysicalCase], ridge: float, fit_stop: float
) -> tuple[ParametricHankel, AffineScaleLaw]:
    law, statistics = fit_scale_law(cases, FIT_START_US, fit_stop)
    trajectories = [
        standardized_interval(
            case,
            *statistics[case.ez_kvm],
            FIT_START_US,
            fit_stop,
        )
        for case in cases
    ]
    model = fit_parametric_hankel(
        trajectories,
        [case.ez_kvm for case in cases],
        DELAY,
        RANK,
        ridge,
    )
    return model, law


def predict_target(
    model: ParametricHankel,
    law: AffineScaleLaw,
    target: PhysicalCase,
    method: str,
    scaling: str,
    forecast_start: float,
    forecast_stop: float,
) -> np.ndarray:
    if scaling == "ez_predicted":
        mean, scale = law.predict(target.ez_kvm)
    elif scaling == "target_history":
        mean, scale = local_statistics(target, FIT_START_US, forecast_start)
    else:
        raise ValueError(scaling)
    history = (target.cross_flat[target.time_us < forecast_start] - mean) / scale
    steps = int(np.count_nonzero(mask(target, forecast_start, forecast_stop, True)))
    standardized = rollout(
        model, history, steps, target.ez_kvm, method
    )
    return standardized * scale + mean


def select_ridge(cases: list[PhysicalCase]) -> tuple[float, list[dict]]:
    rows: list[dict] = []
    for ridge in RIDGES:
        fold_values = []
        for heldout in cases:
            training = [case for case in cases if case is not heldout]
            model, law = fit_regime(training, ridge, VALIDATION_START_US)
            prediction = predict_target(
                model,
                law,
                heldout,
                "parametric_affine_operator",
                "ez_predicted",
                VALIDATION_START_US,
                FORECAST_START_US,
            )
            metric_rows, _ = transport_metrics(
                heldout,
                prediction,
                VALIDATION_START_US,
                FORECAST_START_US,
            )
            score = float(np.mean([row["nrmse"] for row in metric_rows]))
            fold_values.append(score)
            rows.append(
                {
                    "ridge": ridge,
                    "heldout_ez_kvm": heldout.ez_kvm,
                    "mean_transport_nrmse": score,
                }
            )
        rows.append(
            {
                "ridge": ridge,
                "heldout_ez_kvm": "mean",
                "mean_transport_nrmse": float(np.mean(fold_values)),
            }
        )
    means = [row for row in rows if row["heldout_ez_kvm"] == "mean"]
    selected = min(means, key=lambda row: row["mean_transport_nrmse"])
    if not np.isfinite(float(selected["mean_transport_nrmse"])):
        raise RuntimeError("All source-only leave-one-Ez-out candidates failed")
    return float(selected["ridge"]), rows


def evaluate_locked(
    regime: str,
    model: ParametricHankel,
    law: AffineScaleLaw,
    target: PhysicalCase,
) -> tuple[list[dict], list[dict]]:
    metrics: list[dict] = []
    rollouts: list[dict] = []
    for method in ("pooled_operator", "parametric_affine_operator"):
        for scaling in ("ez_predicted", "target_history"):
            prediction = predict_target(
                model,
                law,
                target,
                method,
                scaling,
                FORECAST_START_US,
                FORECAST_END_US,
            )
            metric_rows, rollout_rows = transport_metrics(
                target,
                prediction,
                FORECAST_START_US,
                FORECAST_END_US,
            )
            for row in metric_rows:
                metrics.append(
                    {
                        "regime": regime,
                        "target": "E40",
                        "method": method,
                        "scaling": scaling,
                        **row,
                    }
                )
            for row in rollout_rows:
                rollouts.append(
                    {
                        "regime": regime,
                        "target": "E40",
                        "method": method,
                        "scaling": scaling,
                        **row,
                    }
                )
    return metrics, rollouts


def scale_diagnostics(
    regime: str, law: AffineScaleLaw, target: PhysicalCase
) -> dict:
    predicted_mean, predicted_scale = law.predict(target.ez_kvm)
    actual_mean, actual_scale = local_statistics(
        target, FIT_START_US, FORECAST_START_US
    )
    return {
        "regime": regime,
        "target": "E40",
        "mean_error_in_actual_sigma_rms": float(
            np.sqrt(np.mean(((predicted_mean - actual_mean) / actual_scale) ** 2))
        ),
        "median_predicted_over_actual_scale": float(
            np.median(predicted_scale / actual_scale)
        ),
        "mean_abs_log_scale_ratio": float(
            np.mean(np.abs(np.log(predicted_scale / actual_scale)))
        ),
        "maximum_predicted_over_actual_scale": float(
            np.max(predicted_scale / actual_scale)
        ),
    }


def plot_metrics(path: Path, rows: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    combinations = [
        (regime, method, scaling)
        for regime in REGIMES
        for method in ("pooled_operator", "parametric_affine_operator")
        for scaling in ("ez_predicted", "target_history")
    ]
    labels = [
        f"{('all' if regime.startswith('all_') else 'high')}\n"
        f"{('param' if method.startswith('parametric') else 'pooled')}/"
        f"{('strict' if scaling == 'ez_predicted' else 'history')}"
        for regime, method, scaling in combinations
    ]
    colors = ["#0072b2" if "parametric" in item[1] else "#999999" for item in combinations]
    for row_index, band in enumerate(transfer.MODE_BANDS):
        selected = [row for row in rows if row["band"] == band]
        correlation = []
        nrmse = []
        for regime, method, scaling in combinations:
            match = [
                row
                for row in selected
                if row["regime"] == regime
                and row["method"] == method
                and row["scaling"] == scaling
            ][0]
            correlation.append(float(match["correlation"]))
            nrmse.append(float(match["nrmse"]))
        x = np.arange(len(combinations))
        axes[row_index, 0].bar(x, correlation, color=colors)
        axes[row_index, 1].bar(x, nrmse, color=colors)
        axes[row_index, 0].axhline(0.0, color="#222222", linewidth=1.0)
        axes[row_index, 0].set_ylabel(f"{band}\ncorrelation")
        axes[row_index, 1].set_ylabel(f"{band}\nNRMSE (log)")
        axes[row_index, 1].set_yscale("log")
        for axis in axes[row_index]:
            axis.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle("E40 physical cross-spectrum ROM: explicit Ez conditioning")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_rollouts(path: Path, rows: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), sharex=True)
    colors = {
        "all_E10_E20_E25_E30": "#0072b2",
        "high_E20_E25_E30": "#d55e00",
    }
    for column, band in enumerate(transfer.MODE_BANDS):
        reference = [
            row
            for row in rows
            if row["band"] == band
            and row["regime"] == next(iter(REGIMES))
            and row["method"] == "parametric_affine_operator"
            and row["scaling"] == "ez_predicted"
        ]
        for row_index, scaling in enumerate(("ez_predicted", "target_history")):
            axis = axes[row_index, column]
            axis.plot(
                [row["time_us"] for row in reference],
                [row["truth_transport"] for row in reference],
                color="#111111",
                linewidth=1.8,
                label="PIC truth",
            )
            axis.plot(
                [row["time_us"] for row in reference],
                [row["history_mean_transport"] for row in reference],
                color="#999999",
                linestyle=":",
                label="history mean",
            )
            for regime in REGIMES:
                values = [
                    row
                    for row in rows
                    if row["band"] == band
                    and row["regime"] == regime
                    and row["method"] == "parametric_affine_operator"
                    and row["scaling"] == scaling
                ]
                axis.plot(
                    [row["time_us"] for row in values],
                    [row["predicted_transport"] for row in values],
                    color=colors[regime],
                    linewidth=1.0,
                    label=regime,
                )
            axis.set_title(f"{band}, {scaling}")
            axis.set_xlabel("target time (us)")
            axis.set_ylabel("modal transport")
            axis.grid(alpha=0.22)
            axis.legend(loc="lower right", fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_readme(output: Path, rows: list[dict], selected: dict[str, float]) -> None:
    lines = [
        "# Ez-conditioned physical cross-spectrum ROM",
        "",
        "## Protocol",
        "",
        "- State: 16 real/imaginary radial density--electric-field cross-spectrum components.",
        "- Parameter: explicit Ez through A(Ez) = A0 + Ez_normalized A1 in Hankel coordinates.",
        "- Sources: E10/E20/E25/E30, with an additional E20/E25/E30 high-field regime model.",
        "- Target: E40, not loaded until ridge selection and source models were locked.",
        "- Source-only scaling: component means and log standard deviations are fitted as affine functions of Ez.",
        "- Target-history scaling is reported separately as a few-shot diagnostic.",
        "- Model selection: source-condition leave-one-Ez-out over 23--24 us.",
        f"- Fixed delay/rank: {DELAY}/{RANK}.",
        "",
        "## Selected ridge",
        "",
    ]
    for regime, ridge in selected.items():
        lines.append(f"- `{regime}`: `{ridge:g}`")
    lines.extend(
        [
            "",
            "## E40 result",
            "",
            "| regime | method | scaling | band | corr | NRMSE | persistence skill | history-mean skill |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['regime']} | {row['method']} | {row['scaling']} | "
            f"{row['band']} | {row['correlation']:.3f} | {row['nrmse']:.3f} | "
            f"{row['skill_vs_persistence']:.3f} | "
            f"{row['skill_vs_history_mean']:.3f} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "physical_cache"
    cache.mkdir(parents=True, exist_ok=True)

    sources = {value: load_case(value, cache) for value in SOURCE_FIELDS}
    locked_models = {}
    locked_laws = {}
    selected_ridges = {}
    validation_rows = []
    for regime, values in REGIMES.items():
        cases = [sources[value] for value in values]
        ridge, rows = select_ridge(cases)
        selected_ridges[regime] = ridge
        for row in rows:
            validation_rows.append({"regime": regime, **row})
        model, law = fit_regime(cases, ridge, FORECAST_START_US)
        locked_models[regime] = model
        locked_laws[regime] = law
    write_csv(output / "source_leave_one_ez_out_ridge_selection.csv", validation_rows)
    lock = {
        "target_loaded_after_lock": True,
        "target": "E40",
        "source_fields_kvm": SOURCE_FIELDS,
        "regimes": REGIMES,
        "delay": DELAY,
        "rank": RANK,
        "spectral_radius_limit": SPECTRAL_RADIUS_LIMIT,
        "ridge_candidates": RIDGES,
        "selected_ridges": selected_ridges,
        "operator": "A(Ez)=A0+Ez_normalized*A1 with affine intercept",
        "scale_law": "component mean and log(std) affine in Ez",
        "fit_us": [FIT_START_US, FORECAST_START_US],
        "forecast_us": [FORECAST_START_US, FORECAST_END_US],
    }
    (output / "protocol_lock_before_e40_load.json").write_text(
        json.dumps(transfer.json_safe(lock), indent=2), encoding="utf-8"
    )
    print("[LOCK] source-only parametric models saved before E40 load", flush=True)

    target = load_case(40, cache)
    metrics = []
    rollouts = []
    scale_rows = []
    for regime in REGIMES:
        metric_rows, rollout_rows = evaluate_locked(
            regime, locked_models[regime], locked_laws[regime], target
        )
        metrics.extend(metric_rows)
        rollouts.extend(rollout_rows)
        scale_rows.append(
            scale_diagnostics(regime, locked_laws[regime], target)
        )
    write_csv(output / "e40_parametric_rom_metrics.csv", metrics)
    write_csv(output / "e40_parametric_rom_rollouts.csv", rollouts)
    write_csv(output / "e40_scale_law_diagnostics.csv", scale_rows)
    plot_metrics(output / "e40_parametric_rom_metrics.png", metrics)
    plot_rollouts(output / "e40_parametric_rom_rollouts.png", rollouts)
    write_readme(output, metrics, selected_ridges)
    print(f"[DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
