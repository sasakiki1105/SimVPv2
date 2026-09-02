#!/usr/bin/env python3
"""Test temporal direction in the B30 high-to-low spectral reorganization.

This is a pre-rePIC diagnostic. It uses the existing three-component time
series and compares B30 with B25. The results can show temporal precedence and
incremental forecast information, but they cannot measure signed energy flux.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
SOURCE = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / "analysis_results"
    / "compare_magnetic_field_sweep_three_component_B25_B30mT_E10kVm"
)
OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "SimVPv2"
    / "workdirs"
    / "analyze_radaz_b25_b30_pre_repic_directionality"
)

CASES = ("B25mT", "B30mT")
TARGETS = ("long", "mtsi")
EARLY_END_US = 5.0
SMOOTH_US = 0.15
MAX_LAG_US = 1.20
HISTORY_US = 0.30
HISTORY_TAPS = 5
FORECAST_HORIZONS_US = (0.15, 0.30, 0.60)
SURROGATES = 1999
RNG_SEED = 20260825


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1:
        return values.copy()
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def load_case(case_name: str) -> dict[str, np.ndarray]:
    rows = read_csv(SOURCE / case_name / "three_component_time_series.csv")
    names = rows[0].keys()
    data = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in names
    }
    mask = data["time_us"] <= EARLY_END_US + 1.0e-9
    return {name: values[mask] for name, values in data.items()}


def persistent_crossing(
    time_us: np.ndarray,
    values: np.ndarray,
    threshold: float,
    above: bool,
    persistence_frames: int,
    start_index: int = 0,
) -> float:
    condition = values >= threshold if above else values <= threshold
    for index in range(start_index, len(values) - persistence_frames + 1):
        if np.all(condition[index : index + persistence_frames]):
            return float(time_us[index])
    return float("nan")


def chronology(case_name: str, data: dict[str, np.ndarray]) -> list[dict]:
    time_us = data["time_us"]
    dt = float(np.median(np.diff(time_us)))
    width = max(1, int(round(SMOOTH_US / dt)))
    persistence = max(1, int(round(0.15 / dt)))
    output = []
    high = moving_average(np.log(np.maximum(data["ecdi_ey_power"], 1.0e-30)), width)
    high_peak = int(np.argmax(high))
    high_late = float(np.median(high[time_us >= 3.5]))
    high_half = high_late + 0.5 * (high[high_peak] - high_late)
    high_fall = persistent_crossing(
        time_us, high, high_half, False, persistence, start_index=high_peak
    )
    output.append(
        {
            "case": case_name,
            "component": "ecdi",
            "event": "peak_then_half_decay",
            "peak_time_us": float(time_us[high_peak]),
            "persistent_half_transition_time_us": high_fall,
            "delay_from_ecdi_peak_us": high_fall - float(time_us[high_peak]),
        }
    )
    for target in TARGETS:
        values = moving_average(
            np.log(np.maximum(data[f"{target}_ey_power"], 1.0e-30)), width
        )
        early = float(np.median(values[time_us <= 0.30]))
        late = float(np.median(values[time_us >= 3.5]))
        threshold = early + 0.5 * (late - early)
        rise = persistent_crossing(
            time_us,
            values,
            threshold,
            late >= early,
            persistence,
            start_index=0,
        )
        output.append(
            {
                "case": case_name,
                "component": target,
                "event": "persistent_half_transition",
                "peak_time_us": float("nan"),
                "persistent_half_transition_time_us": rise,
                "delay_from_ecdi_peak_us": rise - float(time_us[high_peak]),
                "early_log_power": early,
                "late_log_power": late,
            }
        )
    return output


def lagged_correlation(source: np.ndarray, target: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    lags = np.arange(0, max_lag + 1)
    correlation = np.empty(len(lags), dtype=np.float64)
    for index, lag in enumerate(lags):
        left = source[: len(source) - lag] if lag else source
        right = target[lag:] if lag else target
        if len(left) < 10 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
            correlation[index] = np.nan
        else:
            correlation[index] = np.corrcoef(left, right)[0, 1]
    return lags, correlation


def lead_lag_diagnostics(
    case_name: str,
    data: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[list[dict], dict[str, tuple[np.ndarray, np.ndarray]]]:
    time_us = data["time_us"]
    dt = float(np.median(np.diff(time_us)))
    width = max(1, int(round(SMOOTH_US / dt)))
    max_lag = int(round(MAX_LAG_US / dt))
    min_shift = int(round(0.75 / dt))
    high = moving_average(np.log(np.maximum(data["ecdi_ey_power"], 1.0e-30)), width)
    high_loss = -np.gradient(high, dt)
    output = []
    curves = {}
    for target in TARGETS:
        low = moving_average(
            np.log(np.maximum(data[f"{target}_ey_power"], 1.0e-30)), width
        )
        low_growth = np.gradient(low, dt)
        lags, correlation = lagged_correlation(high_loss, low_growth, max_lag)
        best = int(np.nanargmax(correlation))
        observed = float(correlation[best])
        null = np.empty(SURROGATES, dtype=np.float64)
        valid_shifts = np.arange(min_shift, len(high_loss) - min_shift)
        for draw in range(SURROGATES):
            shifted = np.roll(high_loss, int(rng.choice(valid_shifts)))
            _, local = lagged_correlation(shifted, low_growth, max_lag)
            null[draw] = float(np.nanmax(local))
        p_value = float((1 + np.count_nonzero(null >= observed)) / (SURROGATES + 1))
        output.append(
            {
                "case": case_name,
                "source": "ECDI log-power loss rate",
                "target": f"{target} log-power growth rate",
                "best_positive_lag_us": float(lags[best] * dt),
                "best_correlation": observed,
                "circular_shift_max_correlation_p": p_value,
                "null_q95": float(np.quantile(null, 0.95)),
                "surrogates": SURROGATES,
            }
        )
        curves[target] = (lags * dt, correlation)
    return output, curves


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale > 1.0e-10, scale, 1.0)
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    target_mean = float(np.mean(y_train))
    target_scale = max(float(np.std(y_train)), 1.0e-10)
    target = (y_train - target_mean) / target_scale
    design = np.column_stack((np.ones(len(train)), train))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return (np.column_stack((np.ones(len(test)), test)) @ coefficient) * target_scale + target_mean


def lag_matrix(values: np.ndarray, history: int) -> np.ndarray:
    lags = np.unique(np.linspace(0, history - 1, HISTORY_TAPS, dtype=int))
    output = np.full((len(values), len(lags)), np.nan, dtype=np.float64)
    for column, lag in enumerate(lags):
        output[history - 1 :, column] = values[history - 1 - lag : len(values) - lag]
    return output


def forecast_gain(case_name: str, data: dict[str, np.ndarray]) -> list[dict]:
    time_us = data["time_us"]
    dt = float(np.median(np.diff(time_us)))
    width = max(1, int(round(SMOOTH_US / dt)))
    history = max(2, int(round(HISTORY_US / dt)))
    high = moving_average(np.log(np.maximum(data["ecdi_ey_power"], 1.0e-30)), width)
    high_loss = -np.gradient(high, dt)
    high_history = lag_matrix(high_loss, history)
    output = []
    for target in TARGETS:
        low = moving_average(
            np.log(np.maximum(data[f"{target}_ey_power"], 1.0e-30)), width
        )
        low_growth = np.gradient(low, dt)
        low_history = lag_matrix(low_growth, history)
        for horizon_us in FORECAST_HORIZONS_US:
            horizon = int(round(horizon_us / dt))
            valid = np.arange(history - 1, len(time_us) - horizon)
            y_forward = low[valid + horizon] - low[valid]
            y_reverse = high[valid] - high[valid + horizon]
            direction_specs = (
                ("ECDI_to_target", low_history[valid], high_history[valid], y_forward),
                ("target_to_ECDI", high_history[valid], low_history[valid], y_reverse),
            )
            for direction, baseline, extra, target_values in direction_specs:
                full = np.concatenate((baseline, extra), axis=1)
                truth = []
                prediction_baseline = []
                prediction_full = []
                # Expanding-window blocks retain chronology. The first test block contains
                # the 1.2--2.0 us B30 reorganization emphasized in the prior analysis.
                for train_end_us, test_end_us in ((1.20, 2.00), (2.00, 3.00), (3.00, 4.00), (4.00, 5.00)):
                    train = time_us[valid] < train_end_us - 1.0e-9
                    test = (time_us[valid] >= train_end_us - 1.0e-9) & (
                        time_us[valid] < test_end_us - 1.0e-9
                    )
                    if np.count_nonzero(train) < full.shape[1] + 5 or np.count_nonzero(test) < 5:
                        continue
                    truth.append(target_values[test])
                    prediction_baseline.append(
                        ridge_fit_predict(baseline[train], target_values[train], baseline[test])
                    )
                    prediction_full.append(
                        ridge_fit_predict(full[train], target_values[train], full[test])
                    )
                if not truth:
                    continue
                truth_array = np.concatenate(truth)
                baseline_array = np.concatenate(prediction_baseline)
                full_array = np.concatenate(prediction_full)
                mse_baseline = float(np.mean((truth_array - baseline_array) ** 2))
                mse_full = float(np.mean((truth_array - full_array) ** 2))
                output.append(
                    {
                        "case": case_name,
                        "target_component": target,
                        "direction": direction,
                        "history_us": HISTORY_US,
                        "forecast_horizon_us": horizon_us,
                        "test_samples": len(truth_array),
                        "baseline_mse": mse_baseline,
                        "augmented_mse": mse_full,
                        "incremental_forecast_skill": 1.0
                        - mse_full / max(mse_baseline, 1.0e-30),
                    }
                )
    return output


def plot_power(cases: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(13.0, 8.4), constrained_layout=True, sharex=True)
    colors = {"long": "#0072b2", "mtsi": "#009e73", "ecdi": "#d55e00"}
    for axis, case_name in zip(axes, CASES):
        data = cases[case_name]
        dt = float(np.median(np.diff(data["time_us"])))
        width = max(1, int(round(SMOOTH_US / dt)))
        for component in ("long", "mtsi", "ecdi"):
            power = moving_average(
                np.log10(np.maximum(data[f"{component}_ey_power"], 1.0e-30)), width
            )
            axis.plot(data["time_us"], power, linewidth=2.0, color=colors[component], label=component)
        axis.set_title(case_name)
        axis.set_ylabel("smoothed log10 Ey power")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
    axes[-1].set_xlabel("time [us]")
    figure.suptitle("Early spectral reorganization with B25 as control")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_lead_lag(curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True, sharey=True)
    colors = {"long": "#0072b2", "mtsi": "#009e73"}
    for axis, case_name in zip(axes, CASES):
        for target in TARGETS:
            lag, correlation = curves[case_name][target]
            axis.plot(lag, correlation, linewidth=2.0, color=colors[target], label=target)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(case_name)
        axis.set_xlabel("target delay after ECDI loss [us]")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
    axes[0].set_ylabel("corr(ECDI loss rate, target growth rate)")
    figure.suptitle("Positive-lag directionality diagnostic")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_forecast_gain(rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True, sharey=True)
    for row_index, case_name in enumerate(CASES):
        for column_index, target in enumerate(TARGETS):
            axis = axes[row_index, column_index]
            for direction, color, marker in (
                ("ECDI_to_target", "#d55e00", "o"),
                ("target_to_ECDI", "#0072b2", "s"),
            ):
                local = [
                    row
                    for row in rows
                    if row["case"] == case_name
                    and row["target_component"] == target
                    and row["direction"] == direction
                ]
                axis.plot(
                    [row["forecast_horizon_us"] for row in local],
                    [row["incremental_forecast_skill"] for row in local],
                    color=color,
                    marker=marker,
                    linewidth=2.0,
                    label=direction,
                )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_title(f"{case_name}: {target}")
            axis.set_xlabel("forecast horizon [us]")
            lower, upper = axis.get_ylim()
            axis.set_ylim(min(-12.5, lower), upper)
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right", fontsize=7)
    axes[0, 0].set_ylabel("incremental forecast skill")
    axes[1, 0].set_ylabel("incremental forecast skill")
    figure.suptitle("Does the other band add chronological forecast information?")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    cases = {case_name: load_case(case_name) for case_name in CASES}
    chronology_rows = []
    lead_lag_rows = []
    lead_lag_curves = {}
    forecast_rows = []
    for case_name, data in cases.items():
        chronology_rows.extend(chronology(case_name, data))
        local_rows, local_curves = lead_lag_diagnostics(case_name, data, rng)
        lead_lag_rows.extend(local_rows)
        lead_lag_curves[case_name] = local_curves
        forecast_rows.extend(forecast_gain(case_name, data))
    write_csv(OUTPUT / "transition_chronology.csv", chronology_rows)
    write_csv(OUTPUT / "lead_lag_shift_null.csv", lead_lag_rows)
    write_csv(OUTPUT / "directional_forecast_gain.csv", forecast_rows)
    plot_power(cases, OUTPUT / "early_high_to_low_power_chronology.png")
    plot_lead_lag(lead_lag_curves, OUTPUT / "high_loss_to_low_growth_lead_lag.png")
    plot_forecast_gain(forecast_rows, OUTPUT / "directional_forecast_gain.png")

    elementary_charge = 1.602176634e-19
    electron_mass = 9.1093837015e-31
    ly = 1.28e-2
    ez = 1.0e4
    theoretical_mode = {
        case_name: elementary_charge * (int(case_name[1:3]) * 1.0e-3) ** 2 * ly
        / (2.0 * np.pi * electron_mass * ez)
        for case_name in CASES
    }
    summary = {
        "status": "PASS",
        "question": "Does the B30 high-n transient precede and predict lower-n growth more strongly than B25?",
        "theoretical_ECDI_mode_n0": theoretical_mode,
        "protocol": {
            "interval_us": [0.0, EARLY_END_US],
            "smoothing_us": SMOOTH_US,
            "positive_lag_search_us": [0.0, MAX_LAG_US],
            "history_us": HISTORY_US,
            "history_taps": HISTORY_TAPS,
            "forecast_horizons_us": FORECAST_HORIZONS_US,
            "shift_surrogates": SURROGATES,
            "source_component_masks": {
                "B25mT": {"long": "n=1..4", "MTSI": "n=5..13", "ECDI": "n=14..34"},
                "B30mT": {"long": "n=1..3", "MTSI": "n=4..18", "ECDI": "n=20..48"},
            },
        },
        "chronology": chronology_rows,
        "lead_lag": lead_lag_rows,
        "directional_forecast_gain": forecast_rows,
        "guardrail": (
            "Temporal precedence and incremental forecast information are not signed energy transfer. "
            "An inverse-cascade claim still requires a mode-resolved energy budget or spectral flux."
        ),
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    readme = [
        "# B25/B30 pre-rePIC directionality diagnostic",
        "",
        "This analysis uses the existing three-component mode time series over 0--5 us.",
        "B25 is processed identically as a control for the B30 high-n to low-n reorganization.",
        "",
        "Evidence levels:",
        "",
        "- transition chronology: descriptive temporal order;",
        "- lead-lag with circular-shift null: exploratory directional association;",
        "- expanding-window ridge gain: incremental chronological forecast information;",
        "- none of these is signed spectral energy flux.",
        "",
        "See `analysis_summary.json` and the CSV files for numerical results.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[DONE] {OUTPUT}")


if __name__ == "__main__":
    main()
