"""Test whether the E25-to-E30 1.2 us adaptation depends on one phase window.

Two leakage-safe diagnostics are run with the frozen E25 L+Pcirc+T Hankel ROM:

1. Keep the E30 forecast fixed at 20--30 us and move the 1.2 us adaptation
   window through five disjoint preforecast intervals.
2. Move both adaptation and forecast origins through E30.  Each 1.2 us
   adaptation window is followed by a four-microsecond autonomous forecast.

All scaler and operator choices for a row use target samples strictly before
that row's forecast boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import adapt_radaz_e25_fixed_rom_to_e30 as adaptation
import analyze_radaz_e25_carrier_state_closure as carrier
import analyze_radaz_hankel_havok as hankel
import evaluate_radaz_e25_fixed_rom_zero_shot_e30 as zero


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "compare_radaz_e25_rom_e30_shifted_1p2us_adaptation"
)
ADAPTATION_US = 1.2
FIXED_FORECAST_START_US = 20.0
FIXED_FORECAST_END_US = 30.0
SHIFTED_ADAPTATION_ENDS_US = (15.2, 16.4, 17.6, 18.8, 20.0)
ROLLING_FORECAST_STARTS_US = (16.0, 18.0, 20.0, 22.0, 24.0, 26.0)
ROLLING_FORECAST_DURATION_US = 4.0


def interval_mask(
    times: np.ndarray, start_us: float, end_us: float, include_end: bool
) -> np.ndarray:
    right = times <= end_us + 1.0e-9 if include_end else times < end_us
    return (times >= start_us) & right


def weighted_phi_trace(
    coefficients: np.ndarray, radial_weights: np.ndarray
) -> np.ndarray:
    weights = np.asarray(radial_weights, dtype=float)
    weights = weights / np.sum(weights)
    if coefficients.ndim == 2:
        return np.sum(coefficients * weights[None, :], axis=1)
    if coefficients.ndim == 3:
        return np.sum(coefficients * weights[None, :, None], axis=1)
    raise ValueError(f"Unsupported phi coefficient shape {coefficients.shape}")


def evaluate_rollout(
    experiment: str,
    window_id: str,
    variant: str,
    target: zero.CaseRepresentation,
    scaler,
    model: hankel.HankelModel,
    phi_block: carrier.CarrierBlock,
    adaptation_start_us: float | None,
    adaptation_end_us: float | None,
    forecast_start_us: float,
    forecast_end_us: float,
    model_info: dict,
) -> tuple[list[dict], list[dict], dict]:
    states = scaler.transform(target.groups)
    history_indices = np.flatnonzero(target.raw.time_us < forecast_start_us)
    forecast_mask = interval_mask(
        target.raw.time_us,
        forecast_start_us,
        forecast_end_us,
        include_end=True,
    )
    forecast_indices = np.flatnonzero(forecast_mask)
    if len(history_indices) < model.delay:
        raise ValueError(f"Insufficient history for forecast at {forecast_start_us}")
    if len(forecast_indices) < 3:
        raise ValueError(f"Insufficient forecast frames at {forecast_start_us}")

    history = states[history_indices[-model.delay :]]
    prediction_state = hankel.rollout_hankel(
        model, history, len(forecast_indices)
    )
    prediction = scaler.inverse(prediction_state)

    transport_truth = target.transport[forecast_mask]
    transport_prediction = prediction["transport_direct"][:, 0]
    transport_persistence = np.repeat(
        target.transport[history_indices[-1]], len(transport_truth)
    )
    transport_history_mean = np.repeat(
        float(np.mean(target.transport[history_indices[-zero.DELAY :]])),
        len(transport_truth),
    )

    forecast_frames = target.raw.frame[forecast_mask]
    phi_prediction = phi_block.decode_circular(
        prediction["phi_circular"], forecast_frames
    )
    phi_truth = target.selected_phi[forecast_mask]
    phi_persistence = np.repeat(
        target.selected_phi[history_indices[-1] : history_indices[-1] + 1],
        len(phi_truth),
        axis=0,
    )
    phi_carrier = zero.target_carrier_baseline(
        phi_block, target.raw, history_indices[-1], forecast_frames
    )
    phi_envelope_truth = carrier.phi_envelope(
        phi_truth, target.raw.radial_weights
    )
    phi_envelope_prediction = carrier.phi_envelope(
        phi_prediction, target.raw.radial_weights
    )
    phi_history = carrier.phi_envelope(
        target.selected_phi[history_indices[-zero.DELAY :]],
        target.raw.radial_weights,
    )
    phi_envelope_persistence = np.repeat(phi_history[-1], len(phi_truth))
    phi_envelope_history_mean = np.repeat(
        float(np.mean(phi_history)), len(phi_truth)
    )

    if adaptation_start_us is None or adaptation_end_us is None:
        adaptation_frames = 0
        adaptation_transport_mean = float("nan")
        adaptation_transport_std = float("nan")
        adaptation_phi_phase_end = float("nan")
        adaptation_dominant_phi_mode_index = -1
        overlap_frames = 0
    else:
        adaptation_mask = interval_mask(
            target.raw.time_us,
            adaptation_start_us,
            adaptation_end_us,
            include_end=False,
        )
        adaptation_frames = int(np.count_nonzero(adaptation_mask))
        adaptation_transport_mean = float(
            np.mean(target.transport[adaptation_mask])
        )
        adaptation_transport_std = float(
            np.std(target.transport[adaptation_mask], ddof=1)
        )
        phi_trace = weighted_phi_trace(
            target.selected_phi[adaptation_mask], target.raw.radial_weights
        )
        if phi_trace.ndim == 1:
            adaptation_dominant_phi_mode_index = 0
            adaptation_phi_phase_end = float(np.angle(phi_trace[-1]))
        else:
            adaptation_dominant_phi_mode_index = int(
                np.argmax(np.sqrt(np.mean(np.abs(phi_trace) ** 2, axis=0)))
            )
            adaptation_phi_phase_end = float(
                np.angle(phi_trace[-1, adaptation_dominant_phi_mode_index])
            )
        overlap_frames = int(
            np.count_nonzero(adaptation_mask & forecast_mask)
        )

    common = {
        "experiment": experiment,
        "window_id": window_id,
        "variant": variant,
        "adaptation_us": 0.0
        if adaptation_start_us is None
        else adaptation_end_us - adaptation_start_us,
        "adaptation_start_us": adaptation_start_us,
        "adaptation_end_us": adaptation_end_us,
        "adaptation_frames": adaptation_frames,
        "adaptation_transport_mean": adaptation_transport_mean,
        "adaptation_transport_std": adaptation_transport_std,
        "adaptation_phi_phase_end_rad": adaptation_phi_phase_end,
        "adaptation_dominant_phi_mode_index": adaptation_dominant_phi_mode_index,
        "forecast_start_us": forecast_start_us,
        "forecast_end_us": forecast_end_us,
        "forecast_frames": int(len(forecast_indices)),
        **model_info,
    }
    metrics = [
        {
            **common,
            "quantity": "selected_modal_transport",
            **zero.scalar_summary(
                transport_truth,
                transport_prediction,
                transport_persistence,
                transport_history_mean,
            ),
        },
        {
            **common,
            "quantity": "phi_envelope",
            **zero.scalar_summary(
                phi_envelope_truth,
                phi_envelope_prediction,
                phi_envelope_persistence,
                phi_envelope_history_mean,
            ),
        },
        {
            **common,
            "quantity": "phi_coefficients",
            **carrier.coefficient_metrics(
                phi_truth,
                phi_prediction,
                phi_persistence,
                phi_carrier,
                target.raw.radial_weights,
            ),
        },
    ]
    traces = []
    for index, time_us in enumerate(target.raw.time_us[forecast_mask]):
        traces.append(
            {
                **common,
                "time_us": float(time_us),
                "transport_truth": float(transport_truth[index]),
                "transport_prediction": float(transport_prediction[index]),
                "transport_persistence": float(transport_persistence[index]),
                "phi_envelope_truth": float(phi_envelope_truth[index]),
                "phi_envelope_prediction": float(
                    phi_envelope_prediction[index]
                ),
            }
        )

    changed_states = states.copy()
    changed_states[target.raw.time_us >= forecast_start_us] = 9876.0
    changed_history = changed_states[
        target.raw.time_us < forecast_start_us
    ][-model.delay :]
    audit = {
        **common,
        "adaptation_forecast_overlap_frames": overlap_frames,
        "history_future_perturbation_max_difference": float(
            np.max(np.abs(history - changed_history))
        ),
        "history_last_time_us": float(
            target.raw.time_us[history_indices[-1]]
        ),
        "prediction_finite_fraction": float(
            np.mean(np.isfinite(prediction_state))
        ),
        "prediction_sha256": zero.sha256_array(prediction_state),
    }
    return metrics, traces, audit


def model_info(model: hankel.HankelModel, selection: dict | None) -> dict:
    if selection is None:
        return {
            "correction_rank": 0,
            "correction_ridge": 0.0,
            "correction_shrinkage": 0.0,
            "validation_skill_vs_source_operator": 0.0,
            "spectral_radius": float(
                np.max(np.abs(model.eigenvalues))
            ),
        }
    return {
        "correction_rank": int(selection["correction_rank"]),
        "correction_ridge": float(selection["ridge"]),
        "correction_shrinkage": float(
            selection["final_shrinkage_after_stability_backoff"]
        ),
        "validation_skill_vs_source_operator": float(
            selection["validation_skill_vs_source_operator"]
        ),
        "spectral_radius": float(selection["final_spectral_radius"]),
    }


def run_adapted_window(
    experiment: str,
    window_id: str,
    source_scaler,
    source_model: hankel.HankelModel,
    target: zero.CaseRepresentation,
    phi_block: carrier.CarrierBlock,
    adaptation_end_us: float,
    forecast_start_us: float,
    forecast_end_us: float,
) -> tuple[list[dict], list[dict], list[dict], dict, list[dict]]:
    adaptation_start_us = adaptation_end_us - ADAPTATION_US
    scaler, calibration = adaptation.calibrated_scaler(
        source_scaler,
        target,
        ADAPTATION_US,
        adaptation_end_us=adaptation_end_us,
    )
    for row in calibration:
        row["experiment"] = experiment
        row["window_id"] = window_id
        row["forecast_start_us"] = forecast_start_us
        row["forecast_end_us"] = forecast_end_us

    metrics: list[dict] = []
    traces: list[dict] = []
    audits: list[dict] = []
    result = evaluate_rollout(
        experiment,
        window_id,
        "pt_affine",
        target,
        scaler,
        source_model,
        phi_block,
        adaptation_start_us,
        adaptation_end_us,
        forecast_start_us,
        forecast_end_us,
        model_info(source_model, None),
    )
    metrics.extend(result[0])
    traces.extend(result[1])
    audits.append(result[2])

    target_states = scaler.transform(target.groups)
    corrected, selection, candidates = adaptation.corrected_model(
        source_model,
        target_states,
        target.raw.time_us,
        ADAPTATION_US,
        adaptation_end_us=adaptation_end_us,
    )
    selection.update(
        {
            "experiment": experiment,
            "window_id": window_id,
            "forecast_start_us": forecast_start_us,
            "forecast_end_us": forecast_end_us,
        }
    )
    for row in candidates:
        row["experiment"] = experiment
        row["window_id"] = window_id
        row["forecast_start_us"] = forecast_start_us
        row["forecast_end_us"] = forecast_end_us
    result = evaluate_rollout(
        experiment,
        window_id,
        "pt_affine_lowrank",
        target,
        scaler,
        corrected,
        phi_block,
        adaptation_start_us,
        adaptation_end_us,
        forecast_start_us,
        forecast_end_us,
        model_info(corrected, selection),
    )
    metrics.extend(result[0])
    traces.extend(result[1])
    audits.append(result[2])
    return metrics, traces, audits, selection, candidates + calibration


def selected_metric(
    rows: list[dict],
    experiment: str,
    window_id: str,
    variant: str,
    quantity: str,
) -> dict:
    matches = [
        row
        for row in rows
        if row["experiment"] == experiment
        and row["window_id"] == window_id
        and row["variant"] == variant
        and row["quantity"] == quantity
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for {experiment}/{window_id}/{variant}/{quantity}"
        )
    return matches[0]


def plot_fixed_sensitivity(path: Path, rows: list[dict]) -> None:
    fixed = [
        row
        for row in rows
        if row["experiment"] == "fixed_forecast"
        and row["quantity"] in {
            "selected_modal_transport",
            "phi_coefficients",
        }
    ]
    figure, axes = plt.subplots(
        2, 2, figsize=(14.5, 8.5), constrained_layout=True
    )
    panels = (
        ("selected_modal_transport", "correlation", "transport correlation"),
        (
            "selected_modal_transport",
            "skill_vs_persistence",
            "transport skill vs persistence",
        ),
        ("phi_coefficients", "coefficient_correlation", "phi coefficient correlation"),
        ("phi_coefficients", "weighted_phase_mae_rad", "phi phase MAE [rad]"),
    )
    variants = (
        ("pt_affine", "Pcirc+T affine", "#e69f00", "o"),
        (
            "pt_affine_lowrank",
            "affine + low-rank operator",
            "#0072b2",
            "s",
        ),
    )
    for axis, (quantity, key, ylabel) in zip(axes.ravel(), panels):
        for variant, label, color, marker in variants:
            chosen = sorted(
                [
                    row
                    for row in fixed
                    if row["variant"] == variant
                    and row["quantity"] == quantity
                ],
                key=lambda row: float(row["adaptation_end_us"]),
            )
            axis.plot(
                [row["adaptation_end_us"] for row in chosen],
                [row[key] for row in chosen],
                color=color,
                marker=marker,
                linewidth=1.7,
                label=label,
            )
        axis.set_xlabel("adaptation-window end [us]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 1].axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        fontsize=9,
    )
    figure.suptitle(
        "E30 fixed 20--30 us forecast: sensitivity to the 1.2 us adaptation window"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_rolling_summary(path: Path, rows: list[dict]) -> None:
    rolling = [row for row in rows if row["experiment"] == "rolling_origin"]
    figure, axes = plt.subplots(
        2, 2, figsize=(14.5, 8.5), constrained_layout=True
    )
    panels = (
        ("selected_modal_transport", "correlation", "transport correlation"),
        (
            "selected_modal_transport",
            "skill_vs_persistence",
            "transport skill vs persistence",
        ),
        ("phi_coefficients", "coefficient_correlation", "phi coefficient correlation"),
        ("phi_coefficients", "weighted_phase_mae_rad", "phi phase MAE [rad]"),
    )
    variants = (
        ("zero_shot", "strict zero-shot", "#666666", "^"),
        ("pt_affine", "Pcirc+T affine", "#e69f00", "o"),
        (
            "pt_affine_lowrank",
            "affine + low-rank operator",
            "#0072b2",
            "s",
        ),
    )
    for axis, (quantity, key, ylabel) in zip(axes.ravel(), panels):
        for variant, label, color, marker in variants:
            chosen = sorted(
                [
                    row
                    for row in rolling
                    if row["variant"] == variant
                    and row["quantity"] == quantity
                ],
                key=lambda row: float(row["forecast_start_us"]),
            )
            axis.plot(
                [row["forecast_start_us"] for row in chosen],
                [row[key] for row in chosen],
                color=color,
                marker=marker,
                linewidth=1.5,
                label=label,
            )
        axis.set_xlabel("forecast start [us]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 1].axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    zero_skills = [
        row["skill_vs_persistence"]
        for row in rolling
        if row["variant"] == "zero_shot"
        and row["quantity"] == "selected_modal_transport"
    ]
    axes[0, 1].set_ylim(-0.2, 0.65)
    axes[0, 1].text(
        0.02,
        0.04,
        f"zero-shot = {min(zero_skills):.1f} to {max(zero_skills):.1f} (below axis)",
        transform=axes[0, 1].transAxes,
        fontsize=8,
        color="#555555",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        fontsize=9,
    )
    figure.suptitle(
        "E30 rolling-origin: 1.2 us adaptation followed by a 4 us forecast"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_rolling_rollouts(path: Path, traces: list[dict]) -> None:
    figure, axes = plt.subplots(
        3, 2, figsize=(14.0, 10.0), constrained_layout=True
    )
    for axis, start_us in zip(axes.ravel(), ROLLING_FORECAST_STARTS_US):
        window_id = f"forecast_{start_us:.1f}_{start_us + ROLLING_FORECAST_DURATION_US:.1f}"
        truth = [
            row
            for row in traces
            if row["experiment"] == "rolling_origin"
            and row["window_id"] == window_id
            and row["variant"] == "zero_shot"
        ]
        adapted = [
            row
            for row in traces
            if row["experiment"] == "rolling_origin"
            and row["window_id"] == window_id
            and row["variant"] == "pt_affine_lowrank"
        ]
        axis.plot(
            [row["time_us"] for row in truth],
            [row["transport_truth"] for row in truth],
            color="#111111",
            linewidth=1.5,
            label="PIC truth",
        )
        axis.plot(
            [row["time_us"] for row in truth],
            [row["transport_prediction"] for row in truth],
            color="#777777",
            linewidth=1.0,
            label="zero-shot",
        )
        axis.plot(
            [row["time_us"] for row in adapted],
            [row["transport_prediction"] for row in adapted],
            color="#0072b2",
            linewidth=1.2,
            label="1.2 us low-rank",
        )
        axis.set_title(f"forecast {start_us:.0f}--{start_us + 4.0:.0f} us")
        axis.set_xlabel("physical time [us]")
        axis.set_ylabel("selected-mode transport")
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        fontsize=9,
    )
    figure.suptitle("E30 rolling-origin transport rollouts")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def transport_rows(rows: list[dict], experiment: str, variant: str) -> list[dict]:
    return sorted(
        [
            row
            for row in rows
            if row["experiment"] == experiment
            and row["variant"] == variant
            and row["quantity"] == "selected_modal_transport"
        ],
        key=lambda row: (
            float(row["forecast_start_us"]),
            -1.0
            if row["adaptation_end_us"] is None
            else float(row["adaptation_end_us"]),
        ),
    )


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "positive_count": int(np.count_nonzero(array > 0.0)),
        "count": int(len(array)),
    }


def write_readme(path: Path, metrics: list[dict], selections: list[dict]) -> None:
    fixed = transport_rows(metrics, "fixed_forecast", "pt_affine_lowrank")
    rolling = transport_rows(metrics, "rolling_origin", "pt_affine_lowrank")
    rolling_zero = transport_rows(metrics, "rolling_origin", "zero_shot")
    fixed_phi = [
        row
        for row in metrics
        if row["experiment"] == "fixed_forecast"
        and row["variant"] == "pt_affine_lowrank"
        and row["quantity"] == "phi_coefficients"
    ]
    rolling_phi = [
        row
        for row in metrics
        if row["experiment"] == "rolling_origin"
        and row["variant"] == "pt_affine_lowrank"
        and row["quantity"] == "phi_coefficients"
    ]
    fixed_skill = summary([row["skill_vs_persistence"] for row in fixed])
    fixed_corr = summary([row["correlation"] for row in fixed])
    rolling_skill = summary([row["skill_vs_persistence"] for row in rolling])
    rolling_corr = summary([row["correlation"] for row in rolling])
    zero_skill = summary([row["skill_vs_persistence"] for row in rolling_zero])
    fixed_phi_corr = summary(
        [row["coefficient_correlation"] for row in fixed_phi]
    )
    rolling_phi_corr = summary(
        [row["coefficient_correlation"] for row in rolling_phi]
    )
    rolling_phase = summary(
        [row["weighted_phase_mae_rad"] for row in rolling_phi]
    )

    lines = [
        "# E25 ROM to E30: shifted 1.2 us adaptation windows",
        "",
        "固定E25 `L+Pcirc+T / Hankel DMD`に対する1.2 us E30適応が、18.8--20.0 usという一つの履歴区間だけに依存するかを調べた。全てのscaler、低ランク補正、rank/ridge/shrinkage選択は各forecast開始前だけで行った。",
        "",
        "## A. Fixed 20--30 us forecast, shifted adaptation windows",
        "",
        "予測は20--30 usへ固定し、1.2 usの適応窓だけを互いに重ならない5区間へ移動した。",
        "",
        "| adaptation window [us] | rank | val skill | transport corr | persistence skill | history-mean skill | std ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    selection_map = {
        (row["experiment"], row["window_id"]): row for row in selections
    }
    for row in fixed:
        selected = selection_map[(row["experiment"], row["window_id"])]
        lines.append(
            f"| {row['adaptation_start_us']:.1f}--{row['adaptation_end_us']:.1f} | "
            f"{int(row['correction_rank'])} | "
            f"{selected['validation_skill_vs_source_operator']:.3f} | "
            f"{row['correlation']:.3f} | {row['skill_vs_persistence']:.3f} | "
            f"{row['skill_vs_initial_history_mean']:.3f} | "
            f"{row['prediction_std_over_truth_std']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## B. Rolling-origin 4 us forecasts",
            "",
            "各forecast開始直前の1.2 usだけで適応し、その後4 usを自律予測した。",
            "",
            "| forecast [us] | rank | val skill | zero-shot skill | adapted corr | adapted skill | history-mean skill | std ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    zero_map = {row["window_id"]: row for row in rolling_zero}
    for row in rolling:
        selected = selection_map[(row["experiment"], row["window_id"])]
        zero_row = zero_map[row["window_id"]]
        lines.append(
            f"| {row['forecast_start_us']:.0f}--{row['forecast_end_us']:.0f} | "
            f"{int(row['correction_rank'])} | "
            f"{selected['validation_skill_vs_source_operator']:.3f} | "
            f"{zero_row['skill_vs_persistence']:.3f} | "
            f"{row['correlation']:.3f} | {row['skill_vs_persistence']:.3f} | "
            f"{row['skill_vs_initial_history_mean']:.3f} | "
            f"{row['prediction_std_over_truth_std']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate result",
            "",
            f"- Fixed forecast: median correlation `{fixed_corr['median']:.3f}`, median persistence skill `{fixed_skill['median']:.3f}`, positive-skill windows `{fixed_skill['positive_count']}/{fixed_skill['count']}`.",
            f"- Rolling origin: median correlation `{rolling_corr['median']:.3f}`, median persistence skill `{rolling_skill['median']:.3f}`, positive-skill origins `{rolling_skill['positive_count']}/{rolling_skill['count']}`.",
            f"- Rolling zero-shot: median persistence skill `{zero_skill['median']:.3f}`.",
            f"- Phi dynamics remain poor: median coefficient correlation is `{fixed_phi_corr['median']:.3f}` for the fixed forecast and `{rolling_phi_corr['median']:.3f}` for rolling origins; rolling median phase MAE is `{rolling_phase['median']:.3f} rad`.",
            "",
            "This is a robustness diagnostic within the already inspected E30 trajectory, not an independent confirmatory test. A true confirmation still requires a locked protocol on an unused seed or an untouched continuation.",
            "",
            "## Files",
            "",
            "- `window_metrics.csv`: transport, phi-envelope, and phi-coefficient metrics.",
            "- `window_time_series.csv`: all rollout traces.",
            "- `operator_selection.csv` and `operator_candidates.csv`: preforecast-only model selection.",
            "- `calibration_diagnostics.csv`: affine calibration diagnostics.",
            "- `protocol_and_audit.json`: provenance and leakage audit.",
            "- `fixed_forecast_window_sensitivity.png`: fixed-forecast sensitivity.",
            "- `rolling_origin_summary.png`: rolling-origin metrics.",
            "- `rolling_origin_transport_rollouts.png`: rolling transport traces.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source_raw = carrier.load_raw_physical(zero.SOURCE_PHYSICAL)
    target_raw = carrier.load_raw_physical(zero.TARGET_PHYSICAL)
    source_features, source_time, source_frames = zero.block.load_features(
        zero.SOURCE_FEATURES
    )
    target_features, target_time, target_frames = zero.block.load_features(
        zero.TARGET_FEATURES
    )
    if not np.allclose(source_raw.time_us, source_time, atol=1.0e-9):
        raise ValueError("E25 physical/latent times differ")
    if not np.allclose(target_raw.time_us, target_time, atol=1.0e-9):
        raise ValueError("E30 physical/latent times differ")
    if not np.array_equal(source_raw.frame, source_frames):
        raise ValueError("E25 physical/latent frames differ")
    if not np.array_equal(target_raw.frame, target_frames):
        raise ValueError("E30 physical/latent frames differ")

    (
        latent_models,
        phi_block,
        source_scaler,
        source_model,
        _source,
        source_pca,
    ) = zero.fit_source_model(source_raw, source_features)
    target = zero.transform_target(
        target_raw, target_features, latent_models, phi_block
    )

    metrics: list[dict] = []
    traces: list[dict] = []
    audits: list[dict] = []
    selections: list[dict] = []
    candidates: list[dict] = []
    calibration_rows: list[dict] = []

    for adaptation_end_us in SHIFTED_ADAPTATION_ENDS_US:
        adaptation_start_us = adaptation_end_us - ADAPTATION_US
        window_id = f"adapt_{adaptation_start_us:.1f}_{adaptation_end_us:.1f}"
        result = run_adapted_window(
            "fixed_forecast",
            window_id,
            source_scaler,
            source_model,
            target,
            phi_block,
            adaptation_end_us,
            FIXED_FORECAST_START_US,
            FIXED_FORECAST_END_US,
        )
        metrics.extend(result[0])
        traces.extend(result[1])
        audits.extend(result[2])
        selections.append(result[3])
        for row in result[4]:
            if "group" in row:
                calibration_rows.append(row)
            else:
                candidates.append(row)
        transport = selected_metric(
            metrics,
            "fixed_forecast",
            window_id,
            "pt_affine_lowrank",
            "selected_modal_transport",
        )
        print(
            f"[FIXED {adaptation_start_us:.1f}--{adaptation_end_us:.1f}] "
            f"rank={result[3]['correction_rank']} "
            f"corr={transport['correlation']:.4f} "
            f"skill={transport['skill_vs_persistence']:.4f}",
            flush=True,
        )

    for forecast_start_us in ROLLING_FORECAST_STARTS_US:
        forecast_end_us = forecast_start_us + ROLLING_FORECAST_DURATION_US
        window_id = f"forecast_{forecast_start_us:.1f}_{forecast_end_us:.1f}"
        result = evaluate_rollout(
            "rolling_origin",
            window_id,
            "zero_shot",
            target,
            source_scaler,
            source_model,
            phi_block,
            None,
            None,
            forecast_start_us,
            forecast_end_us,
            model_info(source_model, None),
        )
        metrics.extend(result[0])
        traces.extend(result[1])
        audits.append(result[2])

        result = run_adapted_window(
            "rolling_origin",
            window_id,
            source_scaler,
            source_model,
            target,
            phi_block,
            forecast_start_us,
            forecast_start_us,
            forecast_end_us,
        )
        metrics.extend(result[0])
        traces.extend(result[1])
        audits.extend(result[2])
        selections.append(result[3])
        for row in result[4]:
            if "group" in row:
                calibration_rows.append(row)
            else:
                candidates.append(row)
        transport = selected_metric(
            metrics,
            "rolling_origin",
            window_id,
            "pt_affine_lowrank",
            "selected_modal_transport",
        )
        print(
            f"[ROLL {forecast_start_us:.1f}--{forecast_end_us:.1f}] "
            f"rank={result[3]['correction_rank']} "
            f"corr={transport['correlation']:.4f} "
            f"skill={transport['skill_vs_persistence']:.4f}",
            flush=True,
        )

    protocol = {
        "source": "E25 fixed L+Pcirc+T Hankel DMD",
        "target": "E30",
        "source_fit_us": [zero.FIT_START_US, zero.FORECAST_START_US],
        "adaptation_duration_us": ADAPTATION_US,
        "fixed_forecast_us": [
            FIXED_FORECAST_START_US,
            FIXED_FORECAST_END_US,
        ],
        "shifted_adaptation_ends_us": list(SHIFTED_ADAPTATION_ENDS_US),
        "rolling_forecast_starts_us": list(ROLLING_FORECAST_STARTS_US),
        "rolling_forecast_duration_us": ROLLING_FORECAST_DURATION_US,
        "fixed_delay": zero.DELAY,
        "fixed_rank": zero.RANK,
        "correction_candidate_ranks": list(adaptation.CORRECTION_RANKS),
        "correction_candidate_ridges": list(adaptation.CORRECTION_RIDGES),
        "correction_candidate_shrinkages": list(
            adaptation.CORRECTION_SHRINKAGES
        ),
        "target_forecast_truth_used_for_adaptation": False,
        "independent_confirmatory_test": False,
        "source_pca": source_pca,
        "source_operator_sha256": zero.sha256_array(source_model.matrix),
        "selections": selections,
        "audits": audits,
    }

    adaptation.write_csv(args.output / "window_metrics.csv", metrics)
    adaptation.write_csv(args.output / "window_time_series.csv", traces)
    adaptation.write_csv(args.output / "operator_selection.csv", selections)
    adaptation.write_csv(args.output / "operator_candidates.csv", candidates)
    adaptation.write_csv(
        args.output / "calibration_diagnostics.csv", calibration_rows
    )
    (args.output / "protocol_and_audit.json").write_text(
        json.dumps(adaptation.json_safe(protocol), indent=2),
        encoding="utf-8",
    )
    plot_fixed_sensitivity(
        args.output / "fixed_forecast_window_sensitivity.png", metrics
    )
    plot_rolling_summary(
        args.output / "rolling_origin_summary.png", metrics
    )
    plot_rolling_rollouts(
        args.output / "rolling_origin_transport_rollouts.png", traces
    )
    write_readme(args.output / "README.md", metrics, selections)
    print(f"[DONE] {args.output}", flush=True)


if __name__ == "__main__":
    main()
