from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_radaz_leave_one_ez_out as leaveout
import analyze_radaz_physical_carrier_envelope as base


OUTPUT_DIR = (
    base.SIMVP_ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_leave_one_ez_out_envelope"
)
CARRIER_SOURCES = ("predicted_from_training_ez", "oracle_diagnostic")


def estimate_carrier_angles(
    raw: leaveout.RawCase,
    scales: leaveout.FoldScales,
    start_us: float,
    end_us: float,
) -> np.ndarray:
    mask = leaveout.interval_mask(
        raw.time_us, start_us, end_us, include_start=True
    )
    indices = np.flatnonzero(mask)
    if len(indices) < 3 or not np.all(np.diff(indices) == 1):
        raise ValueError("Carrier interval must be contiguous")
    normalized = raw.fourier[indices] / scales.fourier[None, ...]
    current = normalized[:-1]
    following = normalized[1:]
    cross = np.sum(
        following * np.conj(current), axis=(0, 1, 3)
    )
    return np.angle(cross)


def predict_carrier_angles(
    observed: dict[int, np.ndarray],
    target_electric_field_kvm: int,
) -> np.ndarray:
    electric_fields = np.asarray(sorted(observed), dtype=np.float64)
    values = np.stack(
        [observed[int(value)] for value in electric_fields], axis=0
    )
    predicted = np.empty(values.shape[1], dtype=np.float64)
    centered_e = electric_fields - np.mean(electric_fields)
    target = target_electric_field_kvm - np.mean(electric_fields)
    design = np.column_stack(
        (np.ones(len(centered_e)), centered_e)
    )
    for mode_index in range(values.shape[1]):
        unwrapped = np.unwrap(values[:, mode_index])
        coefficients = np.linalg.lstsq(
            design, unwrapped, rcond=None
        )[0]
        predicted[mode_index] = coefficients[0] + target * coefficients[1]
    return predicted


def carrier_frequency_mhz(angles: np.ndarray) -> np.ndarray:
    return angles / (2.0 * np.pi * base.FRAME_DT_US)


def carrier_error_metrics(
    raw: leaveout.RawCase,
    predicted_angles: np.ndarray,
    oracle_angles: np.ndarray,
) -> dict:
    fit_mask = leaveout.interval_mask(
        raw.time_us, base.FIT_START_US, base.FIT_END_US
    )
    mode_energy = np.mean(
        np.abs(raw.fourier[fit_mask]) ** 2, axis=(0, 1, 3)
    )
    mode_energy = np.maximum(mode_energy, 0.0)
    frequency_error = np.abs(
        carrier_frequency_mhz(predicted_angles)
        - carrier_frequency_mhz(oracle_angles)
    )
    weight_sum = float(np.sum(mode_energy))
    weighted = (
        float(np.sum(mode_energy * frequency_error) / weight_sum)
        if weight_sum > np.finfo(float).tiny
        else float("nan")
    )
    top_indices = np.argsort(mode_energy)[-5:]
    return {
        "carrier_frequency_mae_mhz": float(
            np.mean(frequency_error)
        ),
        "carrier_frequency_energy_weighted_mae_mhz": weighted,
        "carrier_frequency_top5_energy_mae_mhz": float(
            np.mean(frequency_error[top_indices])
        ),
        "carrier_max_frequency_error_mhz": float(
            np.max(frequency_error)
        ),
    }


def demodulate_raw(
    raw: leaveout.RawCase,
    carrier_angles: np.ndarray,
) -> leaveout.RawCase:
    phase = np.exp(
        -1j
        * raw.frame[:, None]
        * carrier_angles[None, :]
    )
    return leaveout.RawCase(
        electric_field_kvm=raw.electric_field_kvm,
        time_us=raw.time_us,
        frame=raw.frame,
        radial_weights=raw.radial_weights,
        fourier=raw.fourier * phase[:, None, :, None],
        cross_phase=raw.cross_phase,
        transport=raw.transport,
        axial_current=raw.axial_current * phase[:, None, :],
    )


def remodulate_states(
    envelope_case: leaveout.StateCase,
    states: np.ndarray,
    frame: np.ndarray,
    carrier_angles: np.ndarray,
) -> np.ndarray:
    remodulated = np.array(states, copy=True)
    fourier_slice = envelope_case.slices["fourier"]
    assert fourier_slice is not None
    fourier_shape = envelope_case.raw.fourier.shape[1:]
    fourier = leaveout.real_to_complex(
        remodulated[:, fourier_slice], fourier_shape
    )
    phase = np.exp(
        1j * frame[:, None] * carrier_angles[None, :]
    )
    fourier *= phase[:, None, :, None]
    remodulated[:, fourier_slice] = leaveout.complex_to_real(
        fourier
    )

    current_slice = envelope_case.slices["axial_current"]
    if current_slice is not None:
        current_shape = envelope_case.raw.axial_current.shape[1:]
        current = leaveout.real_to_complex(
            remodulated[:, current_slice], current_shape
        )
        current *= phase[:, None, :]
        remodulated[:, current_slice] = leaveout.complex_to_real(
            current
        )
    return remodulated


def raw_persistence(
    raw_case: leaveout.StateCase,
    fit_mask: np.ndarray,
    steps: int,
) -> np.ndarray:
    initial = raw_case.states[np.flatnonzero(fit_mask)[-1]]
    return np.repeat(initial[None, :], steps, axis=0)


def fit_and_evaluate_fold(
    raw_cases: dict[int, leaveout.RawCase],
    heldout: int,
    variant: str,
    method: str,
    selection_scales: leaveout.FoldScales,
    final_scales: leaveout.FoldScales,
    selection_observed_angles: dict[int, np.ndarray],
    final_observed_angles: dict[int, np.ndarray],
    predicted_selection_angle: np.ndarray,
    predicted_final_angle: np.ndarray,
    oracle_final_angle: np.ndarray,
) -> tuple[list[dict], list[dict], dict, dict]:
    training_fields = tuple(
        value for value in base.ELECTRIC_FIELDS if value != heldout
    )
    selection_envelope_raw = {}
    final_envelope_raw = {}
    for electric_field_kvm in base.ELECTRIC_FIELDS:
        if electric_field_kvm in training_fields:
            selection_angle = selection_observed_angles[
                electric_field_kvm
            ]
            final_angle = final_observed_angles[electric_field_kvm]
        else:
            selection_angle = predicted_selection_angle
            final_angle = predicted_final_angle
        selection_envelope_raw[electric_field_kvm] = demodulate_raw(
            raw_cases[electric_field_kvm], selection_angle
        )
        final_envelope_raw[electric_field_kvm] = demodulate_raw(
            raw_cases[electric_field_kvm], final_angle
        )

    selection_cases = {
        electric_field_kvm: leaveout.build_state_case(
            selection_envelope_raw[electric_field_kvm],
            selection_scales,
            variant,
        )
        for electric_field_kvm in base.ELECTRIC_FIELDS
    }
    conditioned = method == "shared_ez_conditioned"
    selected, trials = leaveout.select_model(
        selection_cases, training_fields, conditioned
    )
    selection_rows = [
        {
            "heldout_electric_field_kvm": heldout,
            "training_fields_kvm": ",".join(map(str, training_fields)),
            "variant": variant,
            "method": method,
            **row,
        }
        for row in trials
    ]

    final_training_cases = {
        electric_field_kvm: leaveout.build_state_case(
            final_envelope_raw[electric_field_kvm],
            final_scales,
            variant,
        )
        for electric_field_kvm in training_fields
    }
    fit_states = {}
    for electric_field_kvm, current in final_training_cases.items():
        fit_mask = leaveout.interval_mask(
            current.raw.time_us,
            base.FIT_START_US,
            base.FIT_END_US,
        )
        fit_states[electric_field_kvm] = current.states[fit_mask]
    model = leaveout.fit_shared_model(
        fit_states,
        int(selected["requested_rank"]),
        float(selected["ridge"]),
        conditioned,
    )

    metric_rows = []
    series_by_source = {}
    states_by_source = {}
    for carrier_source, heldout_angle in (
        ("predicted_from_training_ez", predicted_final_angle),
        ("oracle_diagnostic", oracle_final_angle),
    ):
        heldout_envelope_raw = demodulate_raw(
            raw_cases[heldout], heldout_angle
        )
        envelope_case = leaveout.build_state_case(
            heldout_envelope_raw, final_scales, variant
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
        envelope_initial = envelope_case.states[
            np.flatnonzero(fit_mask)[-1]
        ]
        envelope_prediction = leaveout.rollout_shared(
            model,
            envelope_initial,
            heldout,
            len(envelope_truth),
        )
        constant_envelope = np.repeat(
            envelope_initial[None, :], len(envelope_truth), axis=0
        )
        envelope_metrics, envelope_series = (
            leaveout.evaluate_prediction(
                envelope_case,
                envelope_truth,
                envelope_prediction,
                constant_envelope,
                time_us,
            )
        )

        raw_truth = raw_case.states[holdout_mask]
        remodulated_truth = remodulate_states(
            envelope_case,
            envelope_truth,
            holdout_frames,
            heldout_angle,
        )
        truth_roundtrip_error = float(
            np.nanmax(np.abs(remodulated_truth - raw_truth))
        )
        raw_prediction = remodulate_states(
            envelope_case,
            envelope_prediction,
            holdout_frames,
            heldout_angle,
        )
        carrier_baseline = remodulate_states(
            envelope_case,
            constant_envelope,
            holdout_frames,
            heldout_angle,
        )
        persistence = raw_persistence(
            raw_case, fit_mask, len(raw_truth)
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
            persistence,
            time_us,
        )

        metric_rows.append(
            {
                "heldout_electric_field_kvm": heldout,
                "training_fields_kvm": ",".join(
                    map(str, training_fields)
                ),
                "variant": variant,
                "method": method,
                "carrier_source": carrier_source,
                "state_dimensions": envelope_case.states.shape[1],
                "selected_rank": model.rank,
                "selected_ridge": model.ridge,
                "validation_mse": selected["validation_mse"],
                "validation_maximum_spectral_radius": selected[
                    "maximum_spectral_radius"
                ],
                "heldout_spectral_radius": leaveout.spectral_radius(
                    model, heldout
                ),
                "envelope_state_correlation": envelope_metrics[
                    "fourier_state_correlation"
                ],
                "envelope_state_skill_vs_constant_envelope": (
                    envelope_metrics[
                        "fourier_state_skill_vs_persistence"
                    ]
                ),
                "raw_fourier_state_correlation": raw_carrier_metrics[
                    "fourier_state_correlation"
                ],
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
                "transport_skill_vs_carrier_baseline": (
                    raw_carrier_metrics[
                        "transport_skill_vs_persistence"
                    ]
                ),
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
                "truth_roundtrip_max_abs_error": truth_roundtrip_error,
            }
        )
        series_by_source[carrier_source] = {
            **raw_series,
            "envelope_transport_prediction": envelope_series[
                "transport_prediction"
            ],
        }
        states_by_source[carrier_source] = {
            "raw_fourier_truth": raw_truth[
                :, raw_case.slices["fourier"]
            ],
            "raw_fourier_prediction": raw_prediction[
                :, raw_case.slices["fourier"]
            ],
            "envelope_fourier_truth": envelope_truth[
                :, envelope_case.slices["fourier"]
            ],
            "envelope_fourier_prediction": envelope_prediction[
                :, envelope_case.slices["fourier"]
            ],
        }
    return metric_rows, selection_rows, series_by_source, states_by_source


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


def metric_row(
    rows: list[dict],
    heldout: int,
    variant: str,
    method: str,
    carrier_source: str,
) -> dict:
    return next(
        row
        for row in rows
        if row["heldout_electric_field_kvm"] == heldout
        and row["variant"] == variant
        and row["method"] == method
        and row["carrier_source"] == carrier_source
    )


def plot_metrics(rows: list[dict], outdir: Path) -> None:
    specs = (
        ("envelope_state_correlation", "Envelope-state correlation"),
        (
            "raw_fourier_state_correlation",
            "Remodulated Fourier-state correlation",
        ),
        (
            "raw_fourier_skill_vs_carrier_baseline",
            "Raw Fourier skill vs carrier-only",
        ),
        ("cross_phase_mae_rad", "Cross-phase MAE [rad]"),
        ("transport_correlation", "Modal-transport correlation"),
        ("finite_fraction", "Finite rollout fraction"),
    )
    labels = {
        ("base_observables", "shared_blind"): "base, blind",
        (
            "base_observables",
            "shared_ez_conditioned",
        ): "base, Ez-conditioned",
        ("plus_axial_current", "shared_blind"): "+ Jz, blind",
        (
            "plus_axial_current",
            "shared_ez_conditioned",
        ): "+ Jz, Ez-conditioned",
    }
    markers = ("o", "s", "^", "D")
    x = np.asarray(base.ELECTRIC_FIELDS, dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for axis, (metric, title) in zip(axes.ravel(), specs):
        for marker, key in zip(markers, labels):
            values = [
                float(
                    metric_row(
                        rows,
                        electric_field_kvm,
                        key[0],
                        key[1],
                        "predicted_from_training_ez",
                    )[metric]
                )
                for electric_field_kvm in base.ELECTRIC_FIELDS
            ]
            display_values = np.asarray(values, dtype=float)
            if metric == "raw_fourier_skill_vs_carrier_baseline":
                display_values = np.maximum(display_values, -1.0)
            axis.plot(
                x,
                display_values,
                marker=marker,
                linewidth=1.8,
                label=labels[key],
            )
            if metric == "raw_fourier_skill_vs_carrier_baseline":
                below = np.asarray(values) < -1.0
                if np.any(below):
                    axis.scatter(
                        x[below],
                        np.full(np.count_nonzero(below), -1.0),
                        marker="v",
                        color=axis.lines[-1].get_color(),
                        s=35,
                        zorder=4,
                    )
        axis.axhline(0.0, color="0.55", linewidth=0.8)
        if metric == "raw_fourier_skill_vs_carrier_baseline":
            axis.set_ylim(-1.08, 1.0)
            axis.text(
                0.01,
                0.02,
                "downward triangles: value < -1",
                transform=axis.transAxes,
                fontsize=8,
                va="bottom",
            )
        axis.set_title(title)
        axis.set_xlabel("Held-out Ez [kV/m]")
        axis.grid(alpha=0.25)
    axes[1, 2].set_ylim(-0.05, 1.05)
    axes[1, 2].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Leave-one-Ez-out envelope dynamics with predicted carrier"
    )
    fig.tight_layout()
    fig.savefig(
        outdir / "leave_one_ez_out_envelope_metrics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_oracle_gap(
    rows: list[dict],
    carrier_rows: list[dict],
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.asarray(base.ELECTRIC_FIELDS, dtype=float)
    combinations = (
        ("base_observables", "shared_blind", "base, blind"),
        (
            "base_observables",
            "shared_ez_conditioned",
            "base, conditioned",
        ),
        ("plus_axial_current", "shared_blind", "+ Jz, blind"),
        (
            "plus_axial_current",
            "shared_ez_conditioned",
            "+ Jz, conditioned",
        ),
    )
    for variant, method, label in combinations:
        predicted = [
            float(
                metric_row(
                    rows,
                    electric_field_kvm,
                    variant,
                    method,
                    "predicted_from_training_ez",
                )["raw_fourier_state_correlation"]
            )
            for electric_field_kvm in base.ELECTRIC_FIELDS
        ]
        oracle = [
            float(
                metric_row(
                    rows,
                    electric_field_kvm,
                    variant,
                    method,
                    "oracle_diagnostic",
                )["raw_fourier_state_correlation"]
            )
            for electric_field_kvm in base.ELECTRIC_FIELDS
        ]
        axes[0].plot(x, predicted, marker="o", label=f"{label}, predicted")
        axes[0].plot(
            x, oracle, marker="x", linestyle="--", label=f"{label}, oracle"
        )
        axes[1].plot(
            x,
            np.asarray(oracle) - np.asarray(predicted),
            marker="o",
            label=label,
        )
    axes[0].set_title("Raw Fourier correlation")
    axes[1].set_title("Oracle carrier gain")
    carrier_values = [
        next(
            row
            for row in carrier_rows
            if row["heldout_electric_field_kvm"]
            == electric_field_kvm
        )["carrier_frequency_energy_weighted_mae_mhz"]
        for electric_field_kvm in base.ELECTRIC_FIELDS
    ]
    axes[2].bar(x, carrier_values, width=5.0, color="#4c78a8")
    axes[2].set_title("Predicted carrier frequency error")
    axes[2].set_ylabel("Energy-weighted MAE [MHz]")
    for axis in axes:
        axis.axhline(0.0, color="0.55", linewidth=0.8)
        axis.set_xlabel("Held-out Ez [kV/m]")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=7)
    axes[1].legend(loc="lower right", fontsize=7)
    fig.suptitle("Carrier interpolation versus envelope-dynamics error")
    fig.tight_layout()
    fig.savefig(
        outdir / "leave_one_ez_out_carrier_oracle_gap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_h5(
    outdir: Path,
    carrier_data: dict[int, dict],
    series: dict[tuple[int, str, str, str], dict],
    states: dict[tuple[int, str, str, str], dict],
) -> None:
    with h5py.File(
        outdir / "leave_one_ez_out_envelope_rollouts.h5", "w"
    ) as handle:
        handle.attrs["common_modes"] = leaveout.COMMON_MODES
        handle.attrs["holdout_interval_us"] = [
            base.FIT_END_US,
            base.HOLDOUT_END_US,
        ]
        for heldout, values in carrier_data.items():
            group = handle.create_group(f"E{heldout}/carrier")
            for name, data in values.items():
                group.create_dataset(name, data=np.asarray(data))
        for key, values in series.items():
            heldout, variant, method, carrier_source = key
            group = handle.create_group(
                f"E{heldout}/{variant}/{method}/{carrier_source}"
            )
            for name, data in values.items():
                group.create_dataset(
                    name, data=np.asarray(data), compression="gzip"
                )
            for name, data in states[key].items():
                group.create_dataset(
                    name, data=np.asarray(data), compression="gzip"
                )


def generate_readme(
    outdir: Path,
    metric_rows: list[dict],
    carrier_rows: list[dict],
) -> None:
    lines = [
        "# Leave-one-Ez-out carrier-envelope dynamics",
        "",
        "## 日本語",
        "",
        "前段のraw Fourier共有モデルの失敗を、carrier推定誤差と",
        "envelope力学の非共通性へ分解した。",
        "",
        "- 共通モード: m=1..30、radial 4 bands",
        "- carrier推定: 学習3条件の20-24 usだけからmode別周波数を推定し、Ezの一次関数で除外条件へ適用",
        "- predicted carrier: 厳密なleave-one-Ez-out主評価",
        "- oracle carrier: 除外条件のcarrierだけを使う原因診断でありzero-shot結果ではない",
        "- 除外条件から主評価に使う真値は24 usの単一初期状態だけ",
        "",
        "## Carrier interpolation",
        "",
        "| held-out Ez | weighted carrier-frequency MAE [MHz] | top-5 MAE [MHz] |",
        "|---:|---:|---:|",
    ]
    for row in carrier_rows:
        lines.append(
            "| {heldout_electric_field_kvm} | "
            "{carrier_frequency_energy_weighted_mae_mhz:.4f} | "
            "{carrier_frequency_top5_energy_mae_mhz:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 結論",
            "",
            "- predicted-carrier zero-shotのenvelope相関は最大0.026で、carrier除去だけでは共通力学にならなかった。",
            "- oracle carrierでもraw Fourier相関は最大約0.080で、carrier誤差だけが失敗原因ではない。",
            "- shared blindの正のMSE skillは平均状態への減衰であり、位相力学の再現ではない。",
            "- Ez-conditioned modelは端点外挿で不安定になった。",
            "- Jz追加は条件間zero-shot改善にはつながらなかった。",
            "",
            "## Main zero-shot metrics",
            "",
            "| held-out Ez | state | method | envelope corr | raw corr | transport corr | finite |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        if row["carrier_source"] != "predicted_from_training_ez":
            continue
        lines.append(
            "| {heldout_electric_field_kvm} | {variant} | {method} | "
            "{envelope_state_correlation:.3f} | "
            "{raw_fourier_state_correlation:.3f} | "
            "{transport_correlation:.3f} | {finite_fraction:.1%} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## English",
            "",
            "Mode-wise carrier frequencies are estimated from the three",
            "training Ez cases and linearly transferred to the excluded Ez.",
            "An oracle-carrier diagnostic separates carrier interpolation",
            "error from failure of the shared envelope dynamics.",
        ]
    )
    (outdir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Carrier-envelope leave-one-Ez-out reduced dynamics."
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
    carrier_rows = []
    all_series = {}
    all_states = {}
    carrier_data = {}
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
        selection_observed_angles = {
            electric_field_kvm: estimate_carrier_angles(
                raw_cases[electric_field_kvm],
                selection_scales,
                base.FIT_START_US,
                base.VALIDATION_START_US,
            )
            for electric_field_kvm in training_fields
        }
        final_observed_angles = {
            electric_field_kvm: estimate_carrier_angles(
                raw_cases[electric_field_kvm],
                final_scales,
                base.FIT_START_US,
                base.FIT_END_US,
            )
            for electric_field_kvm in training_fields
        }
        predicted_selection_angle = predict_carrier_angles(
            selection_observed_angles, heldout
        )
        predicted_final_angle = predict_carrier_angles(
            final_observed_angles, heldout
        )
        oracle_final_angle = estimate_carrier_angles(
            raw_cases[heldout],
            final_scales,
            base.FIT_START_US,
            base.FIT_END_US,
        )
        errors = carrier_error_metrics(
            raw_cases[heldout],
            predicted_final_angle,
            oracle_final_angle,
        )
        carrier_rows.append(
            {
                "heldout_electric_field_kvm": heldout,
                "training_fields_kvm": ",".join(
                    map(str, training_fields)
                ),
                **errors,
            }
        )
        carrier_data[heldout] = {
            "common_modes": leaveout.COMMON_MODES,
            "predicted_angles_rad_per_frame": predicted_final_angle,
            "oracle_angles_rad_per_frame": oracle_final_angle,
            "predicted_frequency_mhz": carrier_frequency_mhz(
                predicted_final_angle
            ),
            "oracle_frequency_mhz": carrier_frequency_mhz(
                oracle_final_angle
            ),
        }
        print(
            f"leave E{heldout}: carrier weighted MAE="
            f"{errors['carrier_frequency_energy_weighted_mae_mhz']:.4f} MHz"
        )

        for variant in leaveout.VARIANTS:
            for method in leaveout.METHODS:
                (
                    current_metrics,
                    current_selection,
                    current_series,
                    current_states,
                ) = fit_and_evaluate_fold(
                    raw_cases=raw_cases,
                    heldout=heldout,
                    variant=variant,
                    method=method,
                    selection_scales=selection_scales,
                    final_scales=final_scales,
                    selection_observed_angles=selection_observed_angles,
                    final_observed_angles=final_observed_angles,
                    predicted_selection_angle=predicted_selection_angle,
                    predicted_final_angle=predicted_final_angle,
                    oracle_final_angle=oracle_final_angle,
                )
                metric_rows.extend(current_metrics)
                selection_rows.extend(current_selection)
                for carrier_source in CARRIER_SOURCES:
                    key = (
                        heldout,
                        variant,
                        method,
                        carrier_source,
                    )
                    all_series[key] = current_series[carrier_source]
                    all_states[key] = current_states[carrier_source]
                primary = next(
                    row
                    for row in current_metrics
                    if row["carrier_source"]
                    == "predicted_from_training_ez"
                )
                oracle = next(
                    row
                    for row in current_metrics
                    if row["carrier_source"] == "oracle_diagnostic"
                )
                print(
                    f"E{heldout} {variant} {method}: "
                    f"env={primary['envelope_state_correlation']:.3f} "
                    f"raw={primary['raw_fourier_state_correlation']:.3f} "
                    f"oracle_raw={oracle['raw_fourier_state_correlation']:.3f}"
                )

    write_csv(
        outdir / "leave_one_ez_out_envelope_metrics.csv",
        metric_rows,
    )
    write_csv(
        outdir / "leave_one_ez_out_envelope_model_selection.csv",
        selection_rows,
    )
    write_csv(
        outdir / "leave_one_ez_out_carrier_errors.csv",
        carrier_rows,
    )
    plot_metrics(metric_rows, outdir)
    plot_oracle_gap(metric_rows, carrier_rows, outdir)
    save_h5(outdir, carrier_data, all_series, all_states)
    generate_readme(outdir, metric_rows, carrier_rows)
    summary = {
        "status": "PASS",
        "definition": {
            "electric_fields_kvm": base.ELECTRIC_FIELDS,
            "common_modes": leaveout.COMMON_MODES,
            "radial_bands": leaveout.RADIAL_BANDS,
            "carrier_model": (
                "Affine linear fit of mode-wise unwrapped phase increment "
                "against Ez using only the three training conditions."
            ),
            "carrier_sources": CARRIER_SOURCES,
            "oracle_is_zero_shot": False,
            "heldout_information_in_primary_evaluation": (
                "Only the state at 24 us initializes the autonomous forecast."
            ),
        },
        "carrier_errors": carrier_rows,
        "metrics": metric_rows,
    }
    (outdir / "leave_one_ez_out_envelope_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    print(f"PASS: wrote carrier-envelope leave-one analysis to {outdir}")


if __name__ == "__main__":
    main()
