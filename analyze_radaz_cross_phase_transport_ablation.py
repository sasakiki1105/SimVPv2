from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_radaz_physical_carrier_envelope as base
import analyze_radaz_radial_band_fourier_ablation as radial


DEFAULT_OUTDIR = (
    base.SIMVP_ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_cross_phase_transport_ablation"
)
RADIAL_BANDS = 4
VARIANTS = (
    "fourier_only",
    "fourier_cross_phase",
    "fourier_cross_phase_transport",
)
VARIANT_LABELS = {
    "fourier_only": "Fourier only",
    "fourier_cross_phase": "+ cross-phase",
    "fourier_cross_phase_transport": "+ cross-phase + transport",
}


@dataclass
class ObservableState:
    case: radial.BandCaseData
    variant: str
    states: np.ndarray
    fourier_slice: slice
    cross_phase_real_slice: slice | None
    cross_phase_imag_slice: slice | None
    transport_slice: slice | None
    transport_scales: np.ndarray

    @property
    def state_dimensions(self) -> int:
        return self.states.shape[1]

    @property
    def fourier_dimensions(self) -> int:
        return self.case.state_dimensions

    def fourier_state(self, states: np.ndarray) -> np.ndarray:
        return states[:, self.fourier_slice]

    def fourier_complex(self, states: np.ndarray) -> np.ndarray:
        return self.case.unflatten(self.fourier_state(states))

    def predicted_cross_phase(self, states: np.ndarray) -> np.ndarray:
        if self.cross_phase_real_slice is None:
            return cross_phase_unit(
                self.case.physical_coefficients(
                    self.fourier_complex(states)
                )
            )
        real = states[:, self.cross_phase_real_slice]
        imag = states[:, self.cross_phase_imag_slice]
        shape = (
            len(states),
            self.case.band_count,
            len(self.case.modes),
        )
        values = real.reshape(shape) + 1j * imag.reshape(shape)
        magnitude = np.abs(values)
        return np.divide(
            values,
            magnitude,
            out=np.zeros_like(values),
            where=magnitude > np.finfo(float).tiny,
        )

    def predicted_transport(self, states: np.ndarray) -> np.ndarray:
        if self.transport_slice is None:
            return modal_transport_components(
                self.case.physical_coefficients(
                    self.fourier_complex(states)
                )
            )
        normalized = states[:, self.transport_slice].reshape(
            len(states),
            self.case.band_count,
            len(self.case.modes),
        )
        return normalized * self.transport_scales[None, :, :]

    def state_from_fourier(self, fourier_complex: np.ndarray) -> np.ndarray:
        return assemble_states(
            self.case,
            fourier_complex,
            self.variant,
            self.transport_scales,
        )[0]


def cross_products(physical: np.ndarray) -> np.ndarray:
    electron_index = base.PHYSICAL_FIELDS.index("electron_den")
    efy_index = base.PHYSICAL_FIELDS.index("efy")
    return (
        physical[..., electron_index]
        * np.conj(physical[..., efy_index])
    )


def cross_phase_unit(physical: np.ndarray) -> np.ndarray:
    cross = cross_products(physical)
    magnitude = np.abs(cross)
    return np.divide(
        cross,
        magnitude,
        out=np.zeros_like(cross),
        where=magnitude > np.finfo(float).tiny,
    )


def modal_transport_components(physical: np.ndarray) -> np.ndarray:
    return -2.0 * np.real(cross_products(physical)) / 0.020


def assemble_states(
    case: radial.BandCaseData,
    normalized_fourier: np.ndarray,
    variant: str,
    transport_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, slice | None], np.ndarray]:
    fourier_state = case.flatten(normalized_fourier)
    physical = case.physical_coefficients(normalized_fourier)
    phase = cross_phase_unit(physical)
    transport = modal_transport_components(physical)
    fit_mask = base.contiguous_mask(
        case.time_us, base.FIT_START_US, base.FIT_END_US
    )
    if transport_scales is None:
        transport_scales = np.sqrt(
            np.mean(transport[fit_mask] ** 2, axis=0)
        )
        transport_scales = np.maximum(
            transport_scales, np.finfo(float).tiny
        )

    blocks = [fourier_state]
    offset = fourier_state.shape[1]
    slices: dict[str, slice | None] = {
        "fourier": slice(0, offset),
        "cross_phase_real": None,
        "cross_phase_imag": None,
        "transport": None,
    }
    if variant in (
        "fourier_cross_phase",
        "fourier_cross_phase_transport",
    ):
        phase_flat = phase.reshape(len(phase), -1)
        width = phase_flat.shape[1]
        slices["cross_phase_real"] = slice(offset, offset + width)
        blocks.append(phase_flat.real)
        offset += width
        slices["cross_phase_imag"] = slice(offset, offset + width)
        blocks.append(phase_flat.imag)
        offset += width
    if variant == "fourier_cross_phase_transport":
        transport_flat = (
            transport / transport_scales[None, :, :]
        ).reshape(len(transport), -1)
        width = transport_flat.shape[1]
        slices["transport"] = slice(offset, offset + width)
        blocks.append(transport_flat)

    return np.concatenate(blocks, axis=1), slices, transport_scales


def build_observable_state(
    case: radial.BandCaseData,
    variant: str,
) -> ObservableState:
    states, slices, transport_scales = assemble_states(
        case, case.normalized_complex, variant
    )
    return ObservableState(
        case=case,
        variant=variant,
        states=states,
        fourier_slice=slices["fourier"],
        cross_phase_real_slice=slices["cross_phase_real"],
        cross_phase_imag_slice=slices["cross_phase_imag"],
        transport_slice=slices["transport"],
        transport_scales=transport_scales,
    )


def select_rank(
    observable: ObservableState,
) -> tuple[dict, list[dict]]:
    case = observable.case
    fit_mask = base.contiguous_mask(
        case.time_us, base.FIT_START_US, base.FIT_END_US
    )
    subtrain_mask = fit_mask & (
        case.time_us < base.VALIDATION_START_US
    )
    validation_mask = fit_mask & (
        case.time_us >= base.VALIDATION_START_US
    )
    subtrain = observable.states[subtrain_mask]
    validation = observable.states[validation_mask]
    rank_candidates = tuple(
        rank
        for rank in radial.RANK_CANDIDATES
        if rank <= min(observable.state_dimensions, len(subtrain) - 1)
    )
    models = radial.fit_rank_family(subtrain, rank_candidates)
    trials = []
    best = {"objective": float("inf")}
    for requested_rank, model in models.items():
        prediction = base.rollout_linear(
            model, subtrain[-1], len(validation)
        )
        mse = base.prediction_mse(validation, prediction)
        radius = float(np.max(np.abs(model.eigenvalues)))
        objective = mse + max(0.0, radius - 1.02) * 10.0
        row = {
            "electric_field_kvm": case.electric_field_kvm,
            "variant": observable.variant,
            "state_dimensions": observable.state_dimensions,
            "requested_rank": requested_rank,
            "effective_rank": model.rank,
            "validation_mse": mse,
            "spectral_radius": radius,
            "objective": objective,
        }
        trials.append(row)
        if objective < best["objective"]:
            best = row
    return best, trials


def build_predictions(
    observable: ObservableState,
    selection: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    case = observable.case
    fit_mask = base.contiguous_mask(
        case.time_us, base.FIT_START_US, base.FIT_END_US
    )
    holdout_mask = (
        (case.time_us > base.FIT_END_US + 1.0e-10)
        & (case.time_us <= base.HOLDOUT_END_US + 1.0e-10)
    )
    fit = observable.states[fit_mask]
    truth = observable.states[holdout_mask]
    holdout_frames = case.frame[holdout_mask]
    model = base.fit_linear_model(
        fit, int(selection["requested_rank"])
    )
    prediction = base.rollout_linear(model, fit[-1], len(truth))

    last_fourier = case.flatten(
        case.normalized_complex[fit_mask][-1:]
    )
    persistence_fourier = np.repeat(
        last_fourier, len(truth), axis=0
    )
    last_envelope = case.flatten(
        case.envelope_complex[fit_mask][-1:]
    )
    repeated_envelope = np.repeat(
        last_envelope, len(truth), axis=0
    )
    carrier_complex = case.remodulate(
        repeated_envelope, holdout_frames
    )
    persistence = observable.state_from_fourier(
        case.unflatten(persistence_fourier)
    )
    carrier = observable.state_from_fourier(carrier_complex)
    return (
        {
            "truth": truth,
            "prediction": prediction,
            "persistence": persistence,
            "constant_carrier": carrier,
            "holdout_frames": holdout_frames,
        },
        {
            "rank": model.rank,
            "spectral_radius": float(
                np.max(np.abs(model.eigenvalues))
            ),
        },
    )


def integrate_transport(
    case: radial.BandCaseData,
    modal_transport: np.ndarray,
) -> np.ndarray:
    per_band = np.sum(modal_transport, axis=2)
    return np.einsum(
        "b,tb->t", case.radial_weights, per_band, optimize=True
    )


def weighted_phase_mae(
    truth_phase: np.ndarray,
    prediction_phase: np.ndarray,
    weights: np.ndarray,
) -> float:
    error = np.abs(
        np.angle(prediction_phase * np.conj(truth_phase))
    )
    finite = (
        np.isfinite(error)
        & np.isfinite(weights)
        & (weights > np.finfo(float).tiny)
    )
    denominator = float(np.sum(weights[finite]))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(np.sum(error[finite] * weights[finite]) / denominator)


def evaluate_prediction(
    observable: ObservableState,
    predictions: dict[str, np.ndarray],
) -> tuple[dict, dict]:
    case = observable.case
    truth = predictions["truth"]
    prediction = predictions["prediction"]
    carrier = predictions["constant_carrier"]
    truth_fourier_state = observable.fourier_state(truth)
    prediction_fourier_state = observable.fourier_state(prediction)
    carrier_fourier_state = observable.fourier_state(carrier)
    fourier_mse = base.prediction_mse(
        truth_fourier_state, prediction_fourier_state
    )
    carrier_fourier_mse = base.prediction_mse(
        truth_fourier_state, carrier_fourier_state
    )
    fourier_skill = (
        1.0 - fourier_mse / carrier_fourier_mse
        if np.isfinite(fourier_mse) and carrier_fourier_mse > 0.0
        else float("-inf")
    )

    truth_fourier = observable.fourier_complex(truth)
    prediction_fourier = observable.fourier_complex(prediction)
    truth_physical = case.physical_coefficients(truth_fourier)
    prediction_physical = case.physical_coefficients(
        prediction_fourier
    )
    truth_cross = cross_products(truth_physical)
    phase_weights = np.abs(truth_cross)
    truth_phase = cross_phase_unit(truth_physical)
    prediction_phase = observable.predicted_cross_phase(prediction)
    phase_mae = weighted_phase_mae(
        truth_phase, prediction_phase, phase_weights
    )

    truth_modal_transport = modal_transport_components(truth_physical)
    prediction_modal_transport = observable.predicted_transport(
        prediction
    )
    carrier_transport = integrate_transport(
        case,
        observable.predicted_transport(carrier),
    )
    truth_transport = integrate_transport(
        case, truth_modal_transport
    )
    prediction_transport = integrate_transport(
        case, prediction_modal_transport
    )
    transport_mse = base.prediction_mse(
        truth_transport[:, None], prediction_transport[:, None]
    )
    carrier_transport_mse = base.prediction_mse(
        truth_transport[:, None], carrier_transport[:, None]
    )
    transport_skill = (
        1.0 - transport_mse / carrier_transport_mse
        if np.isfinite(transport_mse) and carrier_transport_mse > 0.0
        else float("-inf")
    )

    derived_prediction_phase = cross_phase_unit(prediction_physical)
    phase_consistency = weighted_phase_mae(
        derived_prediction_phase,
        prediction_phase,
        phase_weights,
    )
    derived_prediction_transport = integrate_transport(
        case, modal_transport_components(prediction_physical)
    )
    transport_consistency = base.safe_correlation(
        derived_prediction_transport, prediction_transport
    )

    truth_collapsed = case.collapse_normalized(truth_fourier)
    prediction_collapsed = case.collapse_normalized(
        prediction_fourier
    )
    phi_index = base.PHYSICAL_FIELDS.index("phi")
    primary_truth = truth_collapsed[:, 0, phi_index]
    primary_prediction = prediction_collapsed[:, 0, phi_index]
    indices = np.searchsorted(
        case.frame, predictions["holdout_frames"]
    )
    time_us = case.time_us[indices]
    truth_frequency = base.estimate_frequency_mhz(
        primary_truth, time_us
    )
    prediction_frequency = base.estimate_frequency_mhz(
        primary_prediction, time_us
    )

    metrics = {
        "electric_field_kvm": case.electric_field_kvm,
        "variant": observable.variant,
        "state_dimensions": observable.state_dimensions,
        "full_state_correlation": base.safe_correlation(
            truth, prediction
        ),
        "fourier_state_correlation": base.safe_correlation(
            truth_fourier_state, prediction_fourier_state
        ),
        "fourier_state_skill_vs_constant_carrier": fourier_skill,
        "cross_phase_mae_rad": phase_mae,
        "transport_correlation": base.safe_correlation(
            truth_transport, prediction_transport
        ),
        "transport_skill_vs_constant_carrier": transport_skill,
        "primary_frequency_absolute_error_mhz": abs(
            prediction_frequency - truth_frequency
        ),
        "cross_phase_consistency_mae_rad": phase_consistency,
        "transport_consistency_correlation": transport_consistency,
        "finite_fraction": float(
            np.mean(np.isfinite(prediction).all(axis=1))
        ),
    }
    series = {
        "time_us": time_us,
        "transport_truth": truth_transport,
        "transport_prediction": prediction_transport,
        "primary_truth_amplitude": np.abs(primary_truth),
        "primary_prediction_amplitude": np.abs(primary_prediction),
    }
    return metrics, series


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


def plot_metrics(metric_rows: list[dict], outdir: Path) -> None:
    specifications = (
        ("fourier_state_correlation", "Fourier-state correlation"),
        (
            "fourier_state_skill_vs_constant_carrier",
            "Fourier-state skill vs carrier",
        ),
        ("cross_phase_mae_rad", "Cross-phase MAE [rad]"),
        ("transport_correlation", "Modal-transport correlation"),
        (
            "transport_skill_vs_constant_carrier",
            "Transport skill vs carrier",
        ),
        (
            "primary_frequency_absolute_error_mhz",
            "Primary frequency error [MHz]",
        ),
    )
    colors = {
        10: "#2455a4",
        20: "#2a9d5b",
        30: "#d18f00",
        40: "#c43c39",
    }
    x = np.arange(len(VARIANTS))
    fig, axes = plt.subplots(
        2, 3, figsize=(14.5, 8.5), constrained_layout=True
    )
    for ax, (metric, title) in zip(axes.ravel(), specifications):
        for electric_field_kvm in base.ELECTRIC_FIELDS:
            rows = [
                next(
                    row
                    for row in metric_rows
                    if row["electric_field_kvm"] == electric_field_kvm
                    and row["variant"] == variant
                )
                for variant in VARIANTS
            ]
            ax.plot(
                x,
                [row[metric] for row in rows],
                marker="o",
                linewidth=1.8,
                color=colors[electric_field_kvm],
                label=f"Ez={electric_field_kvm} kV/m",
            )
        ax.axhline(0.0, color="#999999", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ("Fourier", "+ phase", "+ phase\n+ transport")
        )
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Nonlinear observable ablation, 4 radial bands, 24-30 us"
    )
    fig.savefig(
        outdir / "cross_phase_transport_ablation_metrics.png",
        dpi=180,
    )
    plt.close(fig)


def plot_transport_rollouts(
    all_series: dict[tuple[int, str], dict],
    outdir: Path,
) -> None:
    colors = {
        "fourier_only": "#777777",
        "fourier_cross_phase": "#2455a4",
        "fourier_cross_phase_transport": "#c43c39",
    }
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 8.5),
        constrained_layout=True,
    )
    for ax, electric_field_kvm in zip(
        axes.ravel(), base.ELECTRIC_FIELDS
    ):
        truth = all_series[
            (electric_field_kvm, "fourier_only")
        ]
        ax.plot(
            truth["time_us"],
            truth["transport_truth"],
            color="#111111",
            linewidth=2.2,
            label="PIC truth",
        )
        for variant in VARIANTS:
            series = all_series[(electric_field_kvm, variant)]
            ax.plot(
                series["time_us"],
                series["transport_prediction"],
                color=colors[variant],
                linewidth=1.4,
                label=VARIANT_LABELS[variant],
            )
        ax.set_title(f"Ez={electric_field_kvm} kV/m")
        ax.set_xlabel("Time [us]")
        ax.set_ylabel("Selected-mode transport")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Direct transport observable in autonomous DMD rollout"
    )
    fig.savefig(
        outdir / "cross_phase_transport_autonomous_rollouts.png",
        dpi=180,
    )
    plt.close(fig)


def save_h5(
    path: Path,
    observables: dict[tuple[int, str], ObservableState],
    predictions: dict[tuple[int, str], dict[str, np.ndarray]],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["radial_bands"] = RADIAL_BANDS
        handle.attrs["fit_interval_us"] = [
            base.FIT_START_US,
            base.FIT_END_US,
        ]
        handle.attrs["holdout_interval_us"] = [
            base.FIT_END_US,
            base.HOLDOUT_END_US,
        ]
        for key, observable in observables.items():
            electric_field_kvm, variant = key
            group = handle.create_group(
                f"E{electric_field_kvm}/{variant}"
            )
            current = predictions[key]
            indices = np.searchsorted(
                observable.case.frame,
                current["holdout_frames"],
            )
            group.create_dataset(
                "time_us", data=observable.case.time_us[indices]
            )
            group.create_dataset(
                "modes", data=observable.case.modes
            )
            group.create_dataset(
                "truth",
                data=current["truth"],
                compression="gzip",
                compression_opts=1,
            )
            group.create_dataset(
                "prediction",
                data=current["prediction"],
                compression="gzip",
                compression_opts=1,
            )
            group.create_dataset(
                "constant_carrier",
                data=current["constant_carrier"],
                compression="gzip",
                compression_opts=1,
            )


def generate_readme(outdir: Path, metric_rows: list[dict]) -> None:
    rows = []
    for electric_field_kvm in base.ELECTRIC_FIELDS:
        values = {
            row["variant"]: row
            for row in metric_rows
            if row["electric_field_kvm"] == electric_field_kvm
        }
        base_row = values["fourier_only"]
        full_row = values["fourier_cross_phase_transport"]
        rows.append(
            "| "
            f"{electric_field_kvm} | "
            f"{base_row['fourier_state_correlation']:.3f} | "
            f"{full_row['fourier_state_correlation']:.3f} | "
            f"{base_row['transport_correlation']:.3f} | "
            f"{full_row['transport_correlation']:.3f} | "
            f"{full_row['cross_phase_mae_rad']:.3f} |"
        )
    text = f"""# Cross-phase and modal-transport observable ablation

## 日本語

4つのradial帯域と既存5モードからなるFourier状態に、電子密度とEyの
cross-phase、およびmodal transportを非線形observableとして追加した
DMD/EDMD的ablationです。

時間分割は20--23 us subtrain、23--24 us validation、20--24 us最終同定、
24--30 us厳密自律予測です。Hankel履歴はまだ使っていません。

| Ez [kV/m] | Fourier corr: base | Fourier corr: full | transport corr: base | transport corr: full | full phase MAE |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

`fourier_cross_phase_transport`ではcross-phaseと輸送を独立の状態として
予測します。そのため、Fourier係数から再計算した値とのconsistencyも
CSVへ保存しています。輸送相関が上がってもconsistencyが壊れている場合、
閉じた物理モデルではなく補助変数だけが別軌道を進んでいる可能性があります。

## English

This EDMD-style ablation augments the four-radial-band Fourier state with
nonlinear observables: the complex unit cross-phase between electron density
and Ey, followed by normalized modal transport. Delay coordinates are not
used in this stage.

## Files

- `cross_phase_transport_metrics.csv`
- `cross_phase_transport_model_selection.csv`
- `cross_phase_transport_selected_models.csv`
- `cross_phase_transport_ablation_metrics.png`
- `cross_phase_transport_autonomous_rollouts.png`
- `cross_phase_transport_rollouts.h5`
- `cross_phase_transport_summary.json`
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add cross-phase and modal-transport observables to the "
            "radial-band physical Fourier state."
        )
    )
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_OUTDIR
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    observables = {}
    predictions = {}
    selections = {}
    final_models = {}
    metrics = []
    trials = []
    all_series = {}
    selected_rows = []

    for electric_field_kvm in base.ELECTRIC_FIELDS:
        source_h5 = radial.analysis_fields_path(electric_field_kvm)
        global_case = base.estimate_representation(
            electric_field_kvm,
            base.diagnostic_path(electric_field_kvm),
        )
        (
            time_us,
            frame,
            signals_by_count,
            edges_by_count,
            weights_by_count,
        ) = radial.extract_band_signals(source_h5, (RADIAL_BANDS,))
        case = radial.build_band_case(
            electric_field_kvm=electric_field_kvm,
            source_h5=source_h5,
            time_us=time_us,
            frame=frame,
            signals=signals_by_count[RADIAL_BANDS],
            radial_edges_m=edges_by_count[RADIAL_BANDS],
            radial_weights=weights_by_count[RADIAL_BANDS],
            modes=global_case.modes.copy(),
            mode_roles=list(global_case.mode_roles),
        )
        print(
            f"E{electric_field_kvm}: extracted 4-band state",
            flush=True,
        )
        for variant in VARIANTS:
            key = (electric_field_kvm, variant)
            observable = build_observable_state(case, variant)
            selection, variant_trials = select_rank(observable)
            current_predictions, final_model = build_predictions(
                observable, selection
            )
            current_metrics, current_series = evaluate_prediction(
                observable, current_predictions
            )
            observables[key] = observable
            predictions[key] = current_predictions
            selections[key] = selection
            final_models[key] = final_model
            metrics.append(current_metrics)
            trials.extend(variant_trials)
            all_series[key] = current_series
            selected_rows.append(
                {
                    "electric_field_kvm": electric_field_kvm,
                    "variant": variant,
                    "state_dimensions": observable.state_dimensions,
                    "selected_rank": selection["requested_rank"],
                    "effective_rank": selection["effective_rank"],
                    "validation_mse": selection["validation_mse"],
                    "validation_spectral_radius": selection[
                        "spectral_radius"
                    ],
                    "final_spectral_radius": final_model[
                        "spectral_radius"
                    ],
                }
            )
            print(
                f"E{electric_field_kvm} {variant}: "
                f"dims={observable.state_dimensions} "
                f"rank={selection['requested_rank']}",
                flush=True,
            )

    write_csv(outdir / "cross_phase_transport_metrics.csv", metrics)
    write_csv(
        outdir / "cross_phase_transport_model_selection.csv", trials
    )
    write_csv(
        outdir / "cross_phase_transport_selected_models.csv",
        selected_rows,
    )
    plot_metrics(metrics, outdir)
    plot_transport_rollouts(all_series, outdir)
    save_h5(
        outdir / "cross_phase_transport_rollouts.h5",
        observables,
        predictions,
    )
    summary = {
        "status": "PASS",
        "definition": {
            "electric_fields_kvm": base.ELECTRIC_FIELDS,
            "magnetic_field_t": 0.020,
            "radial_bands": RADIAL_BANDS,
            "physical_fields": base.PHYSICAL_FIELDS,
            "variants": VARIANTS,
            "cross_phase": (
                "unit complex ne_mode * conj(Ey_mode) / magnitude"
            ),
            "modal_transport": (
                "-2 Re(ne_mode * conj(Ey_mode)) / Bx"
            ),
            "fit_interval_us": [
                base.FIT_START_US,
                base.FIT_END_US,
            ],
            "validation_interval_us": [
                base.VALIDATION_START_US,
                base.FIT_END_US,
            ],
            "holdout_interval_us": [
                base.FIT_END_US,
                base.HOLDOUT_END_US,
            ],
        },
        "selected_models": {
            f"E{electric_field_kvm}_{variant}": {
                "selection": selections[(electric_field_kvm, variant)],
                "final": final_models[(electric_field_kvm, variant)],
            }
            for electric_field_kvm in base.ELECTRIC_FIELDS
            for variant in VARIANTS
        },
        "metrics": metrics,
    }
    with (outdir / "cross_phase_transport_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            json_safe(summary),
            handle,
            indent=2,
            ensure_ascii=False,
        )
    generate_readme(outdir, metrics)
    print(f"PASS: wrote observable ablation to {outdir}")


if __name__ == "__main__":
    main()
