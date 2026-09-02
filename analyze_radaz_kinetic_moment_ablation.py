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

import analyze_radaz_cross_phase_transport_ablation as observable_analysis
import analyze_radaz_physical_carrier_envelope as base
import analyze_radaz_radial_band_fourier_ablation as radial


DEFAULT_OUTDIR = (
    base.SIMVP_ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_kinetic_moment_ablation"
)
ELECTRON_CHARGE_C = -1.602176634e-19
VARIANTS = (
    "base_observables",
    "plus_temperature",
    "plus_axial_current",
    "plus_temperature_axial_current",
    "plus_temperature_all_current",
)
VARIANT_LABELS = {
    "base_observables": "base",
    "plus_temperature": "+ Te",
    "plus_axial_current": "+ Jz",
    "plus_temperature_axial_current": "+ Te + Jz",
    "plus_temperature_all_current": "+ Te + Jx/Jy/Jz",
}


@dataclass
class KineticObservable:
    base_observable: observable_analysis.ObservableState
    variant: str
    states: np.ndarray
    extra_slice: slice
    temperature_slice: slice | None
    current_slice: slice | None

    @property
    def case(self) -> radial.BandCaseData:
        return self.base_observable.case

    @property
    def state_dimensions(self) -> int:
        return self.states.shape[1]

    @property
    def base_dimensions(self) -> int:
        return self.base_observable.state_dimensions

    def base_state(self, states: np.ndarray) -> np.ndarray:
        return states[:, : self.base_dimensions]

    def fourier_state(self, states: np.ndarray) -> np.ndarray:
        return self.base_observable.fourier_state(
            self.base_state(states)
        )

    def fourier_complex(self, states: np.ndarray) -> np.ndarray:
        return self.base_observable.fourier_complex(
            self.base_state(states)
        )

    def predicted_cross_phase(self, states: np.ndarray) -> np.ndarray:
        return self.base_observable.predicted_cross_phase(
            self.base_state(states)
        )

    def predicted_transport(self, states: np.ndarray) -> np.ndarray:
        return self.base_observable.predicted_transport(
            self.base_state(states)
        )


def extract_kinetic_signals(
    source_h5: Path,
    time_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(source_h5, "r") as handle:
        all_time_us = (
            np.asarray(handle["axes/time_s"], dtype=np.float64) * 1.0e6
        )
        x_m = np.asarray(handle["axes/x_m"], dtype=np.float64)
        y_m = np.asarray(handle["axes/y_m"], dtype=np.float64)
        keep = (
            (all_time_us >= base.FIT_START_US - 1.0e-10)
            & (all_time_us <= base.HOLDOUT_END_US + 1.0e-10)
        )
        indices = np.flatnonzero(keep)
        if (
            len(indices) != len(time_us)
            or not np.allclose(all_time_us[indices], time_us)
        ):
            raise ValueError("Kinetic and physical time axes differ")
        start = int(indices[0])
        stop = int(indices[-1]) + 1
        ny = len(y_m) - 1
        _, masks, _ = radial.radial_masks(
            x_m, observable_analysis.RADIAL_BANDS
        )
        _, complete_masks, _ = radial.radial_masks(x_m, 1)
        radial_start = int(complete_masks[0][0])
        radial_stop = int(complete_masks[0][-1]) + 1

        temperature = np.empty(
            (
                len(time_us),
                observable_analysis.RADIAL_BANDS,
                ny,
            ),
            dtype=np.float64,
        )
        current = np.empty(
            (
                len(time_us),
                observable_analysis.RADIAL_BANDS,
                ny,
                3,
            ),
            dtype=np.float64,
        )
        chunk_size = 64
        velocity_fields = (
            "electron_ud",
            "electron_vd",
            "electron_wd",
        )
        for chunk_start in range(start, stop, chunk_size):
            chunk_stop = min(stop, chunk_start + chunk_size)
            local = slice(chunk_start - start, chunk_stop - start)
            density = np.asarray(
                handle["fields/electron_den"][
                    chunk_start:chunk_stop,
                    radial_start:radial_stop,
                    :ny,
                ],
                dtype=np.float64,
            )
            temp = np.asarray(
                handle["fields/electron_Temp"][
                    chunk_start:chunk_stop,
                    radial_start:radial_stop,
                    :ny,
                ],
                dtype=np.float64,
            )
            velocities = [
                np.asarray(
                    handle[f"fields/{field}"][
                        chunk_start:chunk_stop,
                        radial_start:radial_stop,
                        :ny,
                    ],
                    dtype=np.float64,
                )
                for field in velocity_fields
            ]
            for band_index, mask in enumerate(masks):
                first = int(mask[0]) - radial_start
                last = int(mask[-1]) - radial_start + 1
                temperature[local, band_index] = np.mean(
                    temp[:, first:last, :], axis=1
                )
                for component, velocity in enumerate(velocities):
                    local_current = (
                        ELECTRON_CHARGE_C
                        * density[:, first:last, :]
                        * velocity[:, first:last, :]
                    )
                    current[local, band_index, :, component] = (
                        np.mean(local_current, axis=1)
                    )
    return temperature, current


def normalized_selected_coefficients(
    signals: np.ndarray,
    modes: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.fft.rfft(signals, axis=2, norm="forward")
    selected = coefficients[:, :, modes, ...]
    scales = np.sqrt(
        np.mean(np.abs(selected[fit_mask]) ** 2, axis=0)
    )
    scales = np.maximum(scales, np.finfo(float).tiny)
    return selected / scales[None, ...], scales


def complex_to_real(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    return np.concatenate((flat.real, flat.imag), axis=1)


def build_kinetic_observable(
    case: radial.BandCaseData,
    temperature_complex: np.ndarray,
    current_complex: np.ndarray,
    variant: str,
) -> KineticObservable:
    base_observable = observable_analysis.build_observable_state(
        case, "fourier_cross_phase_transport"
    )
    blocks = [base_observable.states]
    offset = base_observable.state_dimensions
    temperature_slice = None
    current_slice = None

    if variant in (
        "plus_temperature",
        "plus_temperature_axial_current",
        "plus_temperature_all_current",
    ):
        values = complex_to_real(temperature_complex)
        temperature_slice = slice(offset, offset + values.shape[1])
        blocks.append(values)
        offset += values.shape[1]

    if variant in (
        "plus_axial_current",
        "plus_temperature_axial_current",
    ):
        values = complex_to_real(current_complex[..., 2:3])
        current_slice = slice(offset, offset + values.shape[1])
        blocks.append(values)
        offset += values.shape[1]
    elif variant == "plus_temperature_all_current":
        values = complex_to_real(current_complex)
        current_slice = slice(offset, offset + values.shape[1])
        blocks.append(values)
        offset += values.shape[1]

    return KineticObservable(
        base_observable=base_observable,
        variant=variant,
        states=np.concatenate(blocks, axis=1),
        extra_slice=slice(base_observable.state_dimensions, offset),
        temperature_slice=temperature_slice,
        current_slice=current_slice,
    )


def build_predictions(
    observable: KineticObservable,
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
    model = base.fit_linear_model(
        fit, int(selection["requested_rank"])
    )
    prediction = base.rollout_linear(model, fit[-1], len(truth))

    holdout_frames = case.frame[holdout_mask]
    fit_envelope = case.flatten(case.envelope_complex)[fit_mask]
    repeated_envelope = np.repeat(
        fit_envelope[-1:], len(truth), axis=0
    )
    carrier_fourier = case.remodulate(
        repeated_envelope, holdout_frames
    )
    carrier_base = observable.base_observable.state_from_fourier(
        carrier_fourier
    )
    extra_persistence = np.repeat(
        fit[-1:, observable.extra_slice], len(truth), axis=0
    )
    carrier = np.concatenate(
        (carrier_base, extra_persistence), axis=1
    )
    persistence = np.repeat(fit[-1:], len(truth), axis=0)
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


def evaluate(
    observable: KineticObservable,
    predictions: dict[str, np.ndarray],
) -> dict:
    metrics, _ = observable_analysis.evaluate_prediction(
        observable, predictions
    )
    truth = predictions["truth"]
    prediction = predictions["prediction"]
    extra_truth = truth[:, observable.extra_slice]
    extra_prediction = prediction[:, observable.extra_slice]
    metrics["kinetic_variant"] = observable.variant
    metrics["extra_state_correlation"] = (
        base.safe_correlation(extra_truth, extra_prediction)
        if extra_truth.shape[1] > 0
        else float("nan")
    )
    if observable.temperature_slice is not None:
        metrics["temperature_state_correlation"] = (
            base.safe_correlation(
                truth[:, observable.temperature_slice],
                prediction[:, observable.temperature_slice],
            )
        )
    else:
        metrics["temperature_state_correlation"] = float("nan")
    if observable.current_slice is not None:
        metrics["current_state_correlation"] = base.safe_correlation(
            truth[:, observable.current_slice],
            prediction[:, observable.current_slice],
        )
    else:
        metrics["current_state_correlation"] = float("nan")
    return metrics


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


def plot_metrics(metrics: list[dict], outdir: Path) -> None:
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
        2, 3, figsize=(15.0, 8.7), constrained_layout=True
    )
    for ax, (metric, title) in zip(axes.ravel(), specifications):
        for electric_field_kvm in base.ELECTRIC_FIELDS:
            rows = [
                next(
                    row
                    for row in metrics
                    if row["electric_field_kvm"] == electric_field_kvm
                    and row["kinetic_variant"] == variant
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
            ("base", "+ Te", "+ Jz", "+ Te/Jz", "+ Te/Jxyz"),
            rotation=15,
            ha="right",
        )
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Kinetic-moment observable ablation, autonomous 24-30 us"
    )
    fig.savefig(
        outdir / "kinetic_moment_ablation_metrics.png", dpi=180
    )
    plt.close(fig)


def save_h5(
    path: Path,
    observables: dict[tuple[int, str], KineticObservable],
    predictions: dict[tuple[int, str], dict[str, np.ndarray]],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["radial_bands"] = observable_analysis.RADIAL_BANDS
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
            for name in ("truth", "prediction", "constant_carrier"):
                group.create_dataset(
                    name,
                    data=current[name],
                    compression="gzip",
                    compression_opts=1,
                )


def generate_readme(outdir: Path, metrics: list[dict]) -> None:
    lines = []
    for electric_field_kvm in base.ELECTRIC_FIELDS:
        values = {
            row["kinetic_variant"]: row
            for row in metrics
            if row["electric_field_kvm"] == electric_field_kvm
        }
        base_row = values["base_observables"]
        best = max(
            values.values(),
            key=lambda row: row["transport_correlation"],
        )
        lines.append(
            "| "
            f"{electric_field_kvm} | "
            f"{base_row['transport_correlation']:.3f} | "
            f"{best['kinetic_variant']} | "
            f"{best['transport_correlation']:.3f} | "
            f"{base_row['fourier_state_correlation']:.3f} | "
            f"{best['fourier_state_correlation']:.3f} |"
        )
    text = f"""# Electron kinetic-moment observable ablation

## 日本語

4 radial bandsのFourier/cross-phase/modal-transport状態へ、電子温度Teと
電子電流密度を追加したDMD ablationです。電流は局所場で
`Je = q_e * n_e * u_e`を計算してからradial帯域平均と方位角FFTを行いました。
`Jz`は軸方向電子輸送を直接表す候補です。

| Ez [kV/m] | base transport corr | best kinetic variant | best transport corr | base Fourier corr | best Fourier corr |
|---:|---:|---|---:|---:|---:|
{chr(10).join(lines)}

同じ20--23 us subtrain、23--24 us validation、20--24 us final fit、
24--30 us strict holdoutを使っています。Hankel履歴は使っていません。

## Files

- `kinetic_moment_metrics.csv`
- `kinetic_moment_model_selection.csv`
- `kinetic_moment_selected_models.csv`
- `kinetic_moment_ablation_metrics.png`
- `kinetic_moment_rollouts.h5`
- `kinetic_moment_summary.json`
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
            "Test electron temperature and current-density moments as "
            "missing state variables for physical Fourier dynamics."
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
    metrics = []
    trials = []
    selected_rows = []
    final_models = {}

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
        ) = radial.extract_band_signals(
            source_h5, (observable_analysis.RADIAL_BANDS,)
        )
        case = radial.build_band_case(
            electric_field_kvm=electric_field_kvm,
            source_h5=source_h5,
            time_us=time_us,
            frame=frame,
            signals=signals_by_count[
                observable_analysis.RADIAL_BANDS
            ],
            radial_edges_m=edges_by_count[
                observable_analysis.RADIAL_BANDS
            ],
            radial_weights=weights_by_count[
                observable_analysis.RADIAL_BANDS
            ],
            modes=global_case.modes.copy(),
            mode_roles=list(global_case.mode_roles),
        )
        temperature, current = extract_kinetic_signals(
            source_h5, time_us
        )
        fit_mask = base.contiguous_mask(
            time_us, base.FIT_START_US, base.FIT_END_US
        )
        temperature_complex, _ = normalized_selected_coefficients(
            temperature[..., None],
            case.modes,
            fit_mask,
        )
        current_complex, _ = normalized_selected_coefficients(
            current,
            case.modes,
            fit_mask,
        )
        print(
            f"E{electric_field_kvm}: extracted Te and Je moments",
            flush=True,
        )

        for variant in VARIANTS:
            key = (electric_field_kvm, variant)
            observable = build_kinetic_observable(
                case,
                temperature_complex,
                current_complex,
                variant,
            )
            selection, variant_trials = (
                observable_analysis.select_rank(observable)
            )
            current_predictions, final_model = build_predictions(
                observable, selection
            )
            current_metrics = evaluate(
                observable, current_predictions
            )
            observables[key] = observable
            predictions[key] = current_predictions
            metrics.append(current_metrics)
            trials.extend(variant_trials)
            final_models[key] = final_model
            selected_rows.append(
                {
                    "electric_field_kvm": electric_field_kvm,
                    "variant": variant,
                    "state_dimensions": observable.state_dimensions,
                    "selected_rank": selection["requested_rank"],
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

    write_csv(outdir / "kinetic_moment_metrics.csv", metrics)
    write_csv(outdir / "kinetic_moment_model_selection.csv", trials)
    write_csv(
        outdir / "kinetic_moment_selected_models.csv", selected_rows
    )
    plot_metrics(metrics, outdir)
    save_h5(
        outdir / "kinetic_moment_rollouts.h5",
        observables,
        predictions,
    )
    summary = {
        "status": "PASS",
        "definition": {
            "electric_fields_kvm": base.ELECTRIC_FIELDS,
            "radial_bands": observable_analysis.RADIAL_BANDS,
            "variants": VARIANTS,
            "temperature_field": "electron_Temp [eV]",
            "velocity_fields": [
                "electron_ud [radial m/s]",
                "electron_vd [azimuthal m/s]",
                "electron_wd [axial m/s]",
            ],
            "current_definition": "Je = q_e * electron_den * ue",
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
        "selected_models": selected_rows,
        "metrics": metrics,
    }
    with (outdir / "kinetic_moment_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            json_safe(summary),
            handle,
            indent=2,
            ensure_ascii=False,
        )
    generate_readme(outdir, metrics)
    print(f"PASS: wrote kinetic-moment ablation to {outdir}")


if __name__ == "__main__":
    main()
