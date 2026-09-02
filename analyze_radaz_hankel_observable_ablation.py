from __future__ import annotations

import argparse
import csv
import json
import math
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
    / "analyze_radaz_bx20mt_hankel_observable_ablation"
)
DELAYS = (2, 5, 10, 20, 40)
RANKS = (4, 8, 12, 16, 20, 30, 40, 60, 80, 100, 120, 160)
METHODS = ("ordinary_dmd", "selected_hankel", "selected_overall")


def fit_hankel_rank_family(
    states: np.ndarray,
    delay: int,
    ranks: tuple[int, ...],
) -> dict[int, base.HankelModel]:
    delay_vectors = base.make_delay_vectors(states, delay)
    delay_mean = np.mean(delay_vectors, axis=0)
    centered = delay_vectors - delay_mean
    _, singular_values, right = np.linalg.svd(
        centered, full_matrices=False
    )
    available = min(
        len(delay_vectors) - 1,
        int(np.count_nonzero(singular_values > 1.0e-10)),
    )
    models = {}
    for requested_rank in ranks:
        effective_rank = min(requested_rank, available)
        if effective_rank < 1:
            continue
        basis = right[:effective_rank].T
        coordinates = centered @ basis
        x = coordinates[:-1].T
        y = coordinates[1:].T
        matrix = y @ np.linalg.pinv(x, rcond=1.0e-10)
        models[requested_rank] = base.HankelModel(
            delay=delay,
            rank=effective_rank,
            state_dimensions=states.shape[1],
            delay_mean=delay_mean,
            basis=basis,
            matrix=matrix,
            eigenvalues=np.linalg.eigvals(matrix),
        )
    return models


def select_hankel(
    observable: observable_analysis.ObservableState,
    ordinary_selection: dict,
) -> tuple[dict, dict, list[dict]]:
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

    ordinary = dict(ordinary_selection)
    ordinary["method"] = "ordinary_dmd"
    ordinary["delay"] = 1
    best_hankel = {"objective": float("inf")}
    trials = []
    for delay in DELAYS:
        ranks = tuple(
            rank
            for rank in RANKS
            if rank <= min(
                observable.state_dimensions,
                len(subtrain) - delay,
            )
        )
        models = fit_hankel_rank_family(subtrain, delay, ranks)
        for requested_rank, model in models.items():
            prediction = base.rollout_hankel(
                model, subtrain, len(validation)
            )
            mse = base.prediction_mse(validation, prediction)
            radius = float(np.max(np.abs(model.eigenvalues)))
            objective = mse + max(0.0, radius - 1.02) * 10.0
            row = {
                "electric_field_kvm": case.electric_field_kvm,
                "variant": observable.variant,
                "method": "hankel_dmd",
                "state_dimensions": observable.state_dimensions,
                "delay": delay,
                "requested_rank": requested_rank,
                "effective_rank": model.rank,
                "validation_mse": mse,
                "spectral_radius": radius,
                "objective": objective,
            }
            trials.append(row)
            if objective < best_hankel["objective"]:
                best_hankel = row

    best_overall = min(
        (ordinary, best_hankel),
        key=lambda row: float(row["objective"]),
    )
    return best_hankel, best_overall, trials


def build_hankel_predictions(
    observable: observable_analysis.ObservableState,
    ordinary_predictions: dict[str, np.ndarray],
    selection: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    case = observable.case
    fit_mask = base.contiguous_mask(
        case.time_us, base.FIT_START_US, base.FIT_END_US
    )
    fit = observable.states[fit_mask]
    model = base.fit_hankel_model(
        fit,
        int(selection["delay"]),
        int(selection["requested_rank"]),
    )
    prediction = base.rollout_hankel(
        model,
        fit,
        len(ordinary_predictions["truth"]),
    )
    output = dict(ordinary_predictions)
    output["prediction"] = prediction
    return (
        output,
        {
            "delay": model.delay,
            "rank": model.rank,
            "spectral_radius": float(
                np.max(np.abs(model.eigenvalues))
            ),
        },
    )


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
        "ordinary_dmd": "#777777",
        "selected_hankel": "#2455a4",
    }
    labels = {
        "ordinary_dmd": "ordinary DMD",
        "selected_hankel": "validation-selected Hankel",
    }
    fig, axes = plt.subplots(
        2, 3, figsize=(14.5, 8.5), constrained_layout=True
    )
    for ax, (metric, title) in zip(axes.ravel(), specifications):
        x = np.arange(len(base.ELECTRIC_FIELDS))
        for method in ("ordinary_dmd", "selected_hankel"):
            rows = [
                next(
                    row
                    for row in metric_rows
                    if row["electric_field_kvm"]
                    == electric_field_kvm
                    and row["variant"]
                    == "fourier_cross_phase_transport"
                    and row["method"] == method
                )
                for electric_field_kvm in base.ELECTRIC_FIELDS
            ]
            ax.plot(
                x,
                [row[metric] for row in rows],
                marker="o",
                linewidth=1.8,
                color=colors[method],
                label=labels[method],
            )
        ax.axhline(0.0, color="#999999", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(value) for value in base.ELECTRIC_FIELDS]
        )
        ax.set_xlabel("Ez [kV/m]")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Hankel memory ablation for cross-phase + transport state"
    )
    fig.savefig(
        outdir / "hankel_observable_ablation_metrics.png", dpi=180
    )
    plt.close(fig)


def plot_selected_delays(
    selected_rows: list[dict],
    outdir: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(11.0, 6.5), constrained_layout=True
    )
    width = 0.23
    x = np.arange(len(base.ELECTRIC_FIELDS))
    colors = ("#777777", "#2455a4", "#c43c39")
    for index, variant in enumerate(observable_analysis.VARIANTS):
        rows = [
            next(
                row
                for row in selected_rows
                if row["electric_field_kvm"] == electric_field_kvm
                and row["variant"] == variant
            )
            for electric_field_kvm in base.ELECTRIC_FIELDS
        ]
        ax.bar(
            x + (index - 1) * width,
            [row["selected_hankel_delay"] for row in rows],
            width=width,
            color=colors[index],
            label=observable_analysis.VARIANT_LABELS[variant],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(value) for value in base.ELECTRIC_FIELDS]
    )
    ax.set_xlabel("Ez [kV/m]")
    ax.set_ylabel("Validation-selected delay [frames]")
    ax.set_title("Selected memory length (15 ns per frame)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(outdir / "hankel_selected_delays.png", dpi=180)
    plt.close(fig)


def plot_transport(
    series: dict[tuple[int, str, str], dict],
    outdir: Path,
) -> None:
    variant = "fourier_cross_phase_transport"
    fig, axes = plt.subplots(
        2, 2, figsize=(13.0, 8.5), constrained_layout=True
    )
    for ax, electric_field_kvm in zip(
        axes.ravel(), base.ELECTRIC_FIELDS
    ):
        ordinary = series[
            (electric_field_kvm, variant, "ordinary_dmd")
        ]
        hankel = series[
            (electric_field_kvm, variant, "selected_hankel")
        ]
        ax.plot(
            ordinary["time_us"],
            ordinary["transport_truth"],
            color="#111111",
            linewidth=2.2,
            label="PIC truth",
        )
        ax.plot(
            ordinary["time_us"],
            ordinary["transport_prediction"],
            color="#777777",
            linewidth=1.4,
            label="ordinary DMD",
        )
        ax.plot(
            hankel["time_us"],
            hankel["transport_prediction"],
            color="#2455a4",
            linewidth=1.4,
            label="selected Hankel",
        )
        ax.set_title(f"Ez={electric_field_kvm} kV/m")
        ax.set_xlabel("Time [us]")
        ax.set_ylabel("Selected-mode transport")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Memory effect on direct modal-transport prediction"
    )
    fig.savefig(
        outdir / "hankel_transport_autonomous_rollouts.png",
        dpi=180,
    )
    plt.close(fig)


def save_h5(
    path: Path,
    observables: dict[
        tuple[int, str], observable_analysis.ObservableState
    ],
    predictions: dict[tuple[int, str, str], dict[str, np.ndarray]],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["radial_bands"] = observable_analysis.RADIAL_BANDS
        for (electric_field_kvm, variant), observable in observables.items():
            group = handle.create_group(
                f"E{electric_field_kvm}/{variant}"
            )
            ordinary = predictions[
                (electric_field_kvm, variant, "ordinary_dmd")
            ]
            indices = np.searchsorted(
                observable.case.frame,
                ordinary["holdout_frames"],
            )
            group.create_dataset(
                "time_us", data=observable.case.time_us[indices]
            )
            group.create_dataset("truth", data=ordinary["truth"])
            for method in ("ordinary_dmd", "selected_hankel"):
                group.create_dataset(
                    method,
                    data=predictions[
                        (electric_field_kvm, variant, method)
                    ]["prediction"],
                    compression="gzip",
                    compression_opts=1,
                )


def generate_readme(
    outdir: Path,
    metric_rows: list[dict],
    selected_rows: list[dict],
) -> None:
    lines = []
    variant = "fourier_cross_phase_transport"
    for electric_field_kvm in base.ELECTRIC_FIELDS:
        ordinary = next(
            row
            for row in metric_rows
            if row["electric_field_kvm"] == electric_field_kvm
            and row["variant"] == variant
            and row["method"] == "ordinary_dmd"
        )
        hankel = next(
            row
            for row in metric_rows
            if row["electric_field_kvm"] == electric_field_kvm
            and row["variant"] == variant
            and row["method"] == "selected_hankel"
        )
        selected = next(
            row
            for row in selected_rows
            if row["electric_field_kvm"] == electric_field_kvm
            and row["variant"] == variant
        )
        lines.append(
            "| "
            f"{electric_field_kvm} | "
            f"{selected['selected_hankel_delay']} | "
            f"{ordinary['fourier_state_correlation']:.3f} | "
            f"{hankel['fourier_state_correlation']:.3f} | "
            f"{ordinary['transport_correlation']:.3f} | "
            f"{hankel['transport_correlation']:.3f} |"
        )
    text = f"""# Hankel memory ablation for physical observables

## 日本語

4 radial bandsのFourier/cross-phase/modal-transport状態に過去
2/5/10/20/40フレームを含むHankel DMDを適用しました。delayとrankは
23--24 us validationだけで選び、24--30 usを自律予測しています。

下表は最も完全な`fourier_cross_phase_transport`状態です。

| Ez [kV/m] | selected delay | Fourier corr: DMD | Fourier corr: Hankel | transport corr: DMD | transport corr: Hankel |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Hankel改善があれば、現在状態だけでは不足していたmemory/non-Markov性が
重要だったと解釈できます。ただしdelay増加は実効状態次元も増やすため、
validation改善だけでなくstrict holdout、輸送、物理consistencyを同時に
確認してください。

## Files

- `hankel_observable_metrics.csv`
- `hankel_model_selection.csv`
- `hankel_selected_models.csv`
- `hankel_observable_ablation_metrics.png`
- `hankel_selected_delays.png`
- `hankel_transport_autonomous_rollouts.png`
- `hankel_observable_rollouts.h5`
- `hankel_observable_summary.json`
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
            "Test whether delay coordinates close radial-band Fourier, "
            "cross-phase, and modal-transport dynamics."
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
    all_series = {}
    final_hankel = {}

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
        print(
            f"E{electric_field_kvm}: extracted 4-band state",
            flush=True,
        )

        for variant in observable_analysis.VARIANTS:
            key = (electric_field_kvm, variant)
            observable = observable_analysis.build_observable_state(
                case, variant
            )
            ordinary_selection, _ = (
                observable_analysis.select_rank(observable)
            )
            (
                ordinary_predictions,
                ordinary_final,
            ) = observable_analysis.build_predictions(
                observable, ordinary_selection
            )
            (
                hankel_selection,
                overall_selection,
                hankel_trials,
            ) = select_hankel(observable, ordinary_selection)
            (
                hankel_predictions,
                hankel_final,
            ) = build_hankel_predictions(
                observable,
                ordinary_predictions,
                hankel_selection,
            )
            overall_method = (
                "ordinary_dmd"
                if overall_selection["method"] == "ordinary_dmd"
                else "selected_hankel"
            )
            overall_predictions = {
                "ordinary_dmd": ordinary_predictions,
                "selected_hankel": hankel_predictions,
            }[overall_method]

            observables[key] = observable
            predictions[
                (electric_field_kvm, variant, "ordinary_dmd")
            ] = ordinary_predictions
            predictions[
                (electric_field_kvm, variant, "selected_hankel")
            ] = hankel_predictions
            predictions[
                (electric_field_kvm, variant, "selected_overall")
            ] = overall_predictions
            final_hankel[key] = hankel_final
            trials.extend(hankel_trials)

            for method, current in (
                ("ordinary_dmd", ordinary_predictions),
                ("selected_hankel", hankel_predictions),
                ("selected_overall", overall_predictions),
            ):
                current_metrics, current_series = (
                    observable_analysis.evaluate_prediction(
                        observable, current
                    )
                )
                current_metrics["method"] = method
                metrics.append(current_metrics)
                all_series[
                    (electric_field_kvm, variant, method)
                ] = current_series

            selected_rows.append(
                {
                    "electric_field_kvm": electric_field_kvm,
                    "variant": variant,
                    "state_dimensions": observable.state_dimensions,
                    "ordinary_rank": ordinary_selection[
                        "requested_rank"
                    ],
                    "ordinary_validation_mse": ordinary_selection[
                        "validation_mse"
                    ],
                    "selected_hankel_delay": hankel_selection["delay"],
                    "selected_hankel_rank": hankel_selection[
                        "requested_rank"
                    ],
                    "hankel_validation_mse": hankel_selection[
                        "validation_mse"
                    ],
                    "hankel_final_spectral_radius": hankel_final[
                        "spectral_radius"
                    ],
                    "selected_overall_method": overall_method,
                }
            )
            print(
                f"E{electric_field_kvm} {variant}: "
                f"delay={hankel_selection['delay']} "
                f"rank={hankel_selection['requested_rank']} "
                f"overall={overall_method}",
                flush=True,
            )

    write_csv(outdir / "hankel_observable_metrics.csv", metrics)
    write_csv(outdir / "hankel_model_selection.csv", trials)
    write_csv(outdir / "hankel_selected_models.csv", selected_rows)
    plot_metrics(metrics, outdir)
    plot_selected_delays(selected_rows, outdir)
    plot_transport(all_series, outdir)
    save_h5(
        outdir / "hankel_observable_rollouts.h5",
        observables,
        predictions,
    )
    summary = {
        "status": "PASS",
        "definition": {
            "electric_fields_kvm": base.ELECTRIC_FIELDS,
            "radial_bands": observable_analysis.RADIAL_BANDS,
            "variants": observable_analysis.VARIANTS,
            "delay_candidates": DELAYS,
            "rank_candidates": RANKS,
            "frame_dt_us": base.FRAME_DT_US,
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
        "final_hankel_models": {
            f"E{electric_field_kvm}_{variant}": final_hankel[
                (electric_field_kvm, variant)
            ]
            for electric_field_kvm in base.ELECTRIC_FIELDS
            for variant in observable_analysis.VARIANTS
        },
        "metrics": metrics,
    }
    with (outdir / "hankel_observable_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            json_safe(summary),
            handle,
            indent=2,
            ensure_ascii=False,
        )
    generate_readme(outdir, metrics, selected_rows)
    print(f"PASS: wrote Hankel ablation to {outdir}")


if __name__ == "__main__":
    main()
