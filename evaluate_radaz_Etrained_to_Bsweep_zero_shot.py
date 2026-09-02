#!/usr/bin/env python3
"""Evaluate E-sweep-trained SimVP models on the magnetic sweep without retraining."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate_radaz_regime_generalization as base


ROOT = Path(__file__).resolve().parent
RESULTS = (
    ROOT.parent
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
)
WORKDIR = ROOT / "workdirs" / "2D_RadAz"
OUTPUT = WORKDIR / "compare_radaz_Etrained_to_Bsweep_zero_shot_direct10"
B_VALUES = (10.0, 15.0, 20.0, 25.0, 30.0)
E_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
TARGET_EZ_VPM = 10.0e3
AZIMUTHAL_LENGTH_M = 1.28e-2
MTSI_MAX_N_OVER_N0 = 0.60
ECDI_MIN_N_OVER_N0 = 0.75
ECDI_MAX_N_OVER_N0 = 1.25
MODEL_FAMILIES = {
    "data_only_100ep": base.DATA_ONLY_MODEL_SPECS,
    "spectral_full_50ep": base.SPECTRAL_MODEL_SPECS,
}
MODEL_LABELS = {
    "low_E10_E20": "Low-E MTSI-side model (E10+E20)",
    "high_E30_E40": "High-E ECDI-side model (E30+E40)",
}
SAME_REGIME_TARGET = {"low_E10_E20": 30.0, "high_E30_E40": 10.0}
OPPOSITE_CONTROL_TARGET = {"low_E10_E20": 10.0, "high_E30_E40": 30.0}
SELECTED = (
    ("field_mse", "phi", "phi MSE"),
    ("field_mse", "ey", "Ey MSE"),
    ("phi_band_amplitude_mae", "MTSI", "MTSI phi amplitude"),
    ("phi_band_amplitude_mae", "ECDI", "ECDI phi amplitude"),
    ("modal_transport_mae", "MTSI", "MTSI transport"),
    ("modal_transport_mae", "ECDI", "ECDI transport"),
    ("cross_phase_weighted_mae_rad", "MTSI", "MTSI cross-phase"),
    ("cross_phase_weighted_mae_rad", "ECDI", "ECDI cross-phase"),
)


def token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def case_name(b_mT: float) -> str:
    return f"2D_RadAz_Xe1p_Bx{token(b_mT)}mT_Ez10kVm_dt15ps_out15ns"


def case_h5(b_mT: float) -> Path:
    name = case_name(b_mT)
    return RESULTS / name / name / "analysis_fields_uncompressed.h5"


def ecdi_mode_number(b_mT: float) -> float:
    magnetic_field_t = float(b_mT) * 1.0e-3
    return (
        E_CHARGE_C
        * magnetic_field_t**2
        * AZIMUTHAL_LENGTH_M
        / (2.0 * np.pi * ELECTRON_MASS_KG * TARGET_EZ_VPM)
    )


def target_mode_bands(b_mT: float) -> dict[str, np.ndarray]:
    maximum_mode = base.VALID_W // 2
    n0 = ecdi_mode_number(b_mT)
    mtsi_upper = min(
        maximum_mode,
        max(1, int(math.floor(MTSI_MAX_N_OVER_N0 * n0))),
    )
    ecdi_lower = max(1, int(math.ceil(ECDI_MIN_N_OVER_N0 * n0)))
    ecdi_upper = min(
        maximum_mode,
        int(math.floor(ECDI_MAX_N_OVER_N0 * n0)),
    )
    if ecdi_lower > ecdi_upper:
        ecdi_lower = ecdi_upper = min(maximum_mode, max(1, int(round(n0))))
    return {
        "MTSI": np.arange(1, mtsi_upper + 1, dtype=int),
        "ECDI": np.arange(ecdi_lower, ecdi_upper + 1, dtype=int),
    }


def set_target_physics(b_mT: float) -> dict[str, np.ndarray]:
    bands = target_mode_bands(b_mT)
    base.B_T = b_mT * 1.0e-3
    base.BANDS = bands
    return bands


def read_b_test_segment(
    b_mT: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = case_h5(b_mT)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        frame_count = int(handle["axes/time_s"].shape[0])
        if frame_count < 2001:
            raise ValueError(f"{path} has only {frame_count} frames")
        times = np.asarray(handle["axes/time_s"][1800:2001], dtype=np.float64)
        x_m = np.asarray(handle["axes/x_m"][: base.VALID_H], dtype=np.float64)
        y_m = np.asarray(handle["axes/y_m"][: base.VALID_W], dtype=np.float64)
        data = np.empty(
            (201, len(base.CHANNELS), base.VALID_H, base.VALID_W),
            dtype=np.float32,
        )
        for channel_index, channel in enumerate(base.CHANNELS):
            data[:, channel_index] = np.asarray(
                handle[f"fields/{channel}"][
                    1800:2001, : base.VALID_H, : base.VALID_W
                ],
                dtype=np.float32,
            )
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite test data in {path}")
    return data, times, x_m, y_m


def role_for(model_key: str, b_mT: float) -> str:
    if b_mT == SAME_REGIME_TARGET[model_key]:
        if b_mT == 30.0:
            return "same_regime_MTSI_longwave_final"
        return "same_regime_ECDI_final"
    if b_mT == OPPOSITE_CONTROL_TARGET[model_key]:
        return "opposite_regime_control"
    return "magnetic_sweep_intermediate"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relabel_rows(rows: list[dict], family: str) -> list[dict]:
    output = []
    for row in rows:
        row = dict(row)
        row["model_family"] = family
        row["target_Bx_mT"] = row.pop("target_Ez_kVm")
        output.append(row)
    return output


def lookup(
    rows: list[dict],
    family: str,
    model: str,
    b_mT: float,
    variant: str,
    metric: str,
    component: str,
) -> dict:
    matches = [
        row
        for row in rows
        if row["model_family"] == family
        and row["model_key"] == model
        and float(row["target_Bx_mT"]) == b_mT
        and row["variant"] == variant
        and row["metric"] == metric
        and row["component"] == component
    ]
    if len(matches) != 1:
        raise ValueError((family, model, b_mT, variant, metric, component, len(matches)))
    return matches[0]


def plot_metric_grid(
    rows: list[dict],
    metric: str,
    components: tuple[str, str],
    title: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, layout="constrained")
    colors = {"data_only_100ep": "#64748b", "spectral_full_50ep": "#dc2626"}
    labels = {"data_only_100ep": "data-only", "spectral_full_50ep": "spectral"}
    for row_index, model in enumerate(MODEL_LABELS):
        for column, component in enumerate(components):
            axis = axes[row_index, column]
            for family in MODEL_FAMILIES:
                values = [
                    float(
                        lookup(
                            rows,
                            family,
                            model,
                            b_mT,
                            "strict_source_normalization",
                            metric,
                            component,
                        )["model_over_copy"]
                    )
                    for b_mT in B_VALUES
                ]
                axis.plot(
                    B_VALUES,
                    values,
                    marker="o",
                    linewidth=2,
                    color=colors[family],
                    label=labels[family],
                )
            axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
            axis.axvline(
                SAME_REGIME_TARGET[model], color="#16a34a", linestyle="--", linewidth=1
            )
            axis.set_yscale("log")
            axis.set_title(component)
            axis.set_ylabel(MODEL_LABELS[model] + "\nmodel error / copy error")
            axis.legend(loc="lower right")
    for axis in axes[-1]:
        axis.set_xlabel("Target Bx [mT]")
        axis.set_xticks(B_VALUES)
    fig.suptitle(title + " (green line: expected same-regime target)")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_same_vs_control(rows: list[dict], output: Path) -> None:
    metrics = (
        ("field_mse", "phi", "phi MSE"),
        ("field_mse", "ey", "Ey MSE"),
        ("modal_transport_mae", "MTSI", "MTSI transport"),
        ("modal_transport_mae", "ECDI", "ECDI transport"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    targets = ((10.0, "B10 ECDI-side"), (30.0, "B30 MTSI/long-wave-side"))
    combos = (
        ("data_only_100ep", "low_E10_E20", "D-low"),
        ("data_only_100ep", "high_E30_E40", "D-high"),
        ("spectral_full_50ep", "low_E10_E20", "S-low"),
        ("spectral_full_50ep", "high_E30_E40", "S-high"),
    )
    colors = ("#94a3b8", "#475569", "#fca5a5", "#dc2626")
    for row_index, (b_mT, target_label) in enumerate(targets):
        expected_model = "high_E30_E40" if b_mT == 10.0 else "low_E10_E20"
        for column, (metric, component, label) in enumerate(metrics):
            axis = axes[row_index, column]
            values = [
                float(
                    lookup(
                        rows,
                        family,
                        model,
                        b_mT,
                        "strict_source_normalization",
                        metric,
                        component,
                    )["model_over_copy"]
                )
                for family, model, _ in combos
            ]
            bars = axis.bar(range(len(values)), values, color=colors)
            for bar, (_, model, _) in zip(bars, combos):
                if model == expected_model:
                    bar.set_edgecolor("#16a34a")
                    bar.set_linewidth(2.5)
            axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
            axis.set_yscale("log")
            axis.set_xticks(range(len(combos)), [item[2] for item in combos])
            axis.set_title(label)
            if column == 0:
                axis.set_ylabel(target_label + "\nmodel error / copy error")
    fig.suptitle("Same-regime E-trained model versus opposite-regime control (green outline: expected)")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_main_horizons(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, layout="constrained")
    targets = (
        (10.0, "high_E30_E40", "low_E10_E20", "B10 ECDI-side"),
        (30.0, "low_E10_E20", "high_E30_E40", "B30 MTSI/long-wave-side"),
    )
    styles = (
        ("data_only_100ep", "same", "#64748b", "-"),
        ("spectral_full_50ep", "same", "#dc2626", "-"),
        ("data_only_100ep", "control", "#64748b", "--"),
        ("spectral_full_50ep", "control", "#dc2626", "--"),
    )
    for row_index, (b_mT, same_model, control_model, target_label) in enumerate(targets):
        for column, component in enumerate(("phi", "ey")):
            axis = axes[row_index, column]
            for family, role, color, linestyle in styles:
                model = same_model if role == "same" else control_model
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["model_family"] == family
                        and row["model_key"] == model
                        and float(row["target_Bx_mT"]) == b_mT
                        and row["variant"] == "strict_source_normalization"
                        and row["metric"] == "field_mse"
                        and row["component"] == component
                    ),
                    key=lambda row: float(row["horizon_ns"]),
                )
                axis.plot(
                    [float(row["horizon_ns"]) for row in selected],
                    [float(row["model_over_copy"]) for row in selected],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    marker="o",
                    markersize=3,
                    label=("D" if family == "data_only_100ep" else "S")
                    + "-"
                    + role,
                )
            axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
            axis.set_yscale("log")
            axis.set_title(component)
            axis.set_ylabel(target_label + "\nmodel error / copy error")
            axis.legend(loc="upper right", ncol=2, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Forecast horizon [ns]")
    fig.suptitle("Field transfer by forecast horizon (solid: expected same regime; dashed: control)")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_snapshots(results: dict[tuple, dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(22, 8), layout="constrained")
    rows = (
        (10.0, "high_E30_E40", "low_E10_E20", "B10 ECDI-side"),
        (30.0, "low_E10_E20", "high_E30_E40", "B30 MTSI/long-wave-side"),
    )
    for row_index, (b_mT, same_model, control_model, label) in enumerate(rows):
        same = results[("spectral_full_50ep", same_model, b_mT, "strict_source_normalization")]["snapshot"]
        control = results[("spectral_full_50ep", control_model, b_mT, "strict_source_normalization")]["snapshot"]
        fields = (
            ("Last input", same["input_phi"], "field"),
            ("PIC truth", same["truth_phi"], "field"),
            ("Same-regime pred.", same["prediction_phi"], "field"),
            ("Opposite control", control["prediction_phi"], "field"),
            ("Same abs. error", np.abs(same["prediction_phi"] - same["truth_phi"]), "error"),
            ("Control abs. error", np.abs(control["prediction_phi"] - control["truth_phi"]), "error"),
        )
        field_values = np.concatenate([value.ravel() for _, value, kind in fields if kind == "field"])
        vmin, vmax = float(np.min(field_values)), float(np.max(field_values))
        for column, (title, values, kind) in enumerate(fields):
            kwargs = {"cmap": "magma"} if kind == "error" else {"cmap": "viridis", "vmin": vmin, "vmax": vmax}
            image = axes[row_index, column].imshow(values, origin="lower", aspect="auto", **kwargs)
            axes[row_index, column].set_title(title)
            axes[row_index, column].set_xlabel("Azimuthal index")
            if column == 0:
                axes[row_index, column].set_ylabel(label + "\nRadial index")
            fig.colorbar(image, ax=axes[row_index, column], fraction=0.046, pad=0.04)
    fig.suptitle("Spectral-model strict predictions near 28.6 us at 150 ns horizon")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_readme(rows: list[dict], horizon_rows: list[dict], runtime_sec: float) -> None:
    final_metrics = (
        ("field_mse", "phi", "phi"),
        ("field_mse", "ey", "Ey"),
        ("modal_transport_mae", "MTSI", "MTSI transport"),
        ("modal_transport_mae", "ECDI", "ECDI transport"),
        ("cross_phase_weighted_mae_rad", "MTSI", "MTSI cross-phase"),
        ("cross_phase_weighted_mae_rad", "ECDI", "ECDI cross-phase"),
    )
    lines = [
        "# E-sweep-trained SimVPv2 to magnetic-sweep zero-shot transfer",
        "",
        "## Question",
        "",
        "Does transfer depend more on matching the control parameter path or on matching the dominant instability regime? The E-sweep models are frozen and applied to B10--B30 without retraining.",
        "",
        "- Expected ECDI same-regime test: high-E E30+E40 model -> low-B B10.",
        "- Expected MTSI-side same-regime test: low-E E10+E20 model -> high-B B30.",
        "- B30 also contains a strong long-wavelength component, so it is not a pure MTSI replica.",
        "- Opposite-regime models on B10/B30 are explicit controls.",
        "- Strict evaluation uses only E-source training normalization. Input-window calibration uses only the ten observed target frames and no target future.",
        "- Test frames 1800--2000 correspond to 27--30 us; direct10 predicts 15--150 ns ahead.",
        "- Target mode bands are not fixed across B. They follow `n0=e Bx^2 Ly/(2 pi me Ez)`, with MTSI proxy `n/n0<=0.60` and ECDI proxy `0.75<=n/n0<=1.25`.",
        "",
        "## Target-adaptive mode bands",
        "",
        "| Bx [mT] | theoretical n0 | MTSI proxy modes | ECDI proxy modes |",
        "|---:|---:|---:|---:|",
    ]
    for b_mT in B_VALUES:
        bands = target_mode_bands(b_mT)
        lines.append(
            f"| {b_mT:g} | {ecdi_mode_number(b_mT):.3f} | "
            f"{int(bands['MTSI'][0])}--{int(bands['MTSI'][-1])} | "
            f"{int(bands['ECDI'][0])}--{int(bands['ECDI'][-1])} |"
        )
    lines.extend([
        "",
        "A ratio below 1 beats copy/persistence.",
        "",
        "## Main strict comparisons",
        "",
        "| Target | Model role | Objective | " + " | ".join(label for _, _, label in final_metrics) + " |",
        "|---|---|---|" + "---:|" * len(final_metrics),
    ])
    comparisons = (
        (10.0, "high_E30_E40", "expected same-regime ECDI"),
        (10.0, "low_E10_E20", "opposite-regime control"),
        (30.0, "low_E10_E20", "expected same-regime MTSI/long-wave"),
        (30.0, "high_E30_E40", "opposite-regime control"),
    )
    for b_mT, model, role in comparisons:
        for family, family_label in (
            ("data_only_100ep", "data-only"),
            ("spectral_full_50ep", "spectral"),
        ):
            values = [
                float(
                    lookup(
                        rows,
                        family,
                        model,
                        b_mT,
                        "strict_source_normalization",
                        metric,
                        component,
                    )["model_over_copy"]
                )
                for metric, component, _ in final_metrics
            ]
            lines.append(
                f"| B{token(b_mT)} | {role} | {family_label} | "
                + " | ".join(f"{value:.3f}" for value in values)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Main findings",
            "",
            "- B10: the expected high-E/ECDI model does not transfer. Strict phi/copy is 19.652 (data-only) and 35.060 (spectral), while the opposite low-E model is 1.112 and 1.123. Input-window calibration reduces the expected model to 1.770 and 1.401, but it still loses to copy and its adaptive-band transport/cross-phase errors remain far above copy.",
            "- B30: the low-E/MTSI-side spectral model gives partial field-level transfer: phi/copy=0.985, Ey/copy=0.668, and ion-density/copy=0.953. At the 150 ns horizon its phi/copy ratio improves to 0.584. The data-only model reaches Ey/copy=0.826 but phi/copy=1.889.",
            "- B30 is not physically reconstructed. Even the spectral low-E model has MTSI-proxy transport/copy=8.669 and cross-phase/copy=7.549. The field forecast can therefore look useful while the density--electric-field phase relation and transport are wrong.",
            "- The opposite high-E model is much worse on B30 (phi/copy=43.079 data-only and 68.436 spectral). Matching the broad MTSI side helps there, but broad regime matching is not sufficient and it does not generalize symmetrically to B10.",
            "- Spectral loss improves the B30 field forecast, but it does not rescue zero-shot modal transport. It also worsens the high-E -> B10 phi result. This is not a strict loss-only ablation because the schedules used 50 versus 100 epochs.",
            "- Overall conclusion: neither direction demonstrates instability-physics zero-shot transfer. The only positive result is partial low-E -> B30 field-level transfer, strongest at longer direct10 horizons. Control-parameter path, radial structure, amplitudes, and mode coupling remain important beyond the coarse ECDI/MTSI label.",
            "",
            "## Interpretation rule",
            "",
            "Same-regime transfer is supported only if the expected model both beats copy on relevant observables and outperforms the opposite-regime control. Lower error than the control alone is not enough. Field MSE alone is also insufficient; mode amplitude, cross-phase, and modal transport must be inspected together.",
            "",
            "The data-only/spectral comparison remains a practical model comparison rather than a strict loss ablation because the OneCycle schedules used 100 and 50 epochs, respectively.",
            "",
            "## Files",
            "",
            "- `overall_metrics.csv` and `metrics_by_horizon.csv`",
            "- `normalization_diagnostics.csv`",
            "- `mode_band_diagnostics.csv`",
            "- `field_transfer_vs_B.png`",
            "- `transport_transfer_vs_B.png`",
            "- `cross_phase_transfer_vs_B.png`",
            "- `same_regime_vs_control.png`",
            "- `same_regime_field_by_horizon.png`",
            "- `same_regime_phi_snapshots.png`",
            "",
            f"Runtime: {runtime_sec / 60.0:.1f} min.",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    base.read_test_segment = read_b_test_segment
    started = time.time()
    results: dict[tuple, dict] = {}
    overall_rows: list[dict] = []
    horizon_rows: list[dict] = []
    normalization_rows: list[dict] = []
    mode_band_rows = []
    for b_mT in B_VALUES:
        bands = target_mode_bands(b_mT)
        mode_band_rows.append(
            {
                "target_Bx_mT": b_mT,
                "theoretical_ecdi_n0": ecdi_mode_number(b_mT),
                "mtsi_n_min": int(bands["MTSI"][0]),
                "mtsi_n_max": int(bands["MTSI"][-1]),
                "ecdi_n_min": int(bands["ECDI"][0]),
                "ecdi_n_max": int(bands["ECDI"][-1]),
            }
        )

    for family, specs in MODEL_FAMILIES.items():
        for model_key, spec in specs.items():
            print(f"[MODEL] family={family} model={model_key}", flush=True)
            model = base.load_model(spec, device)
            for b_mT in B_VALUES:
                set_target_physics(b_mT)
                result = base.evaluate_case(
                    model,
                    model_key,
                    spec,
                    b_mT,
                    "strict_source_normalization",
                    device,
                    60,
                )
                result["role"] = role_for(model_key, b_mT)
                results[(family, model_key, b_mT, result["variant"])] = result
                current_overall, current_horizons = base.summarize_result(result)
                overall_rows.extend(relabel_rows(current_overall, family))
                horizon_rows.extend(relabel_rows(current_horizons, family))
                for channel_index, channel in enumerate(base.CHANNELS):
                    normalization_rows.append(
                        {
                            "model_family": family,
                            "model_key": model_key,
                            "target_Bx_mT": b_mT,
                            "variant": result["variant"],
                            "channel": channel,
                            "below_source_range_fraction": float(
                                result["normalized_below_zero_fraction"][channel_index]
                            ),
                            "above_source_range_fraction": float(
                                result["normalized_above_one_fraction"][channel_index]
                            ),
                        }
                    )

            calibrated_b = SAME_REGIME_TARGET[model_key]
            set_target_physics(calibrated_b)
            moments = base.source_reference_moments(spec)
            result = base.evaluate_case(
                model,
                model_key,
                spec,
                calibrated_b,
                "input_window_calibrated",
                device,
                60,
                source_moments=moments,
            )
            result["role"] = role_for(model_key, calibrated_b)
            results[(family, model_key, calibrated_b, result["variant"])] = result
            current_overall, current_horizons = base.summarize_result(result)
            overall_rows.extend(relabel_rows(current_overall, family))
            horizon_rows.extend(relabel_rows(current_horizons, family))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(OUTPUT / "overall_metrics.csv", overall_rows)
    write_csv(OUTPUT / "metrics_by_horizon.csv", horizon_rows)
    write_csv(OUTPUT / "normalization_diagnostics.csv", normalization_rows)
    write_csv(OUTPUT / "mode_band_diagnostics.csv", mode_band_rows)
    plot_metric_grid(
        overall_rows,
        "field_mse",
        ("phi", "ey"),
        "E-trained model field transfer across magnetic sweep",
        OUTPUT / "field_transfer_vs_B.png",
    )
    plot_metric_grid(
        overall_rows,
        "modal_transport_mae",
        ("MTSI", "ECDI"),
        "E-trained model modal-transport transfer across magnetic sweep",
        OUTPUT / "transport_transfer_vs_B.png",
    )
    plot_metric_grid(
        overall_rows,
        "cross_phase_weighted_mae_rad",
        ("MTSI", "ECDI"),
        "E-trained model cross-phase transfer across magnetic sweep",
        OUTPUT / "cross_phase_transfer_vs_B.png",
    )
    plot_same_vs_control(overall_rows, OUTPUT / "same_regime_vs_control.png")
    plot_main_horizons(horizon_rows, OUTPUT / "same_regime_field_by_horizon.png")
    plot_snapshots(results, OUTPUT / "same_regime_phi_snapshots.png")
    runtime = time.time() - started
    make_readme(overall_rows, horizon_rows, runtime)
    summary = {
        "status": "PASS",
        "device": str(device),
        "runtime_sec": runtime,
        "source_models": {
            "low_E_MTSI_side": [10.0, 20.0],
            "high_E_ECDI_side": [30.0, 40.0],
        },
        "target_Bx_mT": list(B_VALUES),
        "same_regime_tests": {
            "ECDI": "high-E E30+E40 -> B10",
            "MTSI_longwave": "low-E E10+E20 -> B30",
        },
        "mode_band_definition": {
            "n0_formula": "e * Bx^2 * Ly / (2*pi*me*Ez)",
            "mtsi_n_over_n0_max": MTSI_MAX_N_OVER_N0,
            "ecdi_n_over_n0_min": ECDI_MIN_N_OVER_N0,
            "ecdi_n_over_n0_max": ECDI_MAX_N_OVER_N0,
        },
        "output": str(OUTPUT.resolve()),
    }
    (OUTPUT / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
