#!/usr/bin/env python3
"""Compare matched RadAz regime-transfer metrics for data-only and spectral models."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT / "workdirs" / "2D_RadAz"
DATA_DIR = WORKDIR / "compare_radaz_regime_generalization_direct10"
SPECTRAL_DIR = (
    WORKDIR / "compare_radaz_regime_generalization_direct10_spectral_full_50ep"
)
OUTPUT = WORKDIR / "compare_radaz_regime_generalization_data_vs_spectral_50ep"
E_VALUES = (10.0, 20.0, 22.5, 25.0, 30.0, 40.0)
MODEL_LABELS = {
    "low_E10_E20": "Low-E model (E10+E20)",
    "high_E30_E40": "High-E model (E30+E40)",
}
FINAL_TARGETS = {"low_E10_E20": 40.0, "high_E30_E40": 10.0}
SOURCE_VALUES = {
    "low_E10_E20": (10.0, 20.0),
    "high_E30_E40": (30.0, 40.0),
}
SELECTED_METRICS = (
    ("field_mse", "phi", "phi MSE"),
    ("field_mse", "ey", "Ey MSE"),
    ("phi_band_amplitude_mae", "MTSI", "MTSI phi amplitude"),
    ("phi_band_amplitude_mae", "ECDI", "ECDI phi amplitude"),
    ("modal_transport_mae", "MTSI", "MTSI transport"),
    ("modal_transport_mae", "ECDI", "ECDI transport"),
    ("cross_phase_weighted_mae_rad", "MTSI", "MTSI cross-phase"),
    ("cross_phase_weighted_mae_rad", "ECDI", "ECDI cross-phase"),
)
FINAL_METRICS = (
    ("field_mse", "phi", "phi MSE"),
    ("field_mse", "ey", "Ey MSE"),
    ("modal_transport_mae", "MTSI", "MTSI transport"),
    ("modal_transport_mae", "ECDI", "ECDI transport"),
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row: dict) -> tuple:
    return (
        row["model_key"],
        float(row["target_Ez_kVm"]),
        row["role"],
        row["variant"],
        row["metric"],
        row["component"],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lookup(
    rows: list[dict], model: str, target: float, variant: str, metric: str, component: str
) -> dict:
    matches = [
        row
        for row in rows
        if row["model_key"] == model
        and float(row["target_Ez_kVm"]) == target
        and row["variant"] == variant
        and row["metric"] == metric
        and row["component"] == component
    ]
    if len(matches) != 1:
        raise ValueError((model, target, variant, metric, component, len(matches)))
    return matches[0]


def plot_improvement_heatmap(rows: list[dict], output: Path) -> None:
    strict = [row for row in rows if row["variant"] == "strict_source_normalization"]
    labels = []
    matrix = []
    for model in MODEL_LABELS:
        for metric, component, label in SELECTED_METRICS:
            labels.append(("Low" if model.startswith("low") else "High") + ": " + label)
            values = []
            for target in E_VALUES:
                matches = [
                    row
                    for row in strict
                    if row["model_key"] == model
                    and float(row["target_Ez_kVm"]) == target
                    and row["metric"] == metric
                    and row["component"] == component
                ]
                values.append(float(matches[0]["spectral_error_over_data_only"]))
            matrix.append(values)
    matrix = np.asarray(matrix)
    displayed = np.log10(np.maximum(matrix, 1.0e-3))
    fig, axis = plt.subplots(figsize=(11, 9), layout="constrained")
    image = axis.imshow(displayed, cmap="RdYlBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(E_VALUES)), [str(value).rstrip("0").rstrip(".") for value in E_VALUES])
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Target Ez [kV/m]")
    axis.set_title("Spectral-loss error / data-only error (strict normalization)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value < 0.2 or value > 5.0 else "black"
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("log10(spectral error / data-only error); below 1 improves")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_final_transfer(data_rows: list[dict], spectral_rows: list[dict], output: Path) -> None:
    metrics = FINAL_METRICS
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, layout="constrained")
    for row_index, model in enumerate(MODEL_LABELS):
        target = FINAL_TARGETS[model]
        for column, (metric, component, label) in enumerate(metrics):
            axis = axes[row_index, column]
            variants = ("strict_source_normalization", "input_window_calibrated")
            x = np.arange(len(variants))
            data_values = [
                float(lookup(data_rows, model, target, variant, metric, component)["model_over_copy"])
                for variant in variants
            ]
            spectral_values = [
                float(lookup(spectral_rows, model, target, variant, metric, component)["model_over_copy"])
                for variant in variants
            ]
            axis.bar(x - 0.18, data_values, 0.36, label="data-only", color="#64748b")
            axis.bar(x + 0.18, spectral_values, 0.36, label="spectral", color="#dc2626")
            axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
            axis.set_yscale("log")
            axis.set_xticks(x, ("strict", "input calibrated"))
            axis.set_title(label)
            if column == 0:
                direction = "E10+E20 -> E40" if model.startswith("low") else "E30+E40 -> E10"
                axis.set_ylabel(direction + "\nmodel error / copy error")
            if row_index == 0 and column == 3:
                axis.legend(loc="upper right")
    fig.suptitle("Final opposite-regime zero-shot transfer")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def pct_change(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data_rows = read_csv(DATA_DIR / "overall_metrics.csv")
    spectral_rows = read_csv(SPECTRAL_DIR / "overall_metrics.csv")
    data_index = {key(row): row for row in data_rows}
    spectral_index = {key(row): row for row in spectral_rows}
    if data_index.keys() != spectral_index.keys():
        raise ValueError("Data-only and spectral evaluations do not have matching metric rows")

    comparison = []
    for current_key in sorted(data_index, key=str):
        data = data_index[current_key]
        spectral = spectral_index[current_key]
        data_error = float(data["model_error"])
        spectral_error = float(spectral["model_error"])
        comparison.append(
            {
                "model_key": data["model_key"],
                "target_Ez_kVm": data["target_Ez_kVm"],
                "role": data["role"],
                "variant": data["variant"],
                "metric": data["metric"],
                "component": data["component"],
                "copy_error": data["copy_error"],
                "data_only_model_error": data["model_error"],
                "spectral_model_error": spectral["model_error"],
                "data_only_model_over_copy": data["model_over_copy"],
                "spectral_model_over_copy": spectral["model_over_copy"],
                "spectral_error_over_data_only": spectral_error / data_error,
                "spectral_error_change_percent": pct_change(spectral_error, data_error),
            }
        )
    write_csv(OUTPUT / "spectral_vs_data_only_overall.csv", comparison)
    plot_improvement_heatmap(comparison, OUTPUT / "strict_spectral_error_over_data_only.png")
    plot_final_transfer(data_rows, spectral_rows, OUTPUT / "final_zero_shot_data_vs_spectral.png")

    table_rows = []
    for model, target in FINAL_TARGETS.items():
        direction = "E10+E20 -> E40" if model.startswith("low") else "E30+E40 -> E10"
        for variant in ("strict_source_normalization", "input_window_calibrated"):
            for metric, component, label in FINAL_METRICS:
                data = float(lookup(data_rows, model, target, variant, metric, component)["model_over_copy"])
                spectral = float(lookup(spectral_rows, model, target, variant, metric, component)["model_over_copy"])
                table_rows.append((direction, variant, label, data, spectral, spectral / data))

    strict_selected = [
        row
        for row in comparison
        if row["variant"] == "strict_source_normalization"
        and any(row["metric"] == metric and row["component"] == component for metric, component, _ in SELECTED_METRICS)
    ]
    improved = sum(float(row["spectral_error_over_data_only"]) < 1.0 for row in strict_selected)
    lines = [
        "# RadAz data-only vs spectral-loss regime transfer",
        "",
        "## Protocol",
        "",
        "Both runs use the same source manifests, 8:1:1 temporal split, architecture, direct10 stride1 setup, source-only normalization, test frames 1800--2000, and copy baseline. The spectral run adds azimuthal Fourier amplitude, phase, and density--Ey cross-phase losses.",
        "",
        "This is a matched practical model comparison, not a strict one-factor loss ablation: data-only used a 100-epoch OneCycle schedule (selected checkpoints: epochs 29/30, one-based), whereas spectral used 50 epochs (selected checkpoints: epochs 21/37). A future causal ablation must match the epoch budget and learning-rate schedule.",
        "",
        "## Final zero-shot comparison",
        "",
        "All values are model error / copy error; below 1 beats copy. `spectral/data` below 1 means the spectral objective improves on data-only.",
        "",
        "| Direction | Variant | Metric | Data-only/copy | Spectral/copy | Spectral/data |",
        "|---|---|---|---:|---:|---:|",
    ]
    for direction, variant, label, data, spectral, ratio in table_rows:
        lines.append(f"| {direction} | {variant} | {label} | {data:.3f} | {spectral:.3f} | {ratio:.3f} |")

    low_strict_data = float(lookup(data_rows, "low_E10_E20", 40.0, "strict_source_normalization", "modal_transport_mae", "ECDI")["model_over_copy"])
    low_strict_spectral = float(lookup(spectral_rows, "low_E10_E20", 40.0, "strict_source_normalization", "modal_transport_mae", "ECDI")["model_over_copy"])
    low_cal_data = float(lookup(data_rows, "low_E10_E20", 40.0, "input_window_calibrated", "field_mse", "phi")["model_over_copy"])
    low_cal_spectral = float(lookup(spectral_rows, "low_E10_E20", 40.0, "input_window_calibrated", "field_mse", "phi")["model_over_copy"])
    high_strict_data = float(lookup(data_rows, "high_E30_E40", 10.0, "strict_source_normalization", "field_mse", "phi")["model_over_copy"])
    high_strict_spectral = float(lookup(spectral_rows, "high_E30_E40", 10.0, "strict_source_normalization", "field_mse", "phi")["model_over_copy"])
    low_amp_data = float(lookup(data_rows, "low_E10_E20", 40.0, "strict_source_normalization", "phi_band_amplitude_mae", "ECDI")["model_over_copy"])
    low_amp_spectral = float(lookup(spectral_rows, "low_E10_E20", 40.0, "strict_source_normalization", "phi_band_amplitude_mae", "ECDI")["model_over_copy"])
    low_phase_data = float(lookup(data_rows, "low_E10_E20", 40.0, "strict_source_normalization", "cross_phase_weighted_mae_rad", "ECDI")["model_over_copy"])
    low_phase_spectral = float(lookup(spectral_rows, "low_E10_E20", 40.0, "strict_source_normalization", "cross_phase_weighted_mae_rad", "ECDI")["model_over_copy"])
    high_amp_data = float(lookup(data_rows, "high_E30_E40", 10.0, "strict_source_normalization", "phi_band_amplitude_mae", "ECDI")["model_over_copy"])
    high_amp_spectral = float(lookup(spectral_rows, "high_E30_E40", 10.0, "strict_source_normalization", "phi_band_amplitude_mae", "ECDI")["model_over_copy"])
    high_phase_data = float(lookup(data_rows, "high_E30_E40", 10.0, "strict_source_normalization", "cross_phase_weighted_mae_rad", "ECDI")["model_over_copy"])
    high_phase_spectral = float(lookup(spectral_rows, "high_E30_E40", 10.0, "strict_source_normalization", "cross_phase_weighted_mae_rad", "ECDI")["model_over_copy"])
    lines.extend(
        [
            "",
            "## Main findings",
            "",
            f"- Across the {len(strict_selected)} matched strict metric/case combinations shown in the heatmap, spectral loss reduced error in {improved} ({100.0 * improved / len(strict_selected):.1f}%). The effect is mixed rather than uniformly beneficial.",
            f"- Low-to-high ECDI transport improved strongly: `{low_strict_data:.1f}` to `{low_strict_spectral:.1f}` times copy ({pct_change(low_strict_spectral, low_strict_data):+.1f}%), but remains far from beating copy.",
            f"- Low-to-high calibrated phi improved from `{low_cal_data:.3f}` to `{low_cal_spectral:.3f}` times copy ({pct_change(low_cal_spectral, low_cal_data):+.1f}%), but still does not cross the success threshold of 1.",
            f"- High-to-low strict phi worsened from `{high_strict_data:.1f}` to `{high_strict_spectral:.1f}` times copy ({pct_change(high_strict_spectral, high_strict_data):+.1f}%).",
            f"- In low-to-high ECDI, phi-band amplitude improved `{low_amp_data:.1f}` to `{low_amp_spectral:.1f}` and cross-phase `{low_phase_data:.2f}` to `{low_phase_spectral:.2f}` times copy. This is consistent with the large transport improvement, although none reaches copy accuracy.",
            f"- In high-to-low ECDI, cross-phase improved `{high_phase_data:.1f}` to `{high_phase_spectral:.1f}`, but phi-band amplitude worsened `{high_amp_data:.1f}` to `{high_amp_spectral:.1f}`. Transport therefore did not improve. Correct phase alone is insufficient when mode amplitude is wrong.",
            "- Low-to-high input-calibrated phi beats copy only at the last three horizons (120--150 ns), not over the full 15--150 ns aggregate. This is partial long-horizon field skill, not recovery of the target instability mechanism.",
            "- The spectral-trained checkpoints change mode-sensitive metrics substantially, consistent with the added objective affecting optimization. Because the epoch and OneCycle schedules differ, the exact change cannot be attributed to the loss alone.",
            "- In this practical comparison, mode-aware regularization is not sufficient to create a common low-E/high-E evolution operator.",
            "- The defensible result is partial improvement in selected observables, not opposite-regime zero-shot generalization.",
            "",
            "## Files",
            "",
            "- `spectral_vs_data_only_overall.csv`: matched metric rows and direct error ratios.",
            "- `strict_spectral_error_over_data_only.png`: all selected strict improvements and regressions.",
            "- `final_zero_shot_data_vs_spectral.png`: final opposite-regime comparison against copy.",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS",
        "matched_strict_selected_rows": len(strict_selected),
        "spectral_improved_rows": improved,
        "spectral_improved_fraction": improved / len(strict_selected),
        "comparison_caveat": "Data-only used 100-epoch OneCycle training and spectral used 50 epochs; this is not a strict single-factor loss ablation.",
        "best_checkpoint_epoch_one_based": {
            "data_only_low": 29,
            "data_only_high": 30,
            "spectral_low": 21,
            "spectral_high": 37,
        },
        "output": str(OUTPUT.resolve()),
    }
    (OUTPUT / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
