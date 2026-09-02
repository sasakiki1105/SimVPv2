"""Compare E25 Fourier latent dynamics from data-only and spectral models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_RAW = (
    ROOT / "workdirs" / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_raw"
)
DEFAULT_SPECTRAL_RAW = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_spectral_full_raw"
)
DEFAULT_DATA_BLOCK = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_blockwise_fourier_latent"
)
DEFAULT_SPECTRAL_BLOCK = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_blockwise_fourier_latent_spectral_full"
)
DEFAULT_DATA_PHYSICAL = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_to_physical_modes"
)
DEFAULT_SPECTRAL_PHYSICAL = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_spectral_full_to_physical_modes"
)
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "compare_radaz_e25_data_vs_spectral_fourier_latent"
)

CHECKPOINTS = ("data-only", "spectral-loss")
METHODS = ("oracle", "hankel_dmd", "havok_zero_forcing")
COLORS = {
    "oracle": "#777777",
    "hankel_dmd": "#0072b2",
    "havok_zero_forcing": "#009e73",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_raw(checkpoint: str, path: Path) -> tuple[list[dict], list[dict]]:
    summary = json.loads(
        (path / "fourier_latent_dynamics_summary.json").read_text(
            encoding="utf-8"
        )
    )
    translator = summary["dynamics"]["translator"]
    dimensions = int(
        summary["pca"]["layers"]["translator"]["components_for_target"]
    )
    state_rows = []
    for method in ("hankel_dmd", "havok_zero_forcing"):
        metrics = translator["metrics"][method]["24-30"]
        state_rows.append(
            {
                "checkpoint": checkpoint,
                "representation": "global_raw",
                "dimensions": dimensions,
                "method": method,
                "selected_delay": translator["selected_delay"],
                "selected_rank": translator["selected_rank"],
                "trajectory_correlation": metrics["flattened_correlation"],
                "skill_vs_training_mean": metrics["skill_vs_training_mean"],
                "skill_vs_persistence": metrics["skill_vs_persistence"],
            }
        )
    method_map = {
        "oracle_pca": "oracle",
        "hankel_dmd": "hankel_dmd",
        "havok_zero_forcing": "havok_zero_forcing",
    }
    mode_rows = []
    for row in read_csv(path / "fourier_mode_forecast_metrics.csv"):
        if (
            row["layer"] != "translator"
            or row["scope"] != "band"
            or row["method"] not in method_map
        ):
            continue
        mode_rows.append(
            {
                "checkpoint": checkpoint,
                "representation": "global_raw",
                "dimensions": dimensions,
                "method": method_map[row["method"]],
                "band": "MTSI" if row["label"].startswith("MTSI") else "ECDI",
                "coefficient_nrmse": float(row["coefficient_nrmse"]),
                "amplitude_correlation": float(row["amplitude_correlation"]),
                "mean_amplitude_ratio": float(row["mean_amplitude_ratio"]),
            }
        )
    return state_rows, mode_rows


def load_block(checkpoint: str, path: Path) -> tuple[list[dict], list[dict]]:
    summary = json.loads(
        (path / "blockwise_fourier_dynamics_summary.json").read_text(
            encoding="utf-8"
        )
    )["final"]
    state_rows = []
    for method in ("hankel_dmd", "havok_zero_forcing"):
        metrics = summary["state_metrics"][method]["24-30"]
        state_rows.append(
            {
                "checkpoint": checkpoint,
                "representation": "blockwise",
                "dimensions": summary["components"],
                "selected_budget": summary["selected_budget"],
                "method": method,
                "selected_delay": summary["selected_delay"],
                "selected_rank": summary["selected_rank"],
                "trajectory_correlation": metrics["flattened_correlation"],
                "skill_vs_training_mean": metrics["skill_vs_training_mean"],
                "skill_vs_persistence": metrics["skill_vs_persistence"],
            }
        )
    method_map = {
        "oracle_block_pca": "oracle",
        "hankel_dmd": "hankel_dmd",
        "havok_zero_forcing": "havok_zero_forcing",
    }
    mode_rows = []
    for row in read_csv(path / "blockwise_mode_forecast_metrics.csv"):
        if row["method"] not in method_map or row["block"] not in (
            "MTSI_n1_6",
            "ECDI_n9_21",
        ):
            continue
        mode_rows.append(
            {
                "checkpoint": checkpoint,
                "representation": "blockwise",
                "dimensions": summary["components"],
                "selected_budget": summary["selected_budget"],
                "method": method_map[row["method"]],
                "band": "MTSI" if row["block"].startswith("MTSI") else "ECDI",
                "coefficient_nrmse": float(row["coefficient_nrmse"]),
                "amplitude_correlation": float(row["amplitude_correlation"]),
                "mean_amplitude_ratio": float(row["mean_amplitude_ratio"]),
            }
        )
    return state_rows, mode_rows


def load_physical(checkpoint: str, path: Path) -> list[dict]:
    fields = read_csv(path / "physical_mode_field_metrics.csv")
    physics = read_csv(path / "physical_mode_physics_metrics.csv")
    field_lookup = {
        (row["method"], row["field"], row["band"]): row for row in fields
    }
    physics_lookup = {
        (row["method"], row["band"]): row for row in physics
    }
    method_map = {
        "oracle_pca": "oracle",
        "hankel_dmd": "hankel_dmd",
        "havok_zero_forcing": "havok_zero_forcing",
    }
    rows = []
    for source_method, method in method_map.items():
        for band in ("MTSI", "ECDI"):
            phi = field_lookup[(source_method, "phi", band)]
            physical = physics_lookup[(source_method, band)]
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "method": method,
                    "band": band,
                    "phi_coefficient_nrmse": float(phi["coefficient_nrmse"]),
                    "phi_amplitude_correlation": float(
                        phi["amplitude_correlation"]
                    ),
                    "phi_mean_amplitude_ratio": float(
                        phi["mean_amplitude_ratio"]
                    ),
                    "cross_phase_mae_rad": float(
                        physical["cross_phase_mae_rad"]
                    ),
                    "transport_correlation": float(
                        physical["transport_correlation"]
                    ),
                    "transport_nrmse": float(physical["transport_nrmse"]),
                    "transport_skill_vs_copy": float(
                        physical["transport_mse_skill_vs_physical_copy"]
                    ),
                }
            )
    return rows


def grouped_bars(
    axis,
    rows: list[dict],
    metric: str,
    title: str,
    ylabel: str,
    band: str | None = None,
) -> None:
    x = np.arange(len(CHECKPOINTS), dtype=float)
    methods = tuple(
        method for method in METHODS if any(row["method"] == method for row in rows)
    )
    width = 0.72 / len(methods)
    for index, method in enumerate(methods):
        values = []
        for checkpoint in CHECKPOINTS:
            selected = [
                row
                for row in rows
                if row["checkpoint"] == checkpoint
                and row["method"] == method
                and (band is None or row.get("band") == band)
            ]
            values.append(float(selected[0][metric]))
        axis.bar(
            x + (index - (len(methods) - 1) / 2) * width,
            values,
            width,
            color=COLORS[method],
            label=method,
        )
    axis.set_xticks(x, CHECKPOINTS)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.axhline(0.0, color="#111111", linewidth=0.8)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower right", fontsize=8)


def plot_latent(
    path: Path, state_rows: list[dict], mode_rows: list[dict], title: str
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    grouped_bars(
        axes[0, 0],
        state_rows,
        "trajectory_correlation",
        "Autonomous reduced-state trajectory",
        "correlation",
    )
    grouped_bars(
        axes[0, 1],
        state_rows,
        "skill_vs_training_mean",
        "Skill against training mean",
        "skill",
    )
    grouped_bars(
        axes[1, 0],
        mode_rows,
        "amplitude_correlation",
        "MTSI latent amplitude",
        "correlation",
        "MTSI",
    )
    grouped_bars(
        axes[1, 1],
        mode_rows,
        "amplitude_correlation",
        "ECDI latent amplitude",
        "correlation",
        "ECDI",
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_physical(path: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    grouped_bars(
        axes[0, 0],
        rows,
        "phi_amplitude_correlation",
        "Physical phi MTSI amplitude",
        "correlation",
        "MTSI",
    )
    grouped_bars(
        axes[0, 1],
        rows,
        "phi_amplitude_correlation",
        "Physical phi ECDI amplitude",
        "correlation",
        "ECDI",
    )
    grouped_bars(
        axes[1, 0],
        rows,
        "transport_correlation",
        "MTSI modal transport",
        "correlation",
        "MTSI",
    )
    grouped_bars(
        axes[1, 1],
        rows,
        "transport_correlation",
        "ECDI modal transport",
        "correlation",
        "ECDI",
    )
    fig.suptitle("E25 physical observables decoded from Fourier latent states")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(
    path: Path,
    raw_state: list[dict],
    raw_modes: list[dict],
    block_state: list[dict],
    block_modes: list[dict],
    physical: list[dict],
) -> None:
    def value(rows, checkpoint, method, metric, band=None):
        row = next(
            row
            for row in rows
            if row["checkpoint"] == checkpoint
            and row["method"] == method
            and (band is None or row.get("band") == band)
        )
        return float(row[metric])

    raw_table = []
    for checkpoint in CHECKPOINTS:
        raw_table.append(
            "| {checkpoint} | {corr:.3f} | {skill:.3f} | {mtsi:.3f} | "
            "{ecdi:.3f} |".format(
                checkpoint=checkpoint,
                corr=value(raw_state, checkpoint, "hankel_dmd", "trajectory_correlation"),
                skill=value(raw_state, checkpoint, "hankel_dmd", "skill_vs_training_mean"),
                mtsi=value(raw_modes, checkpoint, "hankel_dmd", "amplitude_correlation", "MTSI"),
                ecdi=value(raw_modes, checkpoint, "hankel_dmd", "amplitude_correlation", "ECDI"),
            )
        )
    block_table = []
    for checkpoint in CHECKPOINTS:
        state = next(
            row
            for row in block_state
            if row["checkpoint"] == checkpoint
            and row["method"] == "hankel_dmd"
        )
        block_table.append(
            "| {checkpoint} | {dimensions} | {corr:.3f} | {skill:.3f} | "
            "{mtsi:.3f} | {ecdi:.3f} |".format(
                checkpoint=checkpoint,
                dimensions=state["dimensions"],
                corr=state["trajectory_correlation"],
                skill=state["skill_vs_training_mean"],
                mtsi=value(block_modes, checkpoint, "hankel_dmd", "amplitude_correlation", "MTSI"),
                ecdi=value(block_modes, checkpoint, "hankel_dmd", "amplitude_correlation", "ECDI"),
            )
        )
    physical_table = []
    for checkpoint in CHECKPOINTS:
        physical_table.append(
            "| {checkpoint} | {oracle_m:.3f} | {oracle_e:.3f} | {hankel_m:.3f} | "
            "{hankel_e:.3f} | {transport_m:.3f} | {transport_e:.3f} |".format(
                checkpoint=checkpoint,
                oracle_m=value(physical, checkpoint, "oracle", "phi_amplitude_correlation", "MTSI"),
                oracle_e=value(physical, checkpoint, "oracle", "phi_amplitude_correlation", "ECDI"),
                hankel_m=value(physical, checkpoint, "hankel_dmd", "phi_amplitude_correlation", "MTSI"),
                hankel_e=value(physical, checkpoint, "hankel_dmd", "phi_amplitude_correlation", "ECDI"),
                transport_m=value(physical, checkpoint, "hankel_dmd", "transport_correlation", "MTSI"),
                transport_e=value(physical, checkpoint, "hankel_dmd", "transport_correlation", "ECDI"),
            )
        )
    text = f"""# E25 data-only vs spectral-loss Fourier latent analysis

Both checkpoints were trained only on Ez=10 kV/m. The same Ez=25 kV/m H5,
target-specific diagnostic normalization, unpooled latent FFT, fit intervals,
and candidate grids were used. Only the checkpoint/training loss differs.

## Raw Fourier 12-PC state

| checkpoint | Hankel state correlation | skill vs train mean | MTSI amplitude corr | ECDI amplitude corr |
|---|---:|---:|---:|---:|
{chr(10).join(raw_table)}

## Validation-selected blockwise state

| checkpoint | dimensions | Hankel state correlation | skill vs train mean | MTSI amplitude corr | ECDI amplitude corr |
|---|---:|---:|---:|---:|---:|
{chr(10).join(block_table)}

## Decoded physical modes

| checkpoint | Oracle phi MTSI corr | Oracle phi ECDI corr | Hankel phi MTSI corr | Hankel phi ECDI corr | Hankel MTSI transport corr | Hankel ECDI transport corr |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(physical_table)}

## 日本語メモ

spectral lossはE25を学習していないにもかかわらず、物理phi modeのOracle上限を
改善した。したがって、10 kV/mで加えたmode・位相制約が、E25でも物理modeを
保持しやすい内部表現を作った可能性がある。

一方、raw 12 PCの全潜在軌道と物理ECDIの自律予測はdata-onlyより悪化した。
blockwise化するとspectral潜在のECDI予測は改善するが、MTSI包絡と全状態軌道は
悪化する。spectral lossは表現力を改善したが、単一の自律Hankel/HAVOKで閉じる
力学を作ったとは言えない。輸送相関も改善せず、輸送閉包にはradial構造、複素
cross-spectrum、modal transportまたはforcingを明示状態として追加する必要がある。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-raw", type=Path, default=DEFAULT_DATA_RAW)
    parser.add_argument("--spectral-raw", type=Path, default=DEFAULT_SPECTRAL_RAW)
    parser.add_argument("--data-block", type=Path, default=DEFAULT_DATA_BLOCK)
    parser.add_argument("--spectral-block", type=Path, default=DEFAULT_SPECTRAL_BLOCK)
    parser.add_argument("--data-physical", type=Path, default=DEFAULT_DATA_PHYSICAL)
    parser.add_argument("--spectral-physical", type=Path, default=DEFAULT_SPECTRAL_PHYSICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw_state = []
    raw_modes = []
    block_state = []
    block_modes = []
    physical = []
    for checkpoint, raw, block, decoded in (
        ("data-only", args.data_raw, args.data_block, args.data_physical),
        ("spectral-loss", args.spectral_raw, args.spectral_block, args.spectral_physical),
    ):
        states, modes = load_raw(checkpoint, raw)
        raw_state.extend(states)
        raw_modes.extend(modes)
        states, modes = load_block(checkpoint, block)
        block_state.extend(states)
        block_modes.extend(modes)
        physical.extend(load_physical(checkpoint, decoded))

    write_csv(args.output / "raw_state_metrics.csv", raw_state)
    write_csv(args.output / "raw_mode_metrics.csv", raw_modes)
    write_csv(args.output / "blockwise_state_metrics.csv", block_state)
    write_csv(args.output / "blockwise_mode_metrics.csv", block_modes)
    write_csv(args.output / "physical_mode_and_transport_metrics.csv", physical)
    plot_latent(
        args.output / "raw_fourier_checkpoint_comparison.png",
        raw_state,
        raw_modes,
        "E25 raw Fourier latent state: checkpoint comparison",
    )
    plot_latent(
        args.output / "blockwise_checkpoint_comparison.png",
        block_state,
        block_modes,
        "E25 blockwise Fourier latent state: checkpoint comparison",
    )
    plot_physical(
        args.output / "physical_checkpoint_comparison.png", physical
    )
    write_readme(
        args.output / "README.md",
        raw_state,
        raw_modes,
        block_state,
        block_modes,
        physical,
    )
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
