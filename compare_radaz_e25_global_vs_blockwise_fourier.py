"""Compare global 12-PC and blockwise 20-D Fourier latent ROMs."""

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
DEFAULT_GLOBAL = (
    ROOT / "workdirs" / "analyze_radaz_bx20mt_ez25kvm_fourier_latent_raw"
)
DEFAULT_BLOCKWISE = (
    ROOT
    / "workdirs"
    / "analyze_radaz_bx20mt_ez25kvm_blockwise_fourier_latent"
)
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "compare_radaz_e25_global_vs_blockwise_fourier_latent"
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mode_rows(global_dir: Path, block_dir: Path) -> list[dict]:
    rows = []
    global_rows = read_csv(global_dir / "fourier_mode_forecast_metrics.csv")
    global_methods = {
        "oracle_pca": "oracle",
        "hankel_dmd": "hankel_dmd",
        "havok_zero_forcing": "havok_zero_forcing",
    }
    for row in global_rows:
        if (
            row["layer"] != "translator"
            or row["scope"] != "band"
            or row["method"] not in global_methods
        ):
            continue
        band = "MTSI" if row["label"].startswith("MTSI") else "ECDI"
        rows.append(
            {
                "representation": "global_raw_12PC",
                "dimensions": 12,
                "method": global_methods[row["method"]],
                "band": band,
                "coefficient_nrmse": float(row["coefficient_nrmse"]),
                "amplitude_correlation": float(row["amplitude_correlation"]),
                "mean_amplitude_ratio": float(row["mean_amplitude_ratio"]),
            }
        )

    block_rows = read_csv(block_dir / "blockwise_mode_forecast_metrics.csv")
    block_methods = {
        "oracle_block_pca": "oracle",
        "hankel_dmd": "hankel_dmd",
        "havok_zero_forcing": "havok_zero_forcing",
    }
    for row in block_rows:
        if row["method"] not in block_methods or row["block"] not in (
            "MTSI_n1_6",
            "ECDI_n9_21",
        ):
            continue
        rows.append(
            {
                "representation": "blockwise_20D",
                "dimensions": 20,
                "method": block_methods[row["method"]],
                "band": "MTSI" if row["block"].startswith("MTSI") else "ECDI",
                "coefficient_nrmse": float(row["coefficient_nrmse"]),
                "amplitude_correlation": float(row["amplitude_correlation"]),
                "mean_amplitude_ratio": float(row["mean_amplitude_ratio"]),
            }
        )
    return rows


def state_rows(global_dir: Path, block_dir: Path) -> list[dict]:
    global_summary = json.loads(
        (global_dir / "fourier_latent_dynamics_summary.json").read_text(
            encoding="utf-8"
        )
    )["dynamics"]["translator"]
    block_summary = json.loads(
        (block_dir / "blockwise_fourier_dynamics_summary.json").read_text(
            encoding="utf-8"
        )
    )["final"]
    rows = []
    for representation, dimensions, summary in (
        ("global_raw_12PC", 12, global_summary),
        ("blockwise_20D", 20, block_summary),
    ):
        for method in ("hankel_dmd", "havok_zero_forcing"):
            metrics = summary[
                "metrics" if representation == "global_raw_12PC" else "state_metrics"
            ][method]["24-30"]
            rows.append(
                {
                    "representation": representation,
                    "dimensions": dimensions,
                    "method": method,
                    "trajectory_correlation": float(
                        metrics["flattened_correlation"]
                    ),
                    "skill_vs_training_mean": float(
                        metrics["skill_vs_training_mean"]
                    ),
                    "skill_vs_persistence": float(
                        metrics["skill_vs_persistence"]
                    ),
                }
            )
    return rows


def plot(path: Path, modes: list[dict], states: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    representations = ("global_raw_12PC", "blockwise_20D")
    methods = ("oracle", "hankel_dmd", "havok_zero_forcing")
    colors = {
        "oracle": "#777777",
        "hankel_dmd": "#0072b2",
        "havok_zero_forcing": "#009e73",
    }
    x = np.arange(len(representations), dtype=float)
    width = 0.24
    for axis, band in zip(axes[0], ("MTSI", "ECDI")):
        for index, method in enumerate(methods):
            selected = [
                next(
                    row
                    for row in modes
                    if row["representation"] == representation
                    and row["method"] == method
                    and row["band"] == band
                )
                for representation in representations
            ]
            axis.bar(
                x + (index - 1) * width,
                [row["coefficient_nrmse"] for row in selected],
                width,
                color=colors[method],
                label=method,
            )
        axis.set_title(f"{band} complex coefficient forecast")
        axis.set_ylabel("NRMSE (lower is better)")
        axis.set_xticks(x, representations)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="upper right")

    state_methods = ("hankel_dmd", "havok_zero_forcing")
    state_colors = ("#0072b2", "#009e73")
    for axis, metric, title in (
        (axes[1, 0], "trajectory_correlation", "Full reduced-state trajectory"),
        (axes[1, 1], "skill_vs_training_mean", "Skill against training mean"),
    ):
        for index, (method, color) in enumerate(zip(state_methods, state_colors)):
            selected = [
                next(
                    row
                    for row in states
                    if row["representation"] == representation
                    and row["method"] == method
                )
                for representation in representations
            ]
            axis.bar(
                x + (index - 0.5) * 0.34,
                [row[metric] for row in selected],
                0.34,
                color=color,
                label=method,
            )
        axis.set_title(title)
        axis.set_ylabel(metric)
        axis.set_xticks(x, representations)
        axis.axhline(0.0, color="#111111", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="lower right")
    fig.suptitle("E25 global vs blockwise Fourier latent dynamics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL)
    parser.add_argument("--block-dir", type=Path, default=DEFAULT_BLOCKWISE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    modes = mode_rows(args.global_dir, args.block_dir)
    states = state_rows(args.global_dir, args.block_dir)
    write_csv(args.output / "mode_comparison.csv", modes)
    write_csv(args.output / "state_comparison.csv", states)
    plot(args.output / "global_vs_blockwise_fourier_comparison.png", modes, states)
    (args.output / "README.md").write_text(
        """# E25 global vs blockwise Fourier latent comparison

This folder compares the original global raw-Fourier 12-PC state with the
20-dimensional blockwise state selected using only 23-24 us validation data.

The blockwise state improves the PCA ceiling and MTSI complex-coefficient
forecast, but its full autonomous state trajectory is slightly worse. MTSI
amplitude timing remains unresolved, so block separation alone does not close
the instability-envelope dynamics.

## 日本語メモ

mode帯別PCAによりMTSI/ECDI情報の保持上限は改善したが、状態全体の自律予測
は従来12 PCよりわずかに悪化した。MTSI包絡の増減時刻も再現できていない。
したがって、単純なmode分割だけで統一的な閉じた低次元力学が得られたとは
結論できない。
""",
        encoding="utf-8",
    )
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
