#!/usr/bin/env python3
"""Place B25 rolling-ROM closure metrics and mode diagnostics on one time axis."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMPONENTS = ("long", "mtsi", "ecdi")
COLORS = {"long": "#c0392b", "mtsi": "#218c4f", "ecdi": "#2469a0"}
LABELS = {"long": "Long wavelength", "mtsi": "MTSI", "ecdi": "ECDI"}
TIME_MIN_US = 20.0
TIME_MAX_US = 40.0


def parse_args() -> argparse.Namespace:
    research = Path(__file__).resolve().parents[1]
    physics = (
        research
        / "research_results"
        / "2D_RadAz"
        / "PEPAPIC"
        / "2D_Landmark"
        / "analysis_results"
        / "compare_magnetic_field_sweep_three_component_B25_B30mT_E10kVm"
        / "B25mT"
    )
    rom = (
        research
        / "research_results"
        / "2D_RadAz"
        / "SimVPv2"
        / "workdirs"
        / "analyze_radaz_b25_rolling_window_rom_0to40us"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-dir", type=Path, default=physics)
    parser.add_argument("--rom-dir", type=Path, default=rom)
    parser.add_argument("--output-dir", type=Path, default=rom / "closure_physics_overlay")
    parser.add_argument("--smooth-us", type=float, default=0.30)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smooth(values: np.ndarray, frames: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if frames <= 1:
        return values.copy()
    kernel = np.ones(frames, dtype=np.float64) / frames
    left = frames // 2
    right = frames - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def circular_smooth_deg(values: np.ndarray, frames: int) -> np.ndarray:
    radians = np.radians(np.asarray(values, dtype=np.float64))
    real = smooth(np.cos(radians), frames)
    imag = smooth(np.sin(radians), frames)
    return np.degrees(np.arctan2(imag, real))


def circular_mean_deg(values: np.ndarray) -> float:
    radians = np.radians(np.asarray(values, dtype=np.float64))
    return float(np.degrees(np.angle(np.mean(np.exp(1j * radians)))))


def float_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def selected_bicoherence(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
    """Match the triad selection used in the original magnetic-sweep figure."""
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["channel"],
            row["output_component"],
            row["kind"],
            row["mode_a"],
            row["mode_b"],
            row["mode_c"],
        )
        groups[key].append(row)

    selected: dict[tuple[str, str], dict] = {}
    for channel in ("field", "density"):
        for component in COMPONENTS:
            candidates = [
                (key, values)
                for key, values in groups.items()
                if key[0] == channel and key[1] == component
            ]
            key, values = max(
                candidates,
                key=lambda pair: max(float(item["bicoherence_squared"]) for item in pair[1]),
            )
            values = sorted(values, key=lambda item: float(item["window_center_us"]))
            symbol = "+" if key[2] == "sum" else "-"
            selected[(channel, component)] = {
                "time_us": float_array(values, "window_center_us"),
                "bicoherence_squared": float_array(values, "bicoherence_squared"),
                "phase_locking": float_array(values, "phase_locking"),
                "triad": f"{key[3]}{symbol}{key[4]}->{key[5]}",
            }
    return selected


def selected_encoder_rom(rows: list[dict[str, str]]) -> list[dict]:
    selected = []
    for row in rows:
        if row["representation"] != "encoder" or row["selected_by_validation"].lower() != "true":
            continue
        fit_end = float(row["fit_end_us"])
        test_end = float(row["test_end_us"])
        selected.append(
            {
                "window": row["window"],
                "fit_start_us": float(row["fit_start_us"]),
                "fit_end_us": fit_end,
                "test_end_us": test_end,
                "holdout_center_us": 0.5 * (fit_end + test_end),
                "method": row["method"],
                "delay": int(row["delay"]),
                "rank": int(row["rank"]),
                "standardized_rmse": float(row["standardized_rmse"]),
                "skill_vs_persistence": float(row["skill_vs_persistence"]),
                "skill_vs_training_mean": float(row["skill_vs_training_mean"]),
                "correlation": float(row["correlation"]),
                "contiguous_horizon_rmse_lt_1_us": float(row["contiguous_horizon_rmse_lt_1_us"]),
            }
        )
    return sorted(selected, key=lambda row: row["fit_end_us"])


def prepare_physics(rows: list[dict[str, str]], smooth_us: float) -> dict[str, np.ndarray]:
    time = float_array(rows, "time_us")
    dt = float(np.median(np.diff(time)))
    frames = max(1, int(round(smooth_us / dt)))
    result: dict[str, np.ndarray] = {"time_us": time}
    for component in COMPONENTS:
        result[f"{component}_power_fraction"] = smooth(float_array(rows, f"{component}_power_fraction"), frames)
        result[f"{component}_exb_transport"] = smooth(float_array(rows, f"{component}_exb_transport"), frames)
        result[f"{component}_cross_phase_deg"] = circular_smooth_deg(
            float_array(rows, f"{component}_cross_phase_deg"), frames
        )
        result[f"{component}_cross_coherence"] = smooth(
            float_array(rows, f"{component}_cross_coherence"), frames
        )
    fractions = np.stack([result[f"{component}_power_fraction"] for component in COMPONENTS], axis=1)
    normalized = fractions / np.maximum(np.sum(fractions, axis=1, keepdims=True), 1.0e-30)
    result["selected_mode_entropy"] = -np.sum(
        np.where(normalized > 0.0, normalized * np.log(normalized), 0.0), axis=1
    ) / math.log(len(COMPONENTS))
    return result


def mean_over(time: np.ndarray, values: np.ndarray, start: float, stop: float) -> float:
    mask = (time >= start - 1.0e-9) & (time <= stop + 1.0e-9)
    return float(np.mean(values[mask]))


def std_over(time: np.ndarray, values: np.ndarray, start: float, stop: float) -> float:
    mask = (time >= start - 1.0e-9) & (time <= stop + 1.0e-9)
    return float(np.std(values[mask]))


def circular_mean_over(time: np.ndarray, values: np.ndarray, start: float, stop: float) -> float:
    mask = (time >= start - 1.0e-9) & (time <= stop + 1.0e-9)
    return circular_mean_deg(values[mask])


def circular_dispersion_over(time: np.ndarray, values: np.ndarray, start: float, stop: float) -> float:
    mask = (time >= start - 1.0e-9) & (time <= stop + 1.0e-9)
    radians = np.radians(values[mask])
    return float(1.0 - abs(np.mean(np.exp(1j * radians))))


def build_window_summary(
    rom: list[dict], physics: dict[str, np.ndarray], bico: dict[tuple[str, str], dict]
) -> list[dict]:
    time = physics["time_us"]
    rows = []
    for item in rom:
        start = item["fit_end_us"]
        stop = item["test_end_us"]
        row = dict(item)
        for component in COMPONENTS:
            fraction = physics[f"{component}_power_fraction"]
            transport = physics[f"{component}_exb_transport"]
            phase = physics[f"{component}_cross_phase_deg"]
            coherence = physics[f"{component}_cross_coherence"]
            row[f"{component}_power_fraction_mean"] = mean_over(time, fraction, start, stop)
            row[f"{component}_power_fraction_std"] = std_over(time, fraction, start, stop)
            row[f"{component}_exb_transport_mean"] = mean_over(time, transport, start, stop)
            row[f"{component}_abs_exb_transport_mean"] = mean_over(time, np.abs(transport), start, stop)
            row[f"{component}_exb_transport_std"] = std_over(time, transport, start, stop)
            row[f"{component}_cross_phase_circular_mean_deg"] = circular_mean_over(time, phase, start, stop)
            row[f"{component}_cross_phase_circular_dispersion"] = circular_dispersion_over(
                time, phase, start, stop
            )
            row[f"{component}_cross_coherence_mean"] = mean_over(time, coherence, start, stop)
            row[f"{component}_cross_coherence_std"] = std_over(time, coherence, start, stop)
            for channel in ("field", "density"):
                series = bico[(channel, component)]
                mask = (series["time_us"] >= start - 1.0e-9) & (series["time_us"] <= stop + 1.0e-9)
                row[f"{channel}_{component}_bicoherence_mean"] = float(
                    np.mean(series["bicoherence_squared"][mask])
                )
                row[f"{channel}_{component}_bicoherence_max"] = float(
                    np.max(series["bicoherence_squared"][mask])
                )
                row[f"{channel}_{component}_bicoherence_std"] = float(
                    np.std(series["bicoherence_squared"][mask])
                )
                row[f"{channel}_{component}_bicoherence_triad"] = series["triad"]
        row["ecdi_to_mtsi_power_ratio"] = row["ecdi_power_fraction_mean"] / max(
            row["mtsi_power_fraction_mean"], 1.0e-30
        )
        row["selected_mode_entropy_mean"] = mean_over(
            time, physics["selected_mode_entropy"], start, stop
        )
        row["power_partition_variability"] = float(
            np.sqrt(sum(row[f"{component}_power_fraction_std"] ** 2 for component in COMPONENTS))
        )
        row["transport_relative_variability"] = float(
            sum(row[f"{component}_exb_transport_std"] for component in COMPONENTS)
            / max(
                sum(abs(row[f"{component}_exb_transport_mean"]) for component in COMPONENTS),
                1.0e-30,
            )
        )
        row["cross_phase_dispersion_mean"] = float(
            np.mean(
                [row[f"{component}_cross_phase_circular_dispersion"] for component in COMPONENTS]
            )
        )
        row["bicoherence_variability_mean"] = float(
            np.mean(
                [
                    row[f"{channel}_{component}_bicoherence_std"]
                    for channel in ("field", "density")
                    for component in COMPONENTS
                ]
            )
        )
        row["log10_standardized_rmse"] = math.log10(max(row["standardized_rmse"], 1.0e-30))
        row["prospective_partial_closure"] = int(
            row["skill_vs_persistence"] > 0.0
            and row["skill_vs_training_mean"] > 0.0
            and row["correlation"] > 0.0
        )
        rows.append(row)
    return rows


def rank_correlations(rows: list[dict]) -> list[dict]:
    outcomes = (
        "skill_vs_persistence",
        "skill_vs_training_mean",
        "correlation",
        "log10_standardized_rmse",
    )
    features = []
    for component in COMPONENTS:
        features.extend(
            (
                f"{component}_power_fraction_mean",
                f"{component}_abs_exb_transport_mean",
                f"{component}_cross_coherence_mean",
                f"field_{component}_bicoherence_mean",
                f"density_{component}_bicoherence_mean",
            )
        )
    features.extend(
        (
            "ecdi_to_mtsi_power_ratio",
            "selected_mode_entropy_mean",
            "power_partition_variability",
            "transport_relative_variability",
            "cross_phase_dispersion_mean",
            "bicoherence_variability_mean",
        )
    )
    result = []
    for scope, selected in (("all_6_windows", rows), ("exclude_first_catastrophic", rows[1:])):
        for outcome in outcomes:
            y = np.asarray([float(row[outcome]) for row in selected], dtype=np.float64)
            for feature in features:
                x = np.asarray([float(row[feature]) for row in selected], dtype=np.float64)
                rho, p_value = spearmanr(x, y)
                result.append(
                    {
                        "scope": scope,
                        "n_windows": len(selected),
                        "outcome": outcome,
                        "physics_feature": feature,
                        "spearman_rho": float(rho),
                        "p_value_exploratory": float(p_value),
                    }
                )
    for outcome in outcomes:
        y = np.diff(np.asarray([float(row[outcome]) for row in rows], dtype=np.float64))
        for feature in features:
            x = np.diff(np.asarray([float(row[feature]) for row in rows], dtype=np.float64))
            rho, p_value = spearmanr(x, y)
            result.append(
                {
                    "scope": "adjacent_window_changes",
                    "n_windows": len(y),
                    "outcome": outcome,
                    "physics_feature": feature,
                    "spearman_rho": float(rho),
                    "p_value_exploratory": float(p_value),
                }
            )
    return result


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.framealpha": 0.92,
        }
    )


def add_window_centers(axes: list[plt.Axes], rom: list[dict]) -> None:
    for axis in axes:
        for index, item in enumerate(rom, start=1):
            axis.axvline(item["holdout_center_us"], color="#777777", lw=0.6, ls=":", alpha=0.42)
    top = axes[0]
    y_max = top.get_ylim()[1]
    for index, item in enumerate(rom, start=1):
        top.text(item["holdout_center_us"], y_max, f"W{index}", ha="center", va="bottom", fontsize=7.5)


def plot_overlay(
    output: Path,
    physics: dict[str, np.ndarray],
    bico: dict[tuple[str, str], dict],
    rom: list[dict],
    smooth_us: float,
) -> None:
    time = physics["time_us"]
    mask = (time >= TIME_MIN_US) & (time <= TIME_MAX_US)
    fig, axes = plt.subplots(5, 1, figsize=(14.5, 15.5), sharex=True)

    for component in COMPONENTS:
        axes[0].plot(
            time[mask],
            physics[f"{component}_power_fraction"][mask],
            color=COLORS[component],
            label=LABELS[component],
            lw=1.55,
        )
    axes[0].set_ylabel("Ey power fraction")
    axes[0].set_title(f"Mode strength ({smooth_us:.2f} us smoothing)")
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(loc="upper right", ncol=3)

    for component in COMPONENTS:
        axes[1].plot(
            time[mask],
            physics[f"{component}_exb_transport"][mask] / 1.0e19,
            color=COLORS[component],
            label=LABELS[component],
            lw=1.45,
        )
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_ylabel(r"$-\mathrm{Re}[n_e E_y^*]/B_x$ [$10^{19}$]")
    axes[1].set_title("Mode-resolved E-cross-B transport proxy")
    axes[1].legend(loc="upper right", ncol=3)

    for component in COMPONENTS:
        axes[2].plot(
            time[mask],
            physics[f"{component}_cross_phase_deg"][mask],
            color=COLORS[component],
            label=LABELS[component],
            lw=1.35,
        )
    axes[2].set_ylabel("density-Ey phase [deg]")
    phase_values = np.concatenate(
        [physics[f"{component}_cross_phase_deg"][mask] for component in COMPONENTS]
    )
    phase_low = max(-180.0, 10.0 * math.floor((float(np.min(phase_values)) - 5.0) / 10.0))
    phase_high = min(180.0, 10.0 * math.ceil((float(np.max(phase_values)) + 5.0) / 10.0))
    axes[2].set_ylim(phase_low, phase_high)
    axes[2].set_title("Band-aggregated density-Ey cross-phase")
    axes[2].legend(loc="upper right", ncol=3)

    for component in COMPONENTS:
        field = bico[("field", component)]
        density = bico[("density", component)]
        field_mask = (field["time_us"] >= TIME_MIN_US) & (field["time_us"] <= TIME_MAX_US)
        density_mask = (density["time_us"] >= TIME_MIN_US) & (density["time_us"] <= TIME_MAX_US)
        axes[3].plot(
            field["time_us"][field_mask],
            field["bicoherence_squared"][field_mask],
            color=COLORS[component],
            lw=1.55,
            label=f"{LABELS[component]} field ({field['triad']})",
        )
        axes[3].plot(
            density["time_us"][density_mask],
            density["bicoherence_squared"][density_mask],
            color=COLORS[component],
            lw=1.0,
            ls="--",
            alpha=0.72,
            label=f"{LABELS[component]} density ({density['triad']})",
        )
    axes[3].set_ylabel(r"bicoherence $b^2$")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].set_title("Selected spatial triads (1.5 us rolling window, 0.3 us step)")
    axes[3].legend(loc="upper right", ncol=2, fontsize=7.4)

    centers = np.asarray([row["holdout_center_us"] for row in rom])
    skill_p = np.asarray([row["skill_vs_persistence"] for row in rom])
    skill_m = np.asarray([row["skill_vs_training_mean"] for row in rom])
    correlation = np.asarray([row["correlation"] for row in rom])
    clip_min, clip_max = -1.2, 1.0
    axes[4].plot(centers, np.clip(skill_p, clip_min, clip_max), "o-", color="#111111", lw=1.4, label="Skill vs persistence")
    axes[4].plot(centers, np.clip(skill_m, clip_min, clip_max), "s-", color="#d17c00", lw=1.35, label="Skill vs training mean")
    axes[4].plot(centers, correlation, "^-", color="#7048a8", lw=1.35, label="Trajectory correlation")
    axes[4].axhline(0.0, color="black", lw=0.8)
    axes[4].set_ylim(clip_min, clip_max)
    axes[4].set_ylabel("Encoder closure indicator")
    axes[4].set_title("Validation-selected encoder ROM; each marker summarizes the following 6 us holdout")
    axes[4].legend(loc="lower right", ncol=3, fontsize=8.2)
    for index, (x, raw) in enumerate(zip(centers, skill_p), start=1):
        if raw < clip_min:
            axes[4].annotate(
                f"W{index}: {raw:.2g}",
                xy=(x, clip_min),
                xytext=(x + 0.25, clip_min + 0.18),
                fontsize=7.5,
                arrowprops={"arrowstyle": "->", "lw": 0.7},
            )

    add_window_centers(list(axes), rom)
    axes[-1].set_xlim(TIME_MIN_US, TIME_MAX_US)
    axes[-1].set_xlabel("PIC time [us]")
    fig.suptitle("B25: rolling ROM closure and physical mode diagnostics on a common time axis", y=0.997, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.987))
    fig.savefig(output / "b25_closure_physics_same_time_axis.png", dpi=190)
    plt.close(fig)


def plot_window_summary(output: Path, rows: list[dict]) -> None:
    centers = np.asarray([row["holdout_center_us"] for row in rows])
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 12.5), sharex=True)
    for component in COMPONENTS:
        axes[0].plot(
            centers,
            [row[f"{component}_power_fraction_mean"] for row in rows],
            "o-",
            color=COLORS[component],
            label=LABELS[component],
        )
        axes[1].plot(
            centers,
            np.asarray([row[f"{component}_exb_transport_mean"] for row in rows]) / 1.0e19,
            "o-",
            color=COLORS[component],
            label=LABELS[component],
        )
        axes[2].plot(
            centers,
            [row[f"field_{component}_bicoherence_mean"] for row in rows],
            "o-",
            color=COLORS[component],
            label=f"{LABELS[component]} field",
        )
        axes[2].plot(
            centers,
            [row[f"density_{component}_bicoherence_mean"] for row in rows],
            "s--",
            color=COLORS[component],
            alpha=0.7,
            label=f"{LABELS[component]} density",
        )
    axes[0].set_ylabel("Holdout mean\nEy power fraction")
    axes[1].set_ylabel("Holdout mean ExB proxy\n[$10^{19}$]")
    axes[2].set_ylabel(r"Holdout mean $b^2$")
    axes[0].legend(loc="upper right", ncol=3)
    axes[1].legend(loc="upper right", ncol=3)
    axes[2].legend(loc="upper right", ncol=2, fontsize=8)
    axes[1].axhline(0.0, color="black", lw=0.7)

    axes[3].plot(centers, np.clip([row["skill_vs_persistence"] for row in rows], -1.2, 1.0), "o-", color="#111111", label="Skill vs persistence")
    axes[3].plot(centers, np.clip([row["skill_vs_training_mean"] for row in rows], -1.2, 1.0), "s-", color="#d17c00", label="Skill vs mean")
    axes[3].plot(centers, [row["correlation"] for row in rows], "^-", color="#7048a8", label="Correlation")
    axes[3].axhline(0.0, color="black", lw=0.7)
    axes[3].set_ylim(-1.2, 1.0)
    axes[3].set_ylabel("Encoder closure")
    axes[3].set_xlabel("Center of 6 us autonomous holdout [us]")
    axes[3].legend(loc="lower right", ncol=3, fontsize=8)
    fig.suptitle("B25: holdout-averaged physics and rolling closure", y=0.996, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(output / "b25_closure_physics_holdout_means.png", dpi=190)
    plt.close(fig)


def plot_correlations(output: Path, rows: list[dict]) -> None:
    outcomes = ("skill_vs_persistence", "skill_vs_training_mean", "correlation", "log10_standardized_rmse")
    outcome_labels = ("Skill vs copy", "Skill vs mean", "Trajectory corr.", "log10 RMSE")
    features = []
    labels = []
    for component in COMPONENTS:
        short = LABELS[component]
        features.extend(
            (
                f"{component}_power_fraction_mean",
                f"{component}_abs_exb_transport_mean",
                f"{component}_cross_coherence_mean",
                f"field_{component}_bicoherence_mean",
                f"density_{component}_bicoherence_mean",
            )
        )
        labels.extend((f"{short} power", f"{short} |transport|", f"{short} coherence", f"{short} field b2", f"{short} density b2"))
    features.extend(
        (
            "ecdi_to_mtsi_power_ratio",
            "selected_mode_entropy_mean",
            "power_partition_variability",
            "transport_relative_variability",
            "cross_phase_dispersion_mean",
            "bicoherence_variability_mean",
        )
    )
    labels.extend(
        (
            "ECDI/MTSI power",
            "Mode-mix entropy",
            "Power variability",
            "Transport variability",
            "Phase dispersion",
            "Bicoherence variability",
        )
    )
    matrix = np.empty((len(outcomes), len(features)), dtype=np.float64)
    for row_index, outcome in enumerate(outcomes):
        y = np.asarray([row[outcome] for row in rows], dtype=np.float64)
        for column, feature in enumerate(features):
            x = np.asarray([row[feature] for row in rows], dtype=np.float64)
            matrix[row_index, column] = spearmanr(x, y).statistic

    fig, axis = plt.subplots(figsize=(15.5, 4.8))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(features)), labels, rotation=48, ha="right")
    axis.set_yticks(np.arange(len(outcomes)), outcome_labels)
    for row_index in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row_index, f"{matrix[row_index, column]:.2f}", ha="center", va="center", fontsize=7.2)
    axis.set_title("Exploratory Spearman correlations across six overlapping holdouts")
    colorbar = fig.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Spearman rho")
    fig.tight_layout()
    fig.savefig(output / "b25_closure_physics_spearman.png", dpi=190)
    plt.close(fig)


def framewise_rows(physics: dict[str, np.ndarray], bico: dict[tuple[str, str], dict]) -> list[dict]:
    time = physics["time_us"]
    mask = (time >= TIME_MIN_US) & (time <= TIME_MAX_US)
    rows = []
    for index in np.flatnonzero(mask):
        row = {"time_us": float(time[index])}
        for component in COMPONENTS:
            for key in ("power_fraction", "exb_transport", "cross_phase_deg", "cross_coherence"):
                row[f"{component}_{key}"] = float(physics[f"{component}_{key}"][index])
            for channel in ("field", "density"):
                series = bico[(channel, component)]
                row[f"{channel}_{component}_bicoherence_interpolated"] = float(
                    np.interp(time[index], series["time_us"], series["bicoherence_squared"])
                )
        row["selected_mode_entropy"] = float(physics["selected_mode_entropy"][index])
        rows.append(row)
    return rows


def write_readme(output: Path, rows: list[dict], correlations: list[dict], bico: dict) -> None:
    best = sorted(
        [row for row in correlations if row["scope"] == "all_6_windows" and row["outcome"] == "correlation"],
        key=lambda row: abs(row["spearman_rho"]),
        reverse=True,
    )[:5]
    best_changes = sorted(
        [
            row
            for row in correlations
            if row["scope"] == "adjacent_window_changes" and row["outcome"] == "correlation"
        ],
        key=lambda row: abs(row["spearman_rho"]),
        reverse=True,
    )[:5]
    lines = [
        "# B25 rolling closure versus physical diagnostics",
        "",
        "This analysis puts the validation-selected encoder ROM closure metrics on the same",
        "20--40 us time axis as the existing magnetic-sweep diagnostics.",
        "",
        "## Definitions",
        "",
        "- Long wavelength: azimuthal n=1--4 and |radial mode|=0--1.",
        "- MTSI candidate: n=5--13 and |radial mode|=1.",
        "- ECDI candidate: n=14--34 and |radial mode|=0--1.",
        "- Modal transport: -Re[Ne(k) Ey(k)*]/Bx.",
        "- Cross-phase: phase of the density--Ey cross-spectrum aggregated in each mask.",
        "- Bicoherence: the same selected spatial triads used by the original B25/B30 analysis.",
        "- Closure: encoder ROM selected only on the last 1 us of each fit window, then",
        "  evaluated autonomously on the following 6 us.",
        "",
        "## Rolling closure table",
        "",
        "| fit -> holdout [us] | skill vs copy | skill vs mean | corr | log10 RMSE | partial closure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fit_start_us']:.0f}-{row['fit_end_us']:.0f} -> {row['fit_end_us']:.0f}-{row['test_end_us']:.0f} "
            f"| {row['skill_vs_persistence']:.4g} | {row['skill_vs_training_mean']:.4g} "
            f"| {row['correlation']:.4f} | {row['log10_standardized_rmse']:.3f} "
            f"| {row['prospective_partial_closure']} |"
        )
    lines.extend(
        [
            "",
            "`partial closure=1` requires positive skill against both persistence and the",
            "training mean, plus positive trajectory correlation. It is a prospective diagnostic",
            "rule, not a claim of complete nonlinear closure.",
            "",
            "## Physical interpretation",
            "",
            "- The catastrophic 24--30 us holdout crosses the strongest reorganization in the",
            "  plotted interval. Its power-partition variability is 0.112, relative transport",
            "  variability is 0.548, and long-wave transport has a large burst near 24--26 us.",
            "- By the 30--36 us holdout, power-partition variability falls to 0.030, relative",
            "  transport variability to 0.142, and mean circular phase dispersion to 0.0010.",
            "  Trajectory correlation reaches 0.530, but both amplitude-sensitive skills remain",
            "  negative. The phase geometry becomes predictable before the rollout amplitude.",
            "- The 32--38 and 34--40 us holdouts satisfy all three partial-closure signs. MTSI",
            "  power fraction rises to 0.213--0.221 while long-wave power falls to 0.050--0.038.",
            "  ECDI still has the largest power fraction (0.360--0.343), so closure recovery is",
            "  not caused by ECDI disappearing. It coincides with a steadier, phase-locked",
            "  coexistence and increasingly MTSI-carried transport.",
            "- Density ECDI bicoherence averages about 0.87 in the last three holdouts, whereas",
            "  bicoherence variability is much lower than in the early failing windows. This is",
            "  consistent with a stable coupling network, but does not establish causality or",
            "  energy-transfer direction.",
            "",
            "## Selected bicoherence triads",
            "",
        ]
    )
    for channel in ("field", "density"):
        for component in COMPONENTS:
            lines.append(f"- {channel} {component}: `{bico[(channel, component)]['triad']}`")
    lines.extend(
        [
            "",
            "## Strongest exploratory associations with trajectory correlation",
            "",
        ]
    )
    for item in best:
        lines.append(
            f"- `{item['physics_feature']}`: Spearman rho={item['spearman_rho']:.3f} "
            f"(n={item['n_windows']}, exploratory p={item['p_value_exploratory']:.3g})"
        )
    lines.extend(
        [
            "",
            "## Adjacent-window change associations",
            "",
            "These compare changes from one overlapping holdout to the next, reducing (but not",
            "eliminating) the shared slow time trend.",
            "",
        ]
    )
    for item in best_changes:
        lines.append(
            f"- `delta {item['physics_feature']}`: Spearman rho={item['spearman_rho']:.3f} "
            f"(n={item['n_windows']}, exploratory p={item['p_value_exploratory']:.3g})"
        )
    lines.extend(
        [
            "",
            "The six holdouts overlap strongly, so these correlations are descriptive clues,",
            "not independent-sample significance tests. The first closure failure is catastrophic;",
            "the plotting range is clipped but the CSV and annotation retain the raw value.",
            "Bicoherence establishes quadratic phase locking, not transfer direction.",
            "",
            "## Files",
            "",
            "- `b25_closure_physics_same_time_axis.png`",
            "- `b25_closure_physics_holdout_means.png`",
            "- `b25_closure_physics_spearman.png`",
            "- `b25_closure_physics_window_summary.csv`",
            "- `b25_closure_physics_framewise.csv`",
            "- `b25_closure_physics_correlations.csv`",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    physics_rows = read_csv(args.physics_dir / "three_component_time_series.csv")
    bico_rows = read_csv(args.physics_dir / "bicoherence_selected_time_series.csv")
    rom_rows = read_csv(args.rom_dir / "rolling_rom_metrics.csv")

    physics = prepare_physics(physics_rows, args.smooth_us)
    bico = selected_bicoherence(bico_rows)
    rom = selected_encoder_rom(rom_rows)
    if len(rom) != 6:
        raise RuntimeError(f"Expected six validation-selected encoder windows, found {len(rom)}")
    summary = build_window_summary(rom, physics, bico)
    correlations = rank_correlations(summary)

    write_csv(args.output_dir / "b25_closure_physics_window_summary.csv", summary)
    write_csv(args.output_dir / "b25_closure_physics_framewise.csv", framewise_rows(physics, bico))
    write_csv(args.output_dir / "b25_closure_physics_correlations.csv", correlations)
    set_style()
    plot_overlay(args.output_dir, physics, bico, rom, args.smooth_us)
    plot_window_summary(args.output_dir, summary)
    plot_correlations(args.output_dir, summary)
    write_readme(args.output_dir, summary, correlations, bico)

    print(f"PASS: wrote B25 closure/physics overlay to {args.output_dir}")
    print(f"windows={len(summary)} framewise_rows={len(framewise_rows(physics, bico))}")


if __name__ == "__main__":
    main()
