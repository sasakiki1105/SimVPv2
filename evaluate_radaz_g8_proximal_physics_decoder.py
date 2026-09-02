#!/usr/bin/env python3
"""Evaluate a truth-free proximal Poisson decoder after the E25 G8 SimVP.

The decoder keeps the predicted electron/ion densities fixed and corrects only
the potential.  For each frame it solves the zero-radial-boundary correction

    Laplacian(delta_phi) = -[Laplacian(phi_pred) + e(ni_pred-ne_pred)/eps0]

with periodic azimuthal boundaries.  A per-frame scalar limits the correction
RMS relative to the spatial RMS fluctuation of the original predicted
potential.  Thus the operational decoder and its strength are selected without
access to PIC truth.  Truth is used only afterwards to measure reconstruction,
field, modal, transport, and saturated-stability effects.

G denotes spatial grid coarsening, not temporal frame stride.
"""

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
import torch
from scipy.linalg import solve_banded

from analyze_radaz_g2_stability_reconstruction import (
    CHANNELS,
    DISPLAY_NAMES,
    E_CHARGE,
    EPS0,
    TINY,
    field_metrics,
    json_safe,
    modal_analysis,
    modal_transport_analysis,
    periodic_physics_metrics,
    predict_unique_frames,
)
from train_radaz_g2_residual_superresolution import (
    DEFAULT_H5,
    make_coarse_size_interpolated,
    make_grid_interpolated,
    segment_starts,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKDIR = (
    ROOT / "workdirs" / "radaz_e25_g8_simvp_residual_sr_sync10_20to30us"
)
DEFAULT_OUTPUT = ROOT / "workdirs" / "radaz_e25_g8_proximal_physics_decoder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-us", type=float, default=20.0)
    parser.add_argument("--end-us", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--budgets",
        default="0.001,0.003,0.01,0.03,0.1",
        help="Comma-separated per-frame correction RMS budgets relative to phi spatial RMS.",
    )
    parser.add_argument(
        "--selected-budget",
        type=float,
        default=0.01,
        help="Predeclared truth-free operational budget (default: 1%%).",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])


def physical_fields(
    values: np.ndarray, norm_low: np.ndarray, norm_high: np.ndarray
) -> np.ndarray:
    return (
        values.astype(np.float64)
        * (norm_high - norm_low)[None, :, None, None]
        + norm_low[None, :, None, None]
    )


def poisson_residual(
    ne: np.ndarray, ni: np.ndarray, phi: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    radial_second = (
        phi[:, 2:, :] - 2.0 * phi[:, 1:-1, :] + phi[:, :-2, :]
    ) / dx**2
    azimuth_second = (
        np.roll(phi[:, 1:-1, :], -1, axis=-1)
        - 2.0 * phi[:, 1:-1, :]
        + np.roll(phi[:, 1:-1, :], 1, axis=-1)
    ) / dy**2
    source = E_CHARGE * (ni[:, 1:-1, :] - ne[:, 1:-1, :]) / EPS0
    return radial_second + azimuth_second + source, source


def solve_periodic_poisson_correction(
    residual: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """Solve Laplacian(delta)=-residual with radial Dirichlet/azimuth periodic BCs."""
    frame_count, radial_interior, azimuth_count = residual.shape
    residual_modes = np.fft.rfft(residual, axis=-1)
    correction_modes = np.empty_like(residual_modes)
    off_diagonal = 1.0 / dx**2
    for mode in range(residual_modes.shape[-1]):
        azimuth_eigenvalue = (
            -4.0 * math.sin(math.pi * mode / azimuth_count) ** 2 / dy**2
        )
        banded = np.zeros((3, radial_interior), dtype=np.float64)
        banded[0, 1:] = off_diagonal
        banded[1, :] = -2.0 / dx**2 + azimuth_eigenvalue
        banded[2, :-1] = off_diagonal
        rhs = -residual_modes[:, :, mode].T
        correction_modes[:, :, mode] = solve_banded(
            (1, 1), banded, rhs, check_finite=False
        ).T
    correction = np.zeros(
        (frame_count, radial_interior + 2, azimuth_count), dtype=np.float64
    )
    correction[:, 1:-1, :] = np.fft.irfft(
        correction_modes, n=azimuth_count, axis=-1
    )
    return correction


def scalar_stats(values: np.ndarray) -> dict[str, float]:
    values64 = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values64)),
        "median": float(np.median(values64)),
        "std": float(np.std(values64)),
        "min": float(np.min(values64)),
        "max": float(np.max(values64)),
    }


def electric_field_truth_metrics(candidate: dict, truth: dict) -> dict[str, float]:
    ex_error = candidate["ex"] - truth["ex"]
    ey_error = candidate["ey"] - truth["ey"]
    truth_power_sum = float(np.sum(truth["ex"] ** 2 + truth["ey"] ** 2))
    return {
        "relative_l2": math.sqrt(
            float(np.sum(ex_error**2 + ey_error**2)) / max(truth_power_sum, TINY)
        ),
        "energy_ratio": float(
            np.mean(candidate["ex"] ** 2 + candidate["ey"] ** 2)
            / max(float(np.mean(truth["ex"] ** 2 + truth["ey"] ** 2)), TINY)
        ),
    }


def electric_field_displacement(candidate: dict, raw: dict) -> float:
    numerator = float(
        np.sum(
            (candidate["ex"] - raw["ex"]) ** 2
            + (candidate["ey"] - raw["ey"]) ** 2
        )
    )
    denominator = float(np.sum(raw["ex"] ** 2 + raw["ey"] ** 2))
    return math.sqrt(numerator / max(denominator, TINY))


def candidate_from_weight(
    prediction: np.ndarray,
    phi_physical: np.ndarray,
    delta_phi: np.ndarray,
    weight: np.ndarray,
    norm_low: np.ndarray,
    norm_high: np.ndarray,
) -> np.ndarray:
    candidate = prediction.copy()
    corrected_phi = phi_physical + weight[:, None, None] * delta_phi
    candidate[:, 2] = (
        (corrected_phi - norm_low[2]) / (norm_high[2] - norm_low[2])
    ).astype(np.float32)
    return candidate


def compact_phi_modal_metrics(
    truth: np.ndarray, candidate: np.ndarray
) -> dict[str, object]:
    truth_power = np.mean(
        np.abs(np.fft.rfft(truth[:, 2], axis=-1, norm="forward")) ** 2,
        axis=1,
    )
    candidate_power = np.mean(
        np.abs(np.fft.rfft(candidate[:, 2], axis=-1, norm="forward")) ** 2,
        axis=1,
    )
    bands = {
        "mtsi_n1_6": slice(1, 7),
        "transition_n7_8": slice(7, 9),
        "ecdi_n9_21": slice(9, 22),
    }
    output: dict[str, object] = {
        "truth_dominant_mode": int(np.argmax(np.mean(truth_power[:, 1:22], axis=0)) + 1),
        "candidate_dominant_mode": int(
            np.argmax(np.mean(candidate_power[:, 1:22], axis=0)) + 1
        ),
    }
    truth_dominant_time = np.argmax(truth_power[:, 1:22], axis=1) + 1
    candidate_dominant_time = np.argmax(candidate_power[:, 1:22], axis=1) + 1
    output["dominant_mode_time_agreement"] = float(
        np.mean(truth_dominant_time == candidate_dominant_time)
    )
    for name, mode_slice in bands.items():
        truth_series = np.sum(truth_power[:, mode_slice], axis=1, dtype=np.float64)
        candidate_series = np.sum(
            candidate_power[:, mode_slice], axis=1, dtype=np.float64
        )
        output[f"{name}_mean_power_ratio"] = float(np.mean(candidate_series)) / max(
            float(np.mean(truth_series)), TINY
        )
    truth_ratio = float(np.mean(np.sum(truth_power[:, 9:22], axis=1))) / max(
        float(np.mean(np.sum(truth_power[:, 1:7], axis=1))), TINY
    )
    candidate_ratio = float(
        np.mean(np.sum(candidate_power[:, 9:22], axis=1))
    ) / max(float(np.mean(np.sum(candidate_power[:, 1:7], axis=1))), TINY)
    output["ecdi_over_mtsi_ratio_to_truth"] = candidate_ratio / max(truth_ratio, TINY)
    return output


def plot_tradeoff(path: Path, rows: list[dict]) -> None:
    finite_rows = [row for row in rows if row["label"] != "exact_poisson"]
    x = np.asarray(
        [100.0 * row["correction_ratio_phi_fluctuation_median"] for row in finite_rows]
    )
    poisson = np.asarray(
        [row["poisson_residual_median_ratio_to_raw"] for row in finite_rows]
    )
    phi_error = np.asarray([row["phi_relative_l2"] for row in finite_rows])
    e_error = np.asarray([row["electric_field_relative_l2"] for row in finite_rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].semilogy(x, poisson, "o-")
    axes[0].set_xlabel("median correction / predicted phi fluctuation RMS [%]")
    axes[0].set_ylabel("median Poisson residual / raw SimVP")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, phi_error, "o-", label="phi relative L2")
    axes[1].plot(x, e_error, "s-", label="electric-field relative L2")
    axes[1].set_xlabel("median correction / predicted phi fluctuation RMS [%]")
    axes[1].set_ylabel("truth error (evaluation only)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_physics_time_series(
    path: Path,
    time_us: np.ndarray,
    truth_diag: dict,
    raw_diag: dict,
    selected_diag: dict,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for label, values, style in (
        ("PIC truth", truth_diag, "-"),
        ("raw G8-SimVP", raw_diag, "--"),
        ("1% proximal", selected_diag, "-."),
    ):
        axes[0].semilogy(
            time_us, values["relative_poisson_residual"], style, label=label
        )
        axes[1].plot(time_us, values["poisson_balance_corr"], style, label=label)
    axes[0].set_ylabel("relative Poisson residual")
    axes[1].set_ylabel("Poisson balance correlation")
    for axis in axes:
        axis.set_xlabel("time [us]")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_snapshot(
    path: Path,
    truth: np.ndarray,
    raw: np.ndarray,
    selected: np.ndarray,
    norm_low: np.ndarray,
    norm_high: np.ndarray,
    dx: float,
    dy: float,
    index: int,
    time_us: float,
) -> None:
    truth_phys = physical_fields(truth, norm_low, norm_high)
    raw_phys = physical_fields(raw, norm_low, norm_high)
    selected_phys = physical_fields(selected, norm_low, norm_high)
    raw_residual, raw_source = poisson_residual(
        raw_phys[:, 0], raw_phys[:, 1], raw_phys[:, 2], dx, dy
    )
    selected_residual, _ = poisson_residual(
        selected_phys[:, 0], selected_phys[:, 1], selected_phys[:, 2], dx, dy
    )
    source_scale = max(float(np.sqrt(np.mean(raw_source[index] ** 2))), TINY)
    panels = (
        (truth[index, 2], "PIC truth phi", None),
        (raw[index, 2], "raw G8-SimVP phi", None),
        (selected[index, 2], "1% proximal phi", None),
        (selected[index, 2] - raw[index, 2], "normalized phi correction", "RdBu_r"),
        (raw[index, 2] - truth[index, 2], "raw phi error", "RdBu_r"),
        (selected[index, 2] - truth[index, 2], "projected phi error", "RdBu_r"),
        (raw_residual[index] / source_scale, "raw residual / source RMS", "RdBu_r"),
        (
            selected_residual[index] / source_scale,
            "projected residual / source RMS",
            "RdBu_r",
        ),
    )
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for axis, (values, title, cmap) in zip(axes.ravel(), panels):
        kwargs = {"origin": "lower", "aspect": "auto"}
        if cmap is not None:
            limit = max(float(np.percentile(np.abs(values), 99)), TINY)
            kwargs.update({"cmap": cmap, "vmin": -limit, "vmax": limit})
        image = axis.imshow(values, **kwargs)
        axis.set_title(title)
        axis.set_xlabel("azimuth index")
        axis.set_ylabel("radial index")
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.suptitle(f"G8 proximal physics decoder at {time_us:.3f} us")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted({float(item) for item in args.budgets.split(",") if item.strip()})
    if args.selected_budget not in budgets:
        budgets.append(args.selected_budget)
        budgets.sort()
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    checkpoint_path = args.workdir / "checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    coarse_size = int(
        metadata.get("coarse_size", 256 // int(metadata.get("grid_factor", 8)))
    )
    grid_factor = float(
        metadata.get("effective_grid_factor", metadata.get("grid_factor", 8))
    )
    grid_label = str(metadata.get("grid_label", f"G{grid_factor:g}"))

    with h5py.File(args.h5, "r") as handle:
        times_s_all = np.asarray(handle["time_s"], dtype=np.float64)
        selected = np.flatnonzero(
            (times_s_all * 1.0e6 >= args.start_us - 1.0e-9)
            & (times_s_all * 1.0e6 <= args.end_us + 1.0e-9)
        )
        fine = np.asarray(
            handle["data_tchw"][int(selected[0]) : int(selected[-1]) + 1],
            dtype=np.float32,
        )
        times_s = times_s_all[selected]
        norm_low = np.asarray(handle["norm_low"], dtype=np.float64)
        norm_high = np.asarray(handle["norm_high"], dtype=np.float64)
        x_m = np.asarray(handle["x_m"], dtype=np.float64)
        y_m = np.asarray(handle["y_m"], dtype=np.float64)

    baseline_all = (
        make_grid_interpolated(fine, 256 // coarse_size)
        if 256 % coarse_size == 0
        else make_coarse_size_interpolated(fine, coarse_size)
    )
    frame_count = len(fine)
    val_end = int(math.floor(0.9 * frame_count))
    sequence_length = int(checkpoint["model_kwargs"]["in_shape"][0])
    hop = int(checkpoint["metadata"]["window_hop"])
    starts = segment_starts(val_end, frame_count, sequence_length, hop)
    prediction, truth, interpolation, local_indices, consistency = predict_unique_frames(
        fine, baseline_all, starts, checkpoint, device
    )
    evaluation_times_s = times_s[local_indices]
    evaluation_times_us = evaluation_times_s * 1.0e6
    print(
        f"[prediction] {len(starts)} windows -> {len(local_indices)} frames; "
        f"{evaluation_times_us[0]:.3f}--{evaluation_times_us[-1]:.3f} us on {device}",
        flush=True,
    )

    dx = float(np.median(np.diff(x_m)))
    dy = float(np.median(np.diff(y_m)))
    truth_phys = physical_fields(truth, norm_low, norm_high)
    raw_phys = physical_fields(prediction, norm_low, norm_high)
    ne_pred, ni_pred, phi_pred = raw_phys[:, 0], raw_phys[:, 1], raw_phys[:, 2]
    residual, _ = poisson_residual(ne_pred, ni_pred, phi_pred, dx, dy)
    delta_phi = solve_periodic_poisson_correction(residual, dx, dy)

    exact_phi = phi_pred + delta_phi
    exact_residual, _ = poisson_residual(ne_pred, ni_pred, exact_phi, dx, dy)
    residual_rms = np.sqrt(np.mean(residual**2, axis=(1, 2)))
    exact_residual_rms = np.sqrt(np.mean(exact_residual**2, axis=(1, 2)))
    solver_ratio = exact_residual_rms / np.maximum(residual_rms, TINY)
    phi_fluctuation_rms = np.sqrt(
        np.mean(
            (phi_pred - np.mean(phi_pred, axis=(1, 2), keepdims=True)) ** 2,
            axis=(1, 2),
        )
    )
    delta_rms = np.sqrt(np.mean(delta_phi**2, axis=(1, 2)))
    exact_correction_ratio = delta_rms / np.maximum(phi_fluctuation_rms, TINY)

    truth_diag = periodic_physics_metrics(truth, norm_low, norm_high, dx, dy)
    raw_diag = periodic_physics_metrics(prediction, norm_low, norm_high, dx, dy)
    raw_residual_median = float(np.median(raw_diag["relative_poisson_residual"]))
    candidate_specs: list[tuple[str, float | None, np.ndarray]] = [
        ("raw_simvp", 0.0, np.zeros(len(prediction), dtype=np.float64))
    ]
    for budget in budgets:
        weight = np.minimum(
            1.0, budget / np.maximum(exact_correction_ratio, TINY)
        )
        candidate_specs.append((f"budget_{100.0 * budget:g}pct", budget, weight))
    candidate_specs.append(
        ("exact_poisson", None, np.ones(len(prediction), dtype=np.float64))
    )

    sweep_rows: list[dict] = []
    candidates: dict[str, np.ndarray] = {}
    detailed_candidates: dict[str, dict] = {}
    for label, budget, weight in candidate_specs:
        candidate = candidate_from_weight(
            prediction, phi_pred, delta_phi, weight, norm_low, norm_high
        )
        candidates[label] = candidate
        candidate_diag = periodic_physics_metrics(
            candidate, norm_low, norm_high, dx, dy
        )
        phi_metrics = field_metrics(truth[:, 2], candidate[:, 2])
        e_metrics = electric_field_truth_metrics(candidate_diag, truth_diag)
        modal_metrics = compact_phi_modal_metrics(truth, candidate)
        correction_ratio = weight * exact_correction_ratio
        row = {
            "label": label,
            "target_budget": budget,
            "applied_exact_correction_weight_median": float(np.median(weight)),
            "applied_exact_correction_weight_min": float(np.min(weight)),
            "applied_exact_correction_weight_max": float(np.max(weight)),
            "correction_ratio_phi_fluctuation_mean": float(np.mean(correction_ratio)),
            "correction_ratio_phi_fluctuation_median": float(
                np.median(correction_ratio)
            ),
            "poisson_residual_mean": float(
                np.mean(candidate_diag["relative_poisson_residual"])
            ),
            "poisson_residual_median": float(
                np.median(candidate_diag["relative_poisson_residual"])
            ),
            "poisson_residual_median_ratio_to_raw": float(
                np.median(candidate_diag["relative_poisson_residual"])
                / max(raw_residual_median, TINY)
            ),
            "poisson_balance_corr_median": float(
                np.median(candidate_diag["poisson_balance_corr"])
            ),
            "electric_field_displacement_relative_l2_to_raw": electric_field_displacement(
                candidate_diag, raw_diag
            ),
            "phi_relative_l2": phi_metrics["relative_l2"],
            "phi_nrmse_std": phi_metrics["nrmse_std"],
            "phi_correlation": phi_metrics["correlation"],
            "electric_field_relative_l2": e_metrics["relative_l2"],
            "electric_field_energy_ratio": e_metrics["energy_ratio"],
            **modal_metrics,
        }
        sweep_rows.append(row)
        detailed_candidates[label] = {
            "truth_free": {
                "target_budget": budget,
                "applied_weight": scalar_stats(weight),
                "correction_ratio_to_predicted_phi_fluctuation": scalar_stats(
                    correction_ratio
                ),
                "relative_poisson_residual": scalar_stats(
                    candidate_diag["relative_poisson_residual"]
                ),
                "poisson_residual_median_ratio_to_raw": row[
                    "poisson_residual_median_ratio_to_raw"
                ],
                "poisson_balance_correlation": scalar_stats(
                    candidate_diag["poisson_balance_corr"]
                ),
                "electric_field_displacement_relative_l2_to_raw": row[
                    "electric_field_displacement_relative_l2_to_raw"
                ],
            },
            "truth_evaluation_only": {
                "phi": phi_metrics,
                "electric_field": e_metrics,
                "phi_modal": modal_metrics,
            },
        }
        print(
            f"[{label}] correction={100*np.median(correction_ratio):.4g}% "
            f"Poisson/raw={row['poisson_residual_median_ratio_to_raw']:.6g} "
            f"phi_L2={row['phi_relative_l2']:.6g} E_L2={row['electric_field_relative_l2']:.6g}",
            flush=True,
        )

    selected_label = f"budget_{100.0 * args.selected_budget:g}pct"
    selected_candidate = candidates[selected_label]
    selected_diag = periodic_physics_metrics(
        selected_candidate, norm_low, norm_high, dx, dy
    )
    _, selected_stability, selected_frequency, selected_powers = modal_analysis(
        truth, prediction, selected_candidate, evaluation_times_s
    )
    selected_transport, selected_transport_time = modal_transport_analysis(
        truth,
        prediction,
        selected_candidate,
        norm_low,
        norm_high,
        dy,
    )

    truth_phi_errors = {row["label"]: row["phi_relative_l2"] for row in sweep_rows}
    truth_e_errors = {
        row["label"]: row["electric_field_relative_l2"] for row in sweep_rows
    }
    summary = {
        "description": "Truth-free proximal Poisson decoder after E25 G8-SimVP",
        "grid_label": grid_label,
        "checkpoint": str(checkpoint_path.resolve()),
        "source_h5": str(args.h5.resolve()),
        "held_out_time_us": [
            float(evaluation_times_us[0]),
            float(evaluation_times_us[-1]),
        ],
        "held_out_unique_frames": int(len(prediction)),
        "decoder_definition": {
            "corrected_variable": "phi only; predicted ne and ni are fixed",
            "poisson_equation": "laplacian(phi)+e*(ni-ne)/eps0=0",
            "correction_boundary": "delta_phi=0 at radial edges; periodic azimuth",
            "constraint": "per-frame RMS(delta_phi_applied) <= budget * spatial_RMS(phi_pred fluctuation)",
            "selection_rule": (
                f"fixed {100.0 * args.selected_budget:g}% budget chosen before viewing truth"
            ),
            "field_gradient_note": (
                "G8-SimVP predicts ne, ni, phi only. Ex and Ey are derived from corrected phi, "
                "so no independent field-gradient residual exists in this model."
            ),
        },
        "truth_free_solver_checks": {
            "exact_correction_ratio_to_phi_fluctuation": scalar_stats(
                exact_correction_ratio
            ),
            "exact_poisson_residual_ratio_to_raw": scalar_stats(solver_ratio),
            "radial_boundary_max_abs_correction_V": float(
                np.max(np.abs(delta_phi[:, (0, -1), :]))
            ),
        },
        "overlap_consistency": consistency,
        "predeclared_selected_candidate": selected_label,
        "candidates": detailed_candidates,
        "selected_saturated_stability": selected_stability,
        "selected_phase_frequency": selected_frequency,
        "selected_modal_transport_proxy_ne_ey": selected_transport,
        "truth_evaluation_oracle_labels_not_used_for_selection": {
            "lowest_phi_relative_l2": min(truth_phi_errors, key=truth_phi_errors.get),
            "lowest_electric_field_relative_l2": min(
                truth_e_errors, key=truth_e_errors.get
            ),
        },
        "interpretation_scope": {
            "truth_free_claim": (
                "The decoder can reduce discrete Poisson residual under a predeclared "
                "small-displacement constraint without PIC truth."
            ),
            "truth_based_claim": (
                "Whether that consistency improves the PIC potential, electric field, "
                "modes, or transport is a separate held-out evaluation."
            ),
            "not_yet_tested": (
                "transfer to an independently run coarse PIC case or another electric-field condition"
            ),
        },
    }

    write_csv(args.output_dir / "proximal_decoder_sweep.csv", sweep_rows)
    (args.output_dir / "proximal_decoder_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_tradeoff(args.output_dir / "proximal_decoder_tradeoff.png", sweep_rows)
    plot_physics_time_series(
        args.output_dir / "proximal_decoder_physics_time_series.png",
        evaluation_times_us,
        truth_diag,
        raw_diag,
        selected_diag,
    )
    plot_snapshot(
        args.output_dir / "proximal_decoder_snapshot.png",
        truth,
        prediction,
        selected_candidate,
        norm_low,
        norm_high,
        dx,
        dy,
        len(prediction) // 2,
        float(evaluation_times_us[len(prediction) // 2]),
    )

    readme = f"""# E25 {grid_label} proximal physics decoder

This experiment applies a truth-free proximal Poisson correction after the
fixed G8-SimVP prediction on the held-out {evaluation_times_us[0]:.3f}--{evaluation_times_us[-1]:.3f} us interval.
Electron and ion densities are held fixed.  The potential correction preserves
the radial boundary values and is periodic azimuthally.  Candidate strengths
are capped by their per-frame correction RMS relative to the original predicted
potential fluctuation RMS.

The operational candidate is `{selected_label}` and was fixed before truth
evaluation.  `proximal_decoder_sweep.csv` deliberately places truth-free
physics/size diagnostics beside truth-only reconstruction diagnostics so that
physical consistency is not conflated with recovery of the PIC realization.

Because this G8 model has no independently predicted electric-field channel,
Ex and Ey are derived from phi.  The meaningful field check is therefore the
electric-field error against held-out PIC truth, not an independent
field-gradient identity loss.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), flush=True)
    print(f"[saved] {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
