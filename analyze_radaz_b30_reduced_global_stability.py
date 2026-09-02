#!/usr/bin/env python3
"""Reduced two-fluid global-stability screen for the B30 long wave.

This is deliberately a screening calculation, not a kinetic stability proof.
It linearizes an electrostatic, isothermal two-fluid model about azimuthally
and temporally averaged PIC profiles.  The radial direction is retained and
the azimuthal direction is represented by one Fourier mode at a time.

The calculation asks whether a low-frequency n=1 eigenvalue with the observed
frequency and radial structure exists robustly enough to justify treating a
physical/global linear mode as a live hypothesis before a domain-length PIC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy import linalg
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
CASE = "2D_RadAz_Xe1p_Bx30mT_Ez10kVm_dt15ps_out15ns"
SOURCE_H5 = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / CASE
    / CASE
    / "analysis_fields_uncompressed.h5"
)
BASE_OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / "analysis_results"
    / "B30_long_wavelength_origin_existing_data"
)
OUTPUT = BASE_OUTPUT / "reduced_global_stability_screen"

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
M_E = 9.1093837015e-31
M_I = 2.18e-25
B_X = 30.0e-3

PROFILE_WINDOWS = (
    ("growth", 0.75, 1.20),
    ("post_growth", 2.0, 5.0),
    ("steady", 20.0, 30.0),
)
FIELDS = (
    "electron_den",
    "electron_Temp",
    "electron_ud",
    "electron_vd",
    "electron_wd",
    "ion_den",
    "ion_Temp",
    "ion_ud",
    "ion_vd",
)
SCAN_MODES = tuple(range(1, 13))


@dataclass
class EpochData:
    name: str
    start_us: float
    end_us: float
    x_m: np.ndarray
    profiles: dict[str, np.ndarray]
    observed_frequency_mhz: float
    observed_profile: np.ndarray
    observed_pod_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, default=SOURCE_H5)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--scan-points", type=int, default=34)
    parser.add_argument("--target-points", type=int, default=50)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def unique_periodic_axis(axis: np.ndarray) -> int:
    if len(axis) < 2:
        raise ValueError("Azimuthal axis has fewer than two points")
    return len(axis) - 1


def phase_frequency_mhz(signal: np.ndarray, time_us: np.ndarray) -> float:
    amplitude = np.abs(signal)
    keep = amplitude >= np.quantile(amplitude, 0.25)
    if np.count_nonzero(keep) < 4:
        keep = np.ones(len(signal), dtype=bool)
    phase = np.unwrap(np.angle(signal))
    weights = amplitude[keep] ** 2
    design = np.column_stack((time_us[keep], np.ones(np.count_nonzero(keep))))
    lhs = design * np.sqrt(weights)[:, None]
    rhs = phase[keep] * np.sqrt(weights)
    slope = np.linalg.lstsq(lhs, rhs, rcond=None)[0][0]
    return float(abs(slope) / (2.0 * np.pi))


def load_epochs(path: Path) -> tuple[list[EpochData], float]:
    epochs: list[EpochData] = []
    with h5py.File(path, "r") as source:
        time_us = np.asarray(source["axes/time_s"], dtype=np.float64) * 1.0e6
        x_m = np.asarray(source["axes/x_m"], dtype=np.float64)
        y_m = np.asarray(source["axes/y_m"], dtype=np.float64)
        ny = unique_periodic_axis(y_m)
        ly_m = float(y_m[-1] - y_m[0])
        for name, start_us, end_us in PROFILE_WINDOWS:
            indices = np.flatnonzero((time_us >= start_us) & (time_us <= end_us))
            if len(indices) < 4:
                raise ValueError(f"Too few frames for {name}: {len(indices)}")
            profiles = {
                field: np.mean(
                    np.asarray(source[f"fields/{field}"][indices, :, :ny], dtype=np.float64),
                    axis=(0, 2),
                )
                for field in FIELDS
            }
            efy = np.asarray(source["fields/efy"][indices, :, :ny], dtype=np.float64)
            n1 = np.fft.rfft(efy, axis=2)[..., 1] / ny
            _, singular, vh = np.linalg.svd(n1, full_matrices=False)
            profile = np.conj(vh[0])
            temporal = n1 @ profile
            pod_fraction = float(singular[0] ** 2 / np.sum(singular**2))
            epochs.append(
                EpochData(
                    name=name,
                    start_us=start_us,
                    end_us=end_us,
                    x_m=x_m,
                    profiles=profiles,
                    observed_frequency_mhz=phase_frequency_mhz(temporal, time_us[indices]),
                    observed_profile=profile / np.linalg.norm(profile),
                    observed_pod_fraction=pod_fraction,
                )
            )
    return epochs, ly_m


def derivative_matrices(points: int, length: float, fluid_bc: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points < 8:
        raise ValueError("At least eight interior points are required")
    dx = length / (points + 1)
    x = dx * np.arange(1, points + 1)
    d1 = np.zeros((points, points), dtype=np.float64)
    d2 = np.zeros((points, points), dtype=np.float64)
    for i in range(points):
        if i > 0:
            d1[i, i - 1] = -0.5 / dx
            d2[i, i - 1] = 1.0 / dx**2
        if i < points - 1:
            d1[i, i + 1] = 0.5 / dx
            d2[i, i + 1] = 1.0 / dx**2
        d2[i, i] = -2.0 / dx**2
    if fluid_bc == "neumann":
        d1[0, 0] -= 0.5 / dx
        d1[-1, -1] += 0.5 / dx
        d2[0, 0] += 1.0 / dx**2
        d2[-1, -1] += 1.0 / dx**2
    elif fluid_bc != "dirichlet":
        raise ValueError(f"Unknown fluid boundary condition: {fluid_bc}")
    return x, d1, d2


def interpolate_profiles(epoch: EpochData, x: np.ndarray, smooth_cells: float) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for field, values in epoch.profiles.items():
        work = gaussian_filter1d(values, smooth_cells, mode="nearest") if smooth_cells > 0 else values
        output[field] = np.interp(x, epoch.x_m, work)
    for density in ("electron_den", "ion_den"):
        floor = max(float(np.quantile(output[density], 0.02)), 1.0e10)
        output[density] = np.maximum(output[density], floor)
    for temperature in ("electron_Temp", "ion_Temp"):
        output[temperature] = np.maximum(output[temperature], 0.05)
    return output


def block_slice(block: int, points: int) -> slice:
    return slice(block * points, (block + 1) * points)


def build_operator(
    epoch: EpochData,
    ly_m: float,
    mode_n: int,
    points: int,
    fluid_bc: str,
    smooth_cells: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    length = float(epoch.x_m[-1] - epoch.x_m[0])
    x_local, d1, _ = derivative_matrices(points, length, fluid_bc)
    x = x_local + float(epoch.x_m[0])
    _, _, d2_phi = derivative_matrices(points, length, "dirichlet")
    p = interpolate_profiles(epoch, x, smooth_cells)
    eye = np.eye(points)
    ky = 2.0 * np.pi * mode_n / ly_m
    poisson = d2_phi - ky**2 * eye
    pe = linalg.solve(poisson, (E_CHARGE / EPS0) * np.diag(p["electron_den"]), assume_a="sym")
    pi = linalg.solve(poisson, (-E_CHARGE / EPS0) * np.diag(p["ion_den"]), assume_a="sym")

    size = 7 * points
    operator = np.zeros((size, size), dtype=np.complex128)
    en, eu, ev, ew, inn, iu, iv = [block_slice(i, points) for i in range(7)]

    def diag(values: np.ndarray) -> np.ndarray:
        return np.diag(values)

    def add(row: slice, column: slice, values: np.ndarray) -> None:
        operator[row, column] += values

    def species_continuity(n0, u0, v0, density, ux, vy):
        dlogn = d1 @ np.log(n0)
        add(density, density, -diag(u0) @ d1 - diag(d1 @ u0 + u0 * dlogn) - 1j * ky * diag(v0))
        add(density, ux, -d1 - diag(dlogn))
        add(density, vy, -1j * ky * eye)

    species_continuity(p["electron_den"], p["electron_ud"], p["electron_vd"], en, eu, ev)
    species_continuity(p["ion_den"], p["ion_ud"], p["ion_vd"], inn, iu, iv)

    ce2 = E_CHARGE * p["electron_Temp"] / M_E
    ci2 = E_CHARGE * p["ion_Temp"] / M_I
    qe_me = -E_CHARGE / M_E
    qi_mi = E_CHARGE / M_I
    omega_e = qe_me * B_X

    adv_e = -diag(p["electron_ud"]) @ d1 - 1j * ky * diag(p["electron_vd"])
    add(eu, eu, adv_e - diag(d1 @ p["electron_ud"]))
    add(eu, en, -diag(ce2) @ d1 - qe_me * d1 @ pe)
    add(eu, inn, -qe_me * d1 @ pi)

    add(ev, ev, adv_e)
    add(ev, eu, -diag(d1 @ p["electron_vd"]))
    add(ev, en, -1j * ky * diag(ce2) - 1j * ky * qe_me * pe)
    add(ev, inn, -1j * ky * qe_me * pi)
    add(ev, ew, omega_e * eye)

    add(ew, ew, adv_e)
    add(ew, eu, -diag(d1 @ p["electron_wd"]))
    add(ew, ev, -omega_e * eye)

    adv_i = -diag(p["ion_ud"]) @ d1 - 1j * ky * diag(p["ion_vd"])
    add(iu, iu, adv_i - diag(d1 @ p["ion_ud"]))
    add(iu, inn, -diag(ci2) @ d1 - qi_mi * d1 @ pi)
    add(iu, en, -qi_mi * d1 @ pe)

    add(iv, iv, adv_i)
    add(iv, iu, -diag(d1 @ p["ion_vd"]))
    add(iv, inn, -1j * ky * diag(ci2) - 1j * ky * qi_mi * pi)
    add(iv, en, -1j * ky * qi_mi * pe)

    scales = np.concatenate(
        (
            np.ones(points),
            np.full(3 * points, 1.0e6),
            np.ones(points),
            np.full(2 * points, 1.0e4),
        )
    )
    operator = (operator * scales[None, :]) / scales[:, None]
    operator *= 1.0e-6
    metadata = {
        "x_m": x,
        "pe": pe,
        "pi": pi,
        "scales": scales,
        "ky_rad_m": ky,
        "fluid_bc": fluid_bc,
        "smooth_cells": smooth_cells,
    }
    return operator, x, metadata


def eigensystem(operator: np.ndarray, vectors: bool) -> tuple[np.ndarray, np.ndarray | None]:
    if vectors:
        values, right = linalg.eig(operator, overwrite_a=True, check_finite=False)
        return values, right
    return linalg.eigvals(operator, overwrite_a=True, check_finite=False), None


def root_table(values: np.ndarray) -> dict[str, np.ndarray]:
    frequency = -values.imag / (2.0 * np.pi)
    growth = values.real
    finite = np.isfinite(frequency) & np.isfinite(growth)
    band = finite & (np.abs(frequency) >= 0.05) & (np.abs(frequency) <= 10.0) & (np.abs(growth) <= 30.0)
    return {"frequency": frequency, "growth": growth, "band": band}


def choose_roots(values: np.ndarray, target_frequency: float) -> tuple[int | None, int | None]:
    table = root_table(values)
    candidates = np.flatnonzero(table["band"])
    if len(candidates) == 0:
        return None, None
    nearest = int(candidates[np.argmin(np.abs(np.abs(table["frequency"][candidates]) - target_frequency))])
    fastest = int(candidates[np.argmax(table["growth"][candidates])])
    return nearest, fastest


def interpolate_observed_profile(epoch: EpochData, x: np.ndarray) -> np.ndarray:
    real = np.interp(x, epoch.x_m, epoch.observed_profile.real)
    imag = np.interp(x, epoch.x_m, epoch.observed_profile.imag)
    profile = real + 1j * imag
    return profile / np.linalg.norm(profile)


def mode_metrics(
    epoch: EpochData,
    ly_m: float,
    mode_n: int,
    points: int,
    fluid_bc: str,
    smooth_cells: float,
    need_vectors: bool,
) -> tuple[dict, dict | None]:
    operator, x, metadata = build_operator(epoch, ly_m, mode_n, points, fluid_bc, smooth_cells)
    values, vectors = eigensystem(operator, need_vectors)
    nearest, fastest = choose_roots(values, epoch.observed_frequency_mhz)
    table = root_table(values)
    row = {
        "epoch": epoch.name,
        "start_us": epoch.start_us,
        "end_us": epoch.end_us,
        "mode_n": mode_n,
        "points": points,
        "fluid_bc": fluid_bc,
        "smooth_cells": smooth_cells,
        "observed_frequency_mhz": epoch.observed_frequency_mhz,
        "observed_pod_fraction": epoch.observed_pod_fraction,
    }
    details = None
    for label, index in (("nearest", nearest), ("fastest", fastest)):
        row[f"{label}_frequency_mhz"] = float(abs(table["frequency"][index])) if index is not None else math.nan
        row[f"{label}_signed_frequency_mhz"] = float(table["frequency"][index]) if index is not None else math.nan
        row[f"{label}_growth_per_us"] = float(table["growth"][index]) if index is not None else math.nan
    if nearest is not None and vectors is not None:
        points_local = len(x)
        observed = interpolate_observed_profile(epoch, x)

        def phi_overlap_roughness(index: int) -> tuple[np.ndarray, float, float]:
            scaled_vector = vectors[:, index] * metadata["scales"]
            eta_e = scaled_vector[block_slice(0, points_local)]
            eta_i = scaled_vector[block_slice(4, points_local)]
            phi_local = metadata["pe"] @ eta_e + metadata["pi"] @ eta_i
            norm = np.linalg.norm(phi_local)
            if norm <= 0:
                return phi_local, 0.0, math.inf
            phi_local /= norm
            roughness = float(np.linalg.norm(np.diff(phi_local, n=2)) / np.linalg.norm(phi_local))
            return phi_local, float(abs(np.vdot(observed, phi_local))), roughness

        nearest_phi, nearest_overlap, nearest_roughness = phi_overlap_roughness(nearest)
        row["nearest_radial_overlap"] = nearest_overlap
        row["nearest_radial_roughness"] = nearest_roughness
        observed_roughness = float(np.linalg.norm(np.diff(observed, n=2)) / np.linalg.norm(observed))
        roughness_limit = max(0.35, 4.0 * observed_roughness)
        row["observed_radial_roughness"] = observed_roughness
        row["candidate_roughness_limit"] = roughness_limit
        candidates = np.flatnonzero(
            table["band"]
            & (np.abs(np.abs(table["frequency"]) - epoch.observed_frequency_mhz) <= 0.25 * epoch.observed_frequency_mhz)
        )
        candidate_data = []
        rejected_data = []
        for index in candidates:
            phi_local, overlap, roughness = phi_overlap_roughness(int(index))
            if roughness <= roughness_limit:
                candidate_data.append((int(index), phi_local, overlap, roughness))
            else:
                rejected_data.append((int(index), phi_local, overlap, roughness))
        row["smooth_candidate_count"] = len(candidate_data)
        shape_best = max(candidate_data, key=lambda item: item[2]) if candidate_data else None
        growing = [item for item in candidate_data if table["growth"][item[0]] > 0]
        growing_best = max(growing, key=lambda item: item[2]) if growing else None
        for label, candidate in (("shape_matched", shape_best), ("growing_shape_matched", growing_best)):
            if candidate is None:
                row[f"{label}_frequency_mhz"] = math.nan
                row[f"{label}_growth_per_us"] = math.nan
                row[f"{label}_radial_overlap"] = math.nan
                row[f"{label}_radial_roughness"] = math.nan
                continue
            index, _, overlap, roughness = candidate
            row[f"{label}_frequency_mhz"] = float(abs(table["frequency"][index]))
            row[f"{label}_growth_per_us"] = float(table["growth"][index])
            row[f"{label}_radial_overlap"] = overlap
            row[f"{label}_radial_roughness"] = roughness
        display = shape_best
        if display is None:
            pool = rejected_data if rejected_data else [(nearest, nearest_phi, nearest_overlap, nearest_roughness)]
            display = min(pool, key=lambda item: item[3])
        details = {
            "x_m": x,
            "phi": display[1],
            "observed": observed,
            "eigenvalue": values[display[0]],
            "accepted_smooth_candidate": shape_best is not None,
            "display_frequency_mhz": float(abs(table["frequency"][display[0]])),
            "display_growth_per_us": float(table["growth"][display[0]]),
            "display_overlap": float(display[2]),
            "display_roughness": float(display[3]),
        }
    else:
        row["nearest_radial_overlap"] = math.nan
        row["nearest_radial_roughness"] = math.nan
        row["observed_radial_roughness"] = math.nan
        row["candidate_roughness_limit"] = math.nan
        row["smooth_candidate_count"] = 0
        row["shape_matched_frequency_mhz"] = math.nan
        row["shape_matched_growth_per_us"] = math.nan
        row["shape_matched_radial_overlap"] = math.nan
        row["shape_matched_radial_roughness"] = math.nan
        row["growing_shape_matched_frequency_mhz"] = math.nan
        row["growing_shape_matched_growth_per_us"] = math.nan
        row["growing_shape_matched_radial_overlap"] = math.nan
        row["growing_shape_matched_radial_roughness"] = math.nan
    return row, details


def plot_scan(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8), sharex=True)
    colors = {"growth": "#0072B2", "post_growth": "#D55E00", "steady": "#009E73"}
    for epoch in colors:
        local = sorted((row for row in rows if row["epoch"] == epoch), key=lambda item: item["mode_n"])
        n = [row["mode_n"] for row in local]
        axes[0].plot(n, [row["nearest_growth_per_us"] for row in local], "o-", color=colors[epoch], label=epoch)
        axes[1].plot(n, [row["nearest_frequency_mhz"] for row in local], "o-", color=colors[epoch], label=epoch)
    axes[0].axhline(0.0, color="0.25", linewidth=1)
    axes[0].set_ylabel("growth rate (1/us)\nroot nearest PIC n=1 frequency")
    axes[1].set_ylabel("frequency (MHz)")
    axes[1].set_xlabel("azimuthal mode n")
    axes[1].set_xticks(SCAN_MODES)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", frameon=True)
    fig.suptitle("B30 reduced two-fluid global-mode screen")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_n1_comparison(details: dict[str, dict], rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(len(PROFILE_WINDOWS), 2, figsize=(12, 9))
    for row_index, (epoch, _, _) in enumerate(PROFILE_WINDOWS):
        row = next(item for item in rows if item["epoch"] == epoch and item["mode_n"] == 1)
        detail = details[epoch]
        phase = np.angle(np.vdot(detail["observed"], detail["phi"]))
        theory = detail["phi"] * np.exp(-1j * phase)
        x_cm = detail["x_m"] * 100.0
        axes[row_index, 0].plot(x_cm, np.abs(detail["observed"]), label="PIC POD |profile|", color="#0072B2")
        axes[row_index, 0].plot(x_cm, np.abs(theory), label="fluid eigenmode |profile|", color="#D55E00")
        axes[row_index, 1].plot(x_cm, np.unwrap(np.angle(detail["observed"])), label="PIC POD phase", color="#0072B2")
        axes[row_index, 1].plot(x_cm, np.unwrap(np.angle(theory)), label="fluid eigenmode phase", color="#D55E00")
        axes[row_index, 0].set_title(
            f"{epoch}: fPIC={row['observed_frequency_mhz']:.2f}, ffluid={detail['display_frequency_mhz']:.2f} MHz"
        )
        accepted = "accepted" if detail["accepted_smooth_candidate"] else "rejected: grid-rough"
        axes[row_index, 1].set_title(
            f"growth={detail['display_growth_per_us']:.2f} 1/us, overlap={detail['display_overlap']:.3f} ({accepted})"
        )
        for axis in axes[row_index]:
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right", frameon=True)
            axis.set_xlabel("radial coordinate (cm)")
    axes[0, 0].set_ylabel("normalized amplitude")
    axes[0, 1].set_ylabel("phase (rad)")
    fig.suptitle("B30 n=1 PIC radial POD vs reduced-fluid global eigenmode")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sensitivity(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = {"growth": "#0072B2", "post_growth": "#D55E00", "steady": "#009E73"}
    markers = {"dirichlet": "o", "neumann": "s"}
    for epoch, color in colors.items():
        for bc, marker in markers.items():
            local = sorted(
                (row for row in rows if row["epoch"] == epoch and row["fluid_bc"] == bc),
                key=lambda item: item["points"],
            )
            label = f"{epoch}, {bc}"
            axes[0].plot([r["points"] for r in local], [r["shape_matched_frequency_mhz"] for r in local], marker + "-", color=color, label=label)
            axes[1].plot([r["points"] for r in local], [r["shape_matched_growth_per_us"] for r in local], marker + "-", color=color)
            axes[2].plot([r["points"] for r in local], [r["shape_matched_radial_overlap"] for r in local], marker + "-", color=color)
    axes[0].set_ylabel("frequency (MHz)")
    axes[1].set_ylabel("growth rate (1/us)")
    axes[2].set_ylabel("PIC radial overlap")
    for axis in axes:
        axis.set_xlabel("interior radial points")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best", frameon=True, fontsize=8)
    fig.suptitle("B30 n=1 reduced-fluid sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.source_h5.is_file():
        raise FileNotFoundError(args.source_h5)
    args.output.mkdir(parents=True, exist_ok=True)
    epochs, ly_m = load_epochs(args.source_h5)

    scan_rows: list[dict] = []
    target_details: dict[str, dict] = {}
    target_rows: list[dict] = []
    for epoch in epochs:
        for mode_n in SCAN_MODES:
            row, _ = mode_metrics(epoch, ly_m, mode_n, args.scan_points, "dirichlet", 1.5, False)
            scan_rows.append(row)
        row, detail = mode_metrics(epoch, ly_m, 1, args.target_points, "dirichlet", 1.5, True)
        target_rows.append(row)
        if detail is None:
            raise RuntimeError(f"No n=1 candidate root for {epoch.name}")
        target_details[epoch.name] = detail

    sensitivity_rows: list[dict] = []
    resolutions = sorted(set((max(18, args.target_points - 16), args.target_points, args.target_points + 16)))
    for epoch in epochs:
        for bc in ("dirichlet", "neumann"):
            for points in resolutions:
                row, _ = mode_metrics(epoch, ly_m, 1, points, bc, 1.5, True)
                sensitivity_rows.append(row)

    write_csv(args.output / "global_mode_scan_n1_to_n12.csv", scan_rows)
    write_csv(args.output / "n1_epoch_comparison.csv", target_rows)
    write_csv(args.output / "n1_resolution_boundary_sensitivity.csv", sensitivity_rows)
    plot_scan(scan_rows, args.output / "global_mode_scan_n1_to_n12.png")
    plot_n1_comparison(target_details, target_rows, args.output / "n1_pic_vs_fluid_radial_structure.png")
    plot_sensitivity(sensitivity_rows, args.output / "n1_resolution_boundary_sensitivity.png")

    robust_rows = []
    for epoch in epochs:
        local = [row for row in sensitivity_rows if row["epoch"] == epoch.name]
        finite = [row for row in local if math.isfinite(row["shape_matched_frequency_mhz"])]
        robust_rows.append(
            {
                "epoch": epoch.name,
                "accepted_configurations": len(finite),
                "total_configurations": len(local),
                "frequency_min_mhz": min((row["shape_matched_frequency_mhz"] for row in finite), default=math.nan),
                "frequency_max_mhz": max((row["shape_matched_frequency_mhz"] for row in finite), default=math.nan),
                "growth_min_per_us": min((row["shape_matched_growth_per_us"] for row in finite), default=math.nan),
                "growth_max_per_us": max((row["shape_matched_growth_per_us"] for row in finite), default=math.nan),
                "overlap_min": min((row["shape_matched_radial_overlap"] for row in finite), default=math.nan),
                "overlap_max": max((row["shape_matched_radial_overlap"] for row in finite), default=math.nan),
                "positive_growth_fraction": float(np.mean([row["shape_matched_growth_per_us"] > 0 for row in finite])) if finite else math.nan,
            }
        )

    criteria = []
    for row in target_rows:
        has_candidate = math.isfinite(row["shape_matched_frequency_mhz"])
        relative_frequency_error = (
            abs(row["shape_matched_frequency_mhz"] - row["observed_frequency_mhz"]) / row["observed_frequency_mhz"]
            if has_candidate
            else math.nan
        )
        criteria.append(
            {
                "epoch": row["epoch"],
                "smooth_candidate_exists": has_candidate,
                "positive_growth": has_candidate and row["shape_matched_growth_per_us"] > 0,
                "frequency_within_25_percent": has_candidate and relative_frequency_error <= 0.25,
                "radial_overlap_at_least_0p7": has_candidate and row["shape_matched_radial_overlap"] >= 0.7,
                "relative_frequency_error": relative_frequency_error,
            }
        )
    all_three_any_epoch = any(
        item["positive_growth"] and item["frequency_within_25_percent"] and item["radial_overlap_at_least_0p7"]
        for item in criteria
    )
    summary = {
        "status": "PASS",
        "source_h5": str(args.source_h5.resolve()),
        "question": "Can a reduced nonuniform two-fluid global eigenmode explain the observed B30 n=1 long wave before a domain-length PIC?",
        "model": {
            "type": "electrostatic isothermal two-fluid radial global eigenvalue screen",
            "retained": [
                "radial density/temperature/mean-flow profiles",
                "electron magnetic coupling in azimuthal-axial velocity",
                "ion inertia",
                "Poisson coupling with radial Dirichlet potential",
                "discrete azimuthal Fourier modes",
            ],
            "omitted": [
                "velocity-distribution functions and cyclotron harmonics",
                "temperature anisotropy and pressure-tensor perturbations",
                "kinetic absorbing-wall/sheath response",
                "collisions, ionization-source perturbations, and nonlinear coupling",
            ],
            "interpretation_limit": "Supportive screen only; agreement is not a kinetic/global-stability proof.",
        },
        "base_n1": target_rows,
        "sensitivity_ranges": robust_rows,
        "criteria": criteria,
        "all_three_criteria_met_in_any_epoch": all_three_any_epoch,
        "decision_rule": {
            "supportive": "positive growth, frequency within 25%, and radial overlap >=0.7 in a physically relevant epoch, with reasonable sensitivity",
            "not_decisive": "Even a supportive result cannot distinguish a physical global mode from a system-size-selected mode at one Ly.",
            "next_if_inconclusive": "Proceed with the pre-registered Ly-doubling PIC control.",
        },
    }
    (args.output / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# B30 reduced global-stability screen",
        "",
        "This is a nonuniform electrostatic two-fluid screening calculation, not a kinetic stability proof.",
        "",
        "## Base n=1 comparison",
        "",
        "| epoch | PIC f (MHz) | fluid f (MHz) | growth (1/us) | radial overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in target_rows:
        fluid_frequency = row["shape_matched_frequency_mhz"]
        fluid_growth = row["shape_matched_growth_per_us"]
        fluid_overlap = row["shape_matched_radial_overlap"]
        lines.append(
            f"| {row['epoch']} | {row['observed_frequency_mhz']:.3f} | "
            f"{fluid_frequency:.3f} | {fluid_growth:.3f} | {fluid_overlap:.3f} |"
            if math.isfinite(fluid_frequency)
            else f"| {row['epoch']} | {row['observed_frequency_mhz']:.3f} | no smooth root | - | - |"
        )
    lines.extend(
        (
            "",
            "## Boundary/resolution sensitivity",
            "",
            "| epoch | accepted / total | frequency range (MHz) | growth range (1/us) | overlap range |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for row in robust_rows:
        if row["accepted_configurations"]:
            lines.append(
                f"| {row['epoch']} | {row['accepted_configurations']} / {row['total_configurations']} | "
                f"{row['frequency_min_mhz']:.3f} to {row['frequency_max_mhz']:.3f} | "
                f"{row['growth_min_per_us']:.3f} to {row['growth_max_per_us']:.3f} | "
                f"{row['overlap_min']:.3f} to {row['overlap_max']:.3f} |"
            )
        else:
            lines.append(f"| {row['epoch']} | 0 / {row['total_configurations']} | - | - | - |")
    lines.extend(
        (
            "",
            "All accepted smooth roots came from the approximate zero-normal-gradient fluid boundary. "
            "The zero-perturbation fluid boundary produced only grid-rough roots in the PIC-frequency band. "
            "The growth sign also changed with resolution/background epoch. The screen is therefore boundary-sensitive and inconclusive.",
            "",
            "## Guardrails",
            "",
            "- A match supports a reduced-fluid global-mode interpretation but does not prove the kinetic PIC eigenmode.",
            "- A mismatch does not rule out a kinetic global mode because VDFs, anisotropic pressure, and kinetic wall response are omitted.",
            "- One azimuthal length cannot distinguish a fixed physical wavelength from a box-selected system-scale mode.",
            "- The pre-registered doubled-Ly PIC remains the decisive test of wavelength selection if this screen is inconclusive.",
            "- The n=1--12 scan is exploratory only because smooth branch identity could not be fixed under both fluid boundary closures.",
            "",
            f"All three screening criteria met in any epoch: `{all_three_any_epoch}`.",
        )
    )
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
