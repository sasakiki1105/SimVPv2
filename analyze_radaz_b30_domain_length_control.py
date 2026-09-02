#!/usr/bin/env python3
"""Compare the B30 baseline with a doubled-azimuthal-length PIC control.

The primary question is whether the mature long wave keeps its physical
wavelength or follows the periodic box length. The script deliberately treats
that mode-number test as primary and frequency/growth/radial-shape comparisons
as secondary diagnostics.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
REFERENCE_CASE = "2D_RadAz_Xe1p_Bx30mT_Ez10kVm_dt15ps_out15ns"
CONTROL_CASE = "2D_RadAz_Xe1p_Bx30mT_Ez10kVm_Ly25p6mm_Ny512_dt15ps_out15ns"
REFERENCE_H5 = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / REFERENCE_CASE
    / REFERENCE_CASE
    / "analysis_fields_uncompressed.h5"
)
CONTROL_H5 = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / CONTROL_CASE
    / CONTROL_CASE
    / "analysis_fields_uncompressed.h5"
)
OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / "analysis_results"
    / "B30_domain_length_control_Ly12p8_vs_25p6mm"
)

FIELDS = ("efy", "electron_den")
LOW_MODES = np.arange(1, 7)
EARLY_END_US = 5.0
STEADY_START_US = 20.0
STEADY_END_US = 30.0
SMOOTH_US = 0.15


@dataclass
class CaseData:
    label: str
    path: Path
    time_us: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    coefficients: dict[str, np.ndarray]

    @property
    def ly_m(self) -> float:
        if len(self.y_m) < 2:
            raise ValueError("Azimuthal axis has fewer than two unique points")
        return float((self.y_m[-1] - self.y_m[0]) + np.median(np.diff(self.y_m)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-h5", type=Path, default=REFERENCE_H5)
    parser.add_argument("--control-h5", type=Path, default=CONTROL_H5)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate available inputs without requiring the future control HDF5.",
    )
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
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


def validate_h5(path: Path, expected_ny: int | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as source:
        required = ["axes/time_s", "axes/x_m", "axes/y_m"] + [
            f"fields/{field}" for field in FIELDS
        ]
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Missing datasets in {path}: {missing}")
        time = np.asarray(source["axes/time_s"])
        x_m = np.asarray(source["axes/x_m"])
        y_full = np.asarray(source["axes/y_m"])
        shape = tuple(source["fields/efy"].shape)
        if shape != (len(time), len(x_m), len(y_full)):
            raise ValueError(f"Axis/field shape mismatch in {path}: {shape}")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(x_m)) or not np.all(np.isfinite(y_full)):
            raise ValueError(f"Non-finite coordinate in {path}")
        ny = len(y_full) - 1
        if expected_ny is not None and ny != expected_ny:
            raise ValueError(f"Expected Ny={expected_ny}, found {ny} in {path}")
        return {
            "path": str(path.resolve()),
            "frames": len(time),
            "nx_unique": len(x_m) - 1,
            "ny_unique": ny,
            "time_start_us": float(time[0] * 1.0e6),
            "time_end_us": float(time[-1] * 1.0e6),
            "frame_interval_ns": float(np.median(np.diff(time)) * 1.0e9),
            "lx_mm": float((x_m[-1] - x_m[0]) * 1.0e3),
            "ly_mm": float((y_full[-1] - y_full[0]) * 1.0e3),
        }


def contiguous_bounds(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    return [(int(group[0]), int(group[-1]) + 1) for group in np.split(indices, cuts)]


def load_case(label: str, path: Path) -> CaseData:
    with h5py.File(path, "r") as source:
        time_us = np.asarray(source["axes/time_s"], dtype=np.float64) * 1.0e6
        x_m = np.asarray(source["axes/x_m"], dtype=np.float64)
        y_full = np.asarray(source["axes/y_m"], dtype=np.float64)
        ny = len(y_full) - 1
        y_m = y_full[:ny]
        radial = np.flatnonzero((x_m >= 0.09e-2) & (x_m <= 1.19e-2))
        if len(radial) == 0:
            raise ValueError(f"No radial samples in the injection range for {path}")
        coefficients: dict[str, np.ndarray] = {}
        for field in FIELDS:
            output = np.empty((len(time_us), len(radial), int(LOW_MODES[-1]) + 1), np.complex128)
            indices = np.arange(len(time_us))
            cursor = 0
            for first, stop in contiguous_bounds(indices):
                for start in range(first, stop, 32):
                    end = min(start + 32, stop)
                    values = np.asarray(source[f"fields/{field}"][start:end, radial, :ny], dtype=np.float64)
                    fft = np.fft.rfft(values, axis=2) / ny
                    count = end - start
                    output[cursor : cursor + count] = fft[..., : int(LOW_MODES[-1]) + 1]
                    cursor += count
            coefficients[field] = output
    return CaseData(label=label, path=path, time_us=time_us, x_m=x_m[radial], y_m=y_m, coefficients=coefficients)


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1:
        return values.copy()
    left = width // 2
    right = width - left - 1
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def mode_amplitude(coefficients: np.ndarray, mode: int) -> np.ndarray:
    return np.sqrt(np.mean(np.abs(coefficients[..., mode]) ** 2, axis=1))


def dominant_frequency_mhz(coefficients: np.ndarray, time_us: np.ndarray, mode: int, steady: np.ndarray) -> float:
    matrix = coefficients[steady, :, mode]
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    temporal = matrix @ np.conj(vh[0])
    temporal = temporal - np.mean(temporal)
    frequency = np.fft.fftfreq(len(temporal), d=float(np.median(np.diff(time_us[steady]))))
    spectrum = np.abs(np.fft.fft(temporal)) ** 2
    spectrum[np.isclose(frequency, 0.0)] = 0.0
    return float(frequency[int(np.argmax(spectrum))])


def radial_profile(coefficients: np.ndarray, mode: int, steady: np.ndarray) -> np.ndarray:
    matrix = coefficients[steady, :, mode]
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    profile = vh[0]
    norm = np.linalg.norm(profile)
    return profile / norm if norm > 0 else profile


def onset_time_us(case: CaseData, field: str, mode: int) -> tuple[float, float]:
    amplitude = mode_amplitude(case.coefficients[field], mode)
    steady = (case.time_us >= STEADY_START_US) & (case.time_us <= STEADY_END_US)
    early = case.time_us <= EARLY_END_US
    steady_level = float(np.median(amplitude[steady]))
    dt_us = float(np.median(np.diff(case.time_us)))
    smoothed = moving_average(amplitude, round(SMOOTH_US / dt_us))
    candidates = np.flatnonzero(early & (smoothed >= 0.1 * steady_level))
    onset = float(case.time_us[candidates[0]]) if len(candidates) else math.nan
    return onset, steady_level


def case_mode_rows(case: CaseData) -> list[dict]:
    steady = (case.time_us >= STEADY_START_US) & (case.time_us <= STEADY_END_US)
    rows: list[dict] = []
    for field in FIELDS:
        power = np.array(
            [np.mean(np.abs(case.coefficients[field][steady, :, mode]) ** 2) for mode in LOW_MODES]
        )
        total = float(np.sum(power))
        for mode, value in zip(LOW_MODES, power):
            onset, steady_level = onset_time_us(case, field, int(mode))
            rows.append(
                {
                    "case": case.label,
                    "field": field,
                    "mode_n": int(mode),
                    "ly_mm": case.ly_m * 1.0e3,
                    "wavelength_mm": case.ly_m * 1.0e3 / int(mode),
                    "steady_power": float(value),
                    "steady_low_mode_power_fraction": float(value / total) if total > 0 else math.nan,
                    "steady_amplitude": steady_level,
                    "onset_10pct_us": onset,
                    "dominant_frequency_mhz": dominant_frequency_mhz(
                        case.coefficients[field], case.time_us, int(mode), steady
                    ),
                }
            )
    return rows


def profile_overlap(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError("Radial grids differ; interpolation is required")
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def classify(reference_n: int, control_n: int, reference_ly: float, control_ly: float) -> str:
    reference_lambda = reference_ly / reference_n
    control_lambda = control_ly / control_n
    relative_error = abs(control_lambda - reference_lambda) / reference_lambda
    if control_n == 2 * reference_n and relative_error <= 0.1:
        return "fixed_physical_wavelength"
    if control_n == reference_n and math.isclose(control_ly / reference_ly, 2.0, rel_tol=0.05):
        return "system_size_selected"
    return "finite_domain_mode_competition_or_ambiguous"


def reserve_legend_space(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.subplots_adjust(right=0.78)
    ax.legend(loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=True)


def make_plots(reference: CaseData, control: CaseData, rows: list[dict], output: Path) -> None:
    colors = {"reference": "#0072B2", "control": "#D55E00"}
    markers = {"reference": "o", "control": "s"}

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for case in (reference, control):
        selected = [row for row in rows if row["case"] == case.label and row["field"] == "efy"]
        ax.plot(
            [row["wavelength_mm"] for row in selected],
            [row["steady_low_mode_power_fraction"] for row in selected],
            marker=markers[case.label],
            color=colors[case.label],
            label=f"{case.label}: Ly={case.ly_m * 1e3:.1f} mm",
        )
    ax.set_xlabel("Physical azimuthal wavelength [mm]")
    ax.set_ylabel("Fraction of steady low-mode Ey power")
    ax.set_title("B30 domain-length control: wavelength selection")
    ax.grid(alpha=0.25)
    reserve_legend_space(fig, ax)
    fig.savefig(output / "b30_domain_length_steady_wavelength_selection.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for case, modes in ((reference, (1,)), (control, (1, 2))):
        for mode in modes:
            amplitude = mode_amplitude(case.coefficients["efy"], mode)
            ax.plot(
                case.time_us,
                amplitude,
                color=colors[case.label],
                linestyle="-" if mode == 1 else "--",
                label=f"{case.label} n={mode}, lambda={case.ly_m * 1e3 / mode:.1f} mm",
            )
    ax.set_xlim(0.0, EARLY_END_US)
    ax.set_yscale("log")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Radial-RMS Ey mode amplitude [V/m]")
    ax.set_title("B30 domain-length control: startup of candidate long modes")
    ax.grid(alpha=0.25)
    reserve_legend_space(fig, ax)
    fig.savefig(output / "b30_domain_length_startup_mode_amplitude.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for case, modes in ((reference, (1,)), (control, (1, 2))):
        steady = (case.time_us >= STEADY_START_US) & (case.time_us <= STEADY_END_US)
        for mode in modes:
            profile = radial_profile(case.coefficients["efy"], mode, steady)
            ax.plot(
                case.x_m * 1.0e2,
                np.abs(profile),
                color=colors[case.label],
                linestyle="-" if mode == 1 else "--",
                label=f"{case.label} n={mode}",
            )
    ax.set_xlabel("Radial coordinate [cm]")
    ax.set_ylabel("Normalized |POD1 radial profile|")
    ax.set_title("B30 domain-length control: mature radial structures")
    ax.grid(alpha=0.25)
    reserve_legend_space(fig, ax)
    fig.savefig(output / "b30_domain_length_radial_profiles.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    reference_health = validate_h5(args.reference_h5, expected_ny=256)
    if args.validate_only:
        control_exists = args.control_h5.is_file()
        control_health = validate_h5(args.control_h5, expected_ny=512) if control_exists else None
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "reference": reference_health,
                    "control_expected": str(args.control_h5.resolve()),
                    "control_exists": control_exists,
                    "control": control_health,
                },
                indent=2,
            )
        )
        return

    control_health = validate_h5(args.control_h5, expected_ny=512)
    args.output.mkdir(parents=True, exist_ok=True)
    reference = load_case("reference", args.reference_h5)
    control = load_case("control", args.control_h5)
    rows = case_mode_rows(reference) + case_mode_rows(control)
    write_csv(args.output / "low_mode_comparison.csv", rows)

    ey_rows = [row for row in rows if row["field"] == "efy"]
    reference_rows = [row for row in ey_rows if row["case"] == "reference"]
    control_rows = [row for row in ey_rows if row["case"] == "control"]
    reference_best = max(reference_rows, key=lambda row: row["steady_power"])
    control_best = max(control_rows, key=lambda row: row["steady_power"])
    verdict = classify(
        int(reference_best["mode_n"]),
        int(control_best["mode_n"]),
        reference.ly_m,
        control.ly_m,
    )

    ref_steady = (reference.time_us >= STEADY_START_US) & (reference.time_us <= STEADY_END_US)
    ctl_steady = (control.time_us >= STEADY_START_US) & (control.time_us <= STEADY_END_US)
    ref_profile = radial_profile(reference.coefficients["efy"], int(reference_best["mode_n"]), ref_steady)
    overlaps = {}
    for mode in (1, 2):
        ctl_profile = radial_profile(control.coefficients["efy"], mode, ctl_steady)
        overlaps[f"reference_n{int(reference_best['mode_n'])}_to_control_n{mode}"] = profile_overlap(
            ref_profile, ctl_profile
        )

    make_plots(reference, control, rows, args.output)
    summary = {
        "status": "PASS",
        "question": "Does the B30 long wave keep a physical wavelength or follow the periodic box length?",
        "primary_verdict": verdict,
        "reference_health": reference_health,
        "control_health": control_health,
        "reference_dominant_low_mode": reference_best,
        "control_dominant_low_mode": control_best,
        "radial_profile_overlaps": overlaps,
        "decision_rule": {
            "control_n2": "fixed physical wavelength near 12.8 mm",
            "control_n1": "system-size-selected mode; not by itself proof of a numerical artifact",
            "other_or_split": "finite-domain mode competition or ambiguous selection",
        },
        "guardrails": [
            "The mode-number result is primary; frequency, onset, and radial overlap are secondary.",
            "A control n=1 result implies system-size selection, not automatically a numerical artifact.",
            "Physical global-mode and periodic-box explanations then require a resolution or stability control.",
        ],
    }
    (args.output / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        "# B30 azimuthal-domain-length control\n\n"
        f"Primary verdict: `{verdict}`.\n\n"
        f"- Reference dominant low mode: n={reference_best['mode_n']}, "
        f"lambda={reference_best['wavelength_mm']:.3f} mm.\n"
        f"- Control dominant low mode: n={control_best['mode_n']}, "
        f"lambda={control_best['wavelength_mm']:.3f} mm.\n"
        "- `n=2` in the doubled domain supports a fixed physical wavelength.\n"
        "- `n=1` supports system-size selection but does not alone prove a numerical artifact.\n"
        "- Any other split requires a finite-domain mode-competition interpretation.\n",
        encoding="ascii",
    )
    print(json.dumps(json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
