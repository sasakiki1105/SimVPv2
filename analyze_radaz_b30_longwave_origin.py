#!/usr/bin/env python3
"""Diagnose the origin of the B30 long-wavelength mode without re-running PIC.

The analysis distinguishes evidence for four non-exclusive explanations:

1. independent finite-time growth of the long-wave mode;
2. difference-frequency generation by a specific adjacent-mode triad;
3. a radial/global eigenstructure tied to the system-scale n=1 mode;
4. a box-selected mode that requires a domain-length control PIC to resolve.

The existing 15 ns outputs are sufficient for the approximately 1.5 MHz long
mode, but not for a definitive signed energy-transfer budget.
"""

from __future__ import annotations

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
CASE_NAME = "2D_RadAz_Xe1p_Bx30mT_Ez10kVm_dt15ps_out15ns"
CASE_DIR = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / CASE_NAME
    / CASE_NAME
)
SOURCE_H5 = CASE_DIR / "analysis_fields_uncompressed.h5"
OUTPUT = (
    RESEARCH
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / "analysis_results"
    / "B30_long_wavelength_origin_existing_data"
)

SHARE_URL = "https://chatgpt.com/s/t_6a8e4cb3d14481918d2388028ca09069"
MAX_MODE = 48
LONG_MODES = np.arange(1, 4)
MTSI_MODES = np.arange(4, 19)
ECDI_MODES = np.arange(20, 49)
EARLY_END_US = 5.0
STEADY_START_US = 20.0
STEADY_END_US = 30.0
TRIAD_WINDOWS = (
    ("discovery", 0.30, 1.50),
    ("confirmation", 1.50, 3.00),
    ("late", 3.00, 5.00),
)
RADIAL_WINDOWS = (
    ("growth", 0.45, 1.50),
    ("post_growth", 2.00, 5.00),
    ("steady", 20.00, 30.00),
)
PROFILE_WINDOWS = (
    ("pre_onset", 0.15, 0.45),
    ("growth", 0.75, 1.20),
    ("steady", 20.00, 30.00),
)
PROFILE_FIELDS = (
    "electron_den",
    "ion_den",
    "electron_Temp",
    "electron_ud",
    "electron_vd",
    "electron_wd",
    "efx",
)
RNG_SEED = 20260826
TRIAD_SURROGATES = 499


@dataclass
class FourierData:
    time_us: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    early_time_us: np.ndarray
    steady_time_us: np.ndarray
    early: dict[str, np.ndarray]
    steady: dict[str, np.ndarray]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def contiguous_bounds(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, split)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def read_fourier_block(
    source: h5py.File,
    field_name: str,
    indices: np.ndarray,
    ny: int,
) -> np.ndarray:
    output = np.empty((len(indices), len(source["axes/x_m"]), MAX_MODE + 1), np.complex128)
    cursor = 0
    for first, stop in contiguous_bounds(indices):
        for start in range(first, stop, 64):
            end = min(start + 64, stop)
            values = np.asarray(source[f"fields/{field_name}"][start:end, :, :ny], dtype=np.float64)
            coefficient = np.fft.rfft(values, axis=2) / ny
            count = end - start
            output[cursor : cursor + count] = coefficient[..., : MAX_MODE + 1]
            cursor += count
    if cursor != len(indices):
        raise RuntimeError(f"Incomplete Fourier extraction for {field_name}: {cursor}/{len(indices)}")
    return output


def load_fourier_data() -> FourierData:
    with h5py.File(SOURCE_H5, "r") as source:
        time_us = np.asarray(source["axes/time_s"], dtype=np.float64) * 1.0e6
        x_m = np.asarray(source["axes/x_m"], dtype=np.float64)
        y_full = np.asarray(source["axes/y_m"], dtype=np.float64)
        # The final azimuthal point duplicates the periodic origin.
        ny = len(y_full) - 1
        y_m = y_full[:ny]
        early_index = np.flatnonzero(time_us <= EARLY_END_US + 1.0e-9)
        steady_index = np.flatnonzero(
            (time_us >= STEADY_START_US - 1.0e-9)
            & (time_us <= STEADY_END_US + 1.0e-9)
        )
        early = {
            field: read_fourier_block(source, field, early_index, ny)
            for field in ("efy", "electron_den")
        }
        steady = {
            field: read_fourier_block(source, field, steady_index, ny)
            for field in ("efy", "electron_den")
        }
    return FourierData(
        time_us=time_us,
        x_m=x_m,
        y_m=y_m,
        early_time_us=time_us[early_index],
        steady_time_us=time_us[steady_index],
        early=early,
        steady=steady,
    )


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1:
        return values.copy()
    left = width // 2
    right = width - left - 1
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def mode_amplitude(coefficient: np.ndarray, modes: np.ndarray | list[int] | int) -> np.ndarray:
    selected = np.atleast_1d(modes).astype(int)
    return np.sqrt(np.mean(np.sum(np.abs(coefficient[..., selected]) ** 2, axis=-1), axis=1))


def persistent_crossing(
    time_us: np.ndarray,
    values: np.ndarray,
    threshold: float,
    persistence_us: float,
    start_index: int = 0,
) -> float:
    dt = float(np.median(np.diff(time_us)))
    count = max(1, int(round(persistence_us / dt)))
    condition = values >= threshold
    for index in range(start_index, len(values) - count + 1):
        if np.all(condition[index : index + count]):
            return float(time_us[index])
    return float("nan")


def linear_fit(time_us: np.ndarray, log_amplitude: np.ndarray) -> dict:
    slope, intercept = np.polyfit(time_us, log_amplitude, 1)
    fitted = slope * time_us + intercept
    residual = log_amplitude - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((log_amplitude - np.mean(log_amplitude)) ** 2))
    r_squared = 1.0 - ss_res / max(ss_tot, 1.0e-300)
    return {
        "gamma_us_inv": float(slope),
        "gamma_s_inv": float(slope * 1.0e6),
        "r_squared": float(r_squared),
        "growth_ratio": float(np.exp(fitted[-1] - fitted[0])),
        "fit_start_us": float(time_us[0]),
        "fit_end_us": float(time_us[-1]),
        "fit_frames": len(time_us),
    }


def growth_sensitivity(
    label: str,
    time_us: np.ndarray,
    amplitude: np.ndarray,
    steady_reference: float,
) -> list[dict]:
    dt = float(np.median(np.diff(time_us)))
    floor = float(np.median(amplitude[time_us <= 0.15 + 1.0e-9]))
    span = steady_reference - floor
    output: list[dict] = []
    if span <= 0:
        return output
    for smooth_us in (0.075, 0.15, 0.30):
        smoothed = moving_average(amplitude, int(round(smooth_us / dt)))
        for upper_fraction in (0.20, 0.40, 0.60):
            lower_threshold = floor + 0.03 * span
            upper_threshold = floor + upper_fraction * span
            start_time = persistent_crossing(time_us, smoothed, lower_threshold, 0.075)
            if not math.isfinite(start_time):
                continue
            start = int(np.searchsorted(time_us, start_time))
            stop_time = persistent_crossing(
                time_us, smoothed, upper_threshold, 0.075, start_index=start
            )
            if not math.isfinite(stop_time):
                continue
            stop = int(np.searchsorted(time_us, stop_time)) + 1
            if stop - start < 10:
                continue
            values = np.log(np.maximum(smoothed[start:stop], np.finfo(float).tiny))
            row = {
                "component": label,
                "smooth_us": smooth_us,
                "lower_fraction": 0.03,
                "upper_fraction": upper_fraction,
                "initial_floor": floor,
                "steady_reference": steady_reference,
            }
            row.update(linear_fit(time_us[start:stop], values))
            output.append(row)
    return output


def chronology_and_growth(data: FourierData) -> tuple[list[dict], list[dict], dict]:
    early_ey = data.early["efy"]
    steady_ey = data.steady["efy"]
    steady_mode_power = np.mean(np.abs(steady_ey) ** 2, axis=(0, 1))
    early_search = (data.early_time_us >= 0.30) & (data.early_time_us <= 2.00)
    early_mode_power = np.mean(np.abs(early_ey[early_search]) ** 2, axis=(0, 1))
    long_n = int(LONG_MODES[np.argmax(steady_mode_power[LONG_MODES])])
    mtsi_n = int(MTSI_MODES[np.argmax(steady_mode_power[MTSI_MODES])])
    high_n = int(ECDI_MODES[np.argmax(early_mode_power[ECDI_MODES])])
    components = {
        f"long_n{long_n}": np.array([long_n]),
        f"mtsi_n{mtsi_n}": np.array([mtsi_n]),
        f"startup_high_n{high_n}": np.array([high_n]),
        "long_band_n1_3": LONG_MODES,
        "mtsi_band_n4_18": MTSI_MODES,
        "high_band_n20_48": ECDI_MODES,
    }
    chronology: list[dict] = []
    growth_rows: list[dict] = []
    dt = float(np.median(np.diff(data.early_time_us)))
    smooth_width = int(round(0.15 / dt))
    for label, modes in components.items():
        amplitude = mode_amplitude(early_ey, modes)
        steady = float(np.median(mode_amplitude(steady_ey, modes)))
        smoothed = moving_average(amplitude, smooth_width)
        floor = float(np.median(smoothed[data.early_time_us <= 0.15 + 1.0e-9]))
        span = steady - floor
        row = {
            "component": label,
            "modes": "-".join(str(int(mode)) for mode in modes),
            "initial_floor": floor,
            "steady_reference": steady,
            "initial_over_steady": floor / max(steady, 1.0e-300),
            "peak_time_0_5us": float(data.early_time_us[np.argmax(smoothed)]),
            "peak_over_steady": float(np.max(smoothed) / max(steady, 1.0e-300)),
        }
        for fraction in (0.10, 0.50):
            threshold = floor + fraction * span
            row[f"persistent_{int(100*fraction)}pct_time_us"] = (
                persistent_crossing(data.early_time_us, smoothed, threshold, 0.15)
                if span > 0
                else float("nan")
            )
        chronology.append(row)
        growth_rows.extend(
            growth_sensitivity(label, data.early_time_us, amplitude, steady)
        )
    metadata = {
        "long_mode_n": long_n,
        "mtsi_mode_n": mtsi_n,
        "startup_high_mode_n": high_n,
        "steady_mode_power": steady_mode_power,
        "early_mode_power": early_mode_power,
        "components": components,
    }
    return chronology, growth_rows, metadata


def dominant_signed_frequency(values: np.ndarray, dt_us: float) -> float:
    centered = values - np.mean(values)
    spectrum = np.fft.fft(centered * np.hanning(len(centered)))
    frequency = np.fft.fftfreq(len(centered), d=dt_us)
    valid = np.abs(frequency) >= 1.0 / (len(values) * dt_us)
    if not np.any(valid):
        return float("nan")
    return float(frequency[valid][np.argmax(np.abs(spectrum[valid]) ** 2)])


def local_mode_series(coefficient: np.ndarray) -> tuple[np.ndarray, int]:
    radial_power = np.mean(np.abs(coefficient) ** 2, axis=0)
    radial_index = int(np.argmax(radial_power))
    return coefficient[:, radial_index], radial_index


def difference_bicoherence_scan(
    coefficient: np.ndarray,
    time_us: np.ndarray,
    field_name: str,
    epoch: str,
    rng: np.random.Generator,
) -> list[dict]:
    dt = float(np.median(np.diff(time_us)))
    duration = float(time_us[-1] - time_us[0] + dt)
    resolution = 1.0 / duration
    centered = coefficient - np.mean(coefficient, axis=0, keepdims=True)
    output_mode = centered[..., 1]
    parent_a = np.arange(5, MAX_MODE + 1)
    parent_b = parent_a - 1
    products = np.stack(
        [centered[..., a] * np.conj(centered[..., b]) for a, b in zip(parent_a, parent_b)],
        axis=-1,
    )
    output_flat = output_mode.reshape(-1)
    products_flat = products.reshape(-1, len(parent_a))
    numerator = np.abs(
        np.mean(products_flat * np.conj(output_flat[:, None]), axis=0)
    ) ** 2
    product_power = np.mean(np.abs(products_flat) ** 2, axis=0)
    output_power = float(np.mean(np.abs(output_flat) ** 2))
    observed = numerator / np.maximum(product_power * output_power, 1.0e-300)

    phase = (
        np.angle(centered[..., parent_a])
        - np.angle(centered[..., parent_b])
        - np.angle(output_mode)[..., None]
    )
    weight = (
        np.abs(centered[..., parent_a])
        * np.abs(centered[..., parent_b])
        * np.abs(output_mode)[..., None]
    )
    phase_locking = np.abs(np.sum(weight * np.exp(1j * phase), axis=(0, 1))) / np.maximum(
        np.sum(weight, axis=(0, 1)), 1.0e-300
    )

    min_shift = max(2, int(round(0.30 / dt)))
    valid_shifts = np.arange(min_shift, len(time_us) - min_shift + 1)
    replace = len(valid_shifts) < TRIAD_SURROGATES
    shifts = rng.choice(valid_shifts, size=TRIAD_SURROGATES, replace=replace)
    null = np.empty((TRIAD_SURROGATES, len(parent_a)), dtype=np.float64)
    for draw, shift in enumerate(shifts):
        shifted = np.roll(output_mode, int(shift), axis=0).reshape(-1)
        local_num = np.abs(
            np.mean(products_flat * np.conj(shifted[:, None]), axis=0)
        ) ** 2
        null[draw] = local_num / np.maximum(product_power * output_power, 1.0e-300)
    max_null = np.max(null, axis=1)

    frequencies = np.empty(MAX_MODE + 1, dtype=np.float64)
    radial_indices = np.empty(MAX_MODE + 1, dtype=np.int64)
    for mode in range(1, MAX_MODE + 1):
        series, radial_index = local_mode_series(centered[..., mode])
        frequencies[mode] = dominant_signed_frequency(series, dt)
        radial_indices[mode] = radial_index
    output_frequency = frequencies[1]
    parent_support = np.sqrt(product_power)
    support_scale = max(float(np.max(parent_support)), 1.0e-300)
    rows: list[dict] = []
    for index, (a, b) in enumerate(zip(parent_a, parent_b)):
        predicted_frequency = frequencies[a] - frequencies[b]
        mismatch = abs(predicted_frequency - output_frequency)
        pair_p = float(
            (1 + np.count_nonzero(null[:, index] >= observed[index]))
            / (TRIAD_SURROGATES + 1)
        )
        family_p = float(
            (1 + np.count_nonzero(max_null >= observed[index]))
            / (TRIAD_SURROGATES + 1)
        )
        rows.append(
            {
                "field": field_name,
                "epoch": epoch,
                "start_us": float(time_us[0]),
                "end_us": float(time_us[-1]),
                "parent_a_n": int(a),
                "parent_b_n": int(b),
                "output_n": 1,
                "bicoherence_squared": float(observed[index]),
                "weighted_phase_locking": float(phase_locking[index]),
                "pairwise_shift_p": pair_p,
                "familywise_max_shift_p": family_p,
                "familywise_null_q95": float(np.quantile(max_null, 0.95)),
                "parent_support": float(parent_support[index]),
                "parent_support_fraction_of_max": float(parent_support[index] / support_scale),
                "parent_a_frequency_mhz": float(frequencies[a]),
                "parent_b_frequency_mhz": float(frequencies[b]),
                "difference_frequency_mhz": float(predicted_frequency),
                "long_frequency_mhz": float(output_frequency),
                "frequency_mismatch_mhz": float(mismatch),
                "frequency_resolution_mhz": float(resolution),
                "frequency_match_within_one_bin": bool(mismatch <= resolution),
                "parent_a_radial_index": int(radial_indices[a]),
                "parent_b_radial_index": int(radial_indices[b]),
                "long_radial_index": int(radial_indices[1]),
                "surrogates": TRIAD_SURROGATES,
            }
        )
    return rows


def select_and_confirm_triads(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for field_name in ("efy", "electron_den"):
        discovery = [
            row for row in rows if row["field"] == field_name and row["epoch"] == "discovery"
        ]
        matched = [row for row in discovery if row["frequency_match_within_one_bin"]]
        candidates = matched if matched else discovery
        selected = max(candidates, key=lambda row: row["bicoherence_squared"])
        a = selected["parent_a_n"]
        b = selected["parent_b_n"]
        for epoch in ("discovery", "confirmation", "late"):
            row = next(
                item
                for item in rows
                if item["field"] == field_name
                and item["epoch"] == epoch
                and item["parent_a_n"] == a
                and item["parent_b_n"] == b
            )
            copied = dict(row)
            copied["selected_in_epoch"] = "discovery"
            copied["selection_rule"] = "max bicoherence among one-bin frequency matches"
            copied["confirmatory"] = epoch != "discovery"
            output.append(copied)
    return output


def epochwise_best_triads(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for field_name in ("efy", "electron_den"):
        for epoch, _, _ in TRIAD_WINDOWS:
            local = [
                row
                for row in rows
                if row["field"] == field_name and row["epoch"] == epoch
            ]
            matched = [row for row in local if row["frequency_match_within_one_bin"]]
            candidates = matched if matched else local
            selected = max(candidates, key=lambda row: row["bicoherence_squared"])
            copied = dict(selected)
            copied["selection_rule"] = "epoch-wise max bicoherence among one-bin frequency matches"
            copied["exploratory"] = True
            output.append(copied)
    return output


def pod_structure(values: np.ndarray, x_m: np.ndarray, label: str) -> tuple[dict, np.ndarray]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    vector = vh[0]
    power = np.abs(vector) ** 2
    power /= max(float(np.sum(power)), 1.0e-300)
    radius_cm = x_m * 100.0
    centroid = float(np.sum(power * radius_cm))
    width = float(np.sqrt(np.sum(power * (radius_cm - centroid) ** 2)))
    edge_count = max(1, int(round(0.10 * len(x_m))))
    edge_fraction = float(np.sum(power[:edge_count]) + np.sum(power[-edge_count:]))
    uniform = np.ones(len(vector), dtype=np.complex128)
    uniform_overlap = vector_overlap(vector, uniform)
    return (
        {
            "epoch": label,
            "frames": len(values),
            "pod1_energy_fraction": float(singular[0] ** 2 / np.sum(singular**2)),
            "radial_centroid_cm": centroid,
            "radial_width_cm": width,
            "outer_20pct_edge_power_fraction": edge_fraction,
            "radially_uniform_overlap": uniform_overlap,
        },
        vector,
    )


def vector_overlap(left: np.ndarray, right: np.ndarray) -> float:
    numerator = abs(np.vdot(left, right)) ** 2
    denominator = float(np.vdot(left, left).real * np.vdot(right, right).real)
    return float(numerator / max(denominator, 1.0e-300))


def exact_dmd(values: np.ndarray, dt_us: float) -> tuple[dict, np.ndarray]:
    x = values[:-1].T
    y = values[1:].T
    u, singular, vh = np.linalg.svd(x, full_matrices=False)
    energy = np.cumsum(singular**2) / max(float(np.sum(singular**2)), 1.0e-300)
    rank = int(min(20, np.searchsorted(energy, 0.999) + 1, len(singular)))
    ur = u[:, :rank]
    sr = singular[:rank]
    vr = vh[:rank].conj().T
    atilde = ur.conj().T @ y @ vr @ np.diag(1.0 / np.maximum(sr, 1.0e-300))
    eigenvalue, eigenvector = np.linalg.eig(atilde)
    mode = y @ vr @ np.diag(1.0 / np.maximum(sr, 1.0e-300)) @ eigenvector
    amplitude = np.linalg.lstsq(mode, values[0], rcond=None)[0]
    continuous = np.log(eigenvalue) / dt_us
    frequency = np.imag(continuous) / (2.0 * np.pi)
    growth = np.real(continuous)
    score = np.abs(amplitude) * np.linalg.norm(mode, axis=0)
    candidate = (np.abs(frequency) >= 0.2) & (np.abs(frequency) <= 5.0)
    if not np.any(candidate):
        candidate = np.ones_like(frequency, dtype=bool)
    index = int(np.flatnonzero(candidate)[np.argmax(score[candidate])])
    vector = mode[:, index]
    return (
        {
            "svd_rank": rank,
            "retained_energy_fraction": float(energy[rank - 1]),
            "selected_eigenvalue_real": float(eigenvalue[index].real),
            "selected_eigenvalue_imag": float(eigenvalue[index].imag),
            "selected_frequency_mhz": float(frequency[index]),
            "selected_growth_us_inv": float(growth[index]),
            "selected_score_fraction": float(score[index] / max(np.sum(score), 1.0e-300)),
        },
        vector,
    )


def radial_structure(data: FourierData) -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    rows: list[dict] = []
    vectors: dict[str, np.ndarray] = {}
    for label, start, stop in RADIAL_WINDOWS:
        if stop <= EARLY_END_US:
            time = data.early_time_us
            coefficient = data.early["efy"]
        else:
            time = data.steady_time_us
            coefficient = data.steady["efy"]
        mask = (time >= start - 1.0e-9) & (time <= stop + 1.0e-9)
        row, vector = pod_structure(coefficient[mask, :, 1], data.x_m, label)
        row.update({"start_us": start, "end_us": stop, "mode_n": 1})
        rows.append(row)
        vectors[label] = vector
    overlaps = {
        "growth_to_post_growth_pod1_overlap": vector_overlap(
            vectors["growth"], vectors["post_growth"]
        ),
        "growth_to_steady_pod1_overlap": vector_overlap(vectors["growth"], vectors["steady"]),
        "post_growth_to_steady_pod1_overlap": vector_overlap(
            vectors["post_growth"], vectors["steady"]
        ),
    }
    growth_mask = (
        (data.early_time_us >= RADIAL_WINDOWS[0][1] - 1.0e-9)
        & (data.early_time_us <= RADIAL_WINDOWS[0][2] + 1.0e-9)
    )
    dt = float(np.median(np.diff(data.early_time_us[growth_mask])))
    dmd, dmd_vector = exact_dmd(data.early["efy"][growth_mask, :, 1], dt)
    dmd["dmd_to_steady_pod1_overlap"] = vector_overlap(dmd_vector, vectors["steady"])
    vectors["growth_dmd"] = dmd_vector
    return rows, {**overlaps, **dmd}, vectors


def average_profile(dataset: h5py.Dataset, indices: np.ndarray, ny: int) -> np.ndarray:
    total = np.zeros(dataset.shape[1], dtype=np.float64)
    samples = 0
    for first, stop in contiguous_bounds(indices):
        for start in range(first, stop, 64):
            end = min(start + 64, stop)
            values = np.asarray(dataset[start:end, :, :ny], dtype=np.float64)
            total += np.sum(values, axis=(0, 2))
            samples += values.shape[0] * values.shape[2]
    return total / max(samples, 1)


def background_profiles(data: FourierData) -> list[dict]:
    rows: list[dict] = []
    with h5py.File(SOURCE_H5, "r") as source:
        ny = len(source["axes/y_m"]) - 1
        for epoch, start, stop in PROFILE_WINDOWS:
            indices = np.flatnonzero(
                (data.time_us >= start - 1.0e-9) & (data.time_us <= stop + 1.0e-9)
            )
            profiles = {
                field: average_profile(source[f"fields/{field}"], indices, ny)
                for field in PROFILE_FIELDS
            }
            for radial_index, radius in enumerate(data.x_m):
                row = {
                    "epoch": epoch,
                    "start_us": start,
                    "end_us": stop,
                    "radial_index": radial_index,
                    "radius_m": float(radius),
                    "radius_cm": float(radius * 100.0),
                }
                row.update({field: float(values[radial_index]) for field, values in profiles.items()})
                rows.append(row)
    return rows


def plot_chronology(
    data: FourierData,
    metadata: dict,
    chronology: list[dict],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    mode_specs = (
        ("long", [metadata["long_mode_n"]], "#0072B2"),
        ("MTSI", [metadata["mtsi_mode_n"]], "#D55E00"),
        ("startup high-n", [metadata["startup_high_mode_n"]], "#009E73"),
    )
    band_specs = (
        ("long n=1-3", LONG_MODES, "#56B4E9"),
        ("MTSI n=4-18", MTSI_MODES, "#E69F00"),
        ("high n=20-48", ECDI_MODES, "#CC79A7"),
    )
    steady = data.steady["efy"]
    for axis, specs in zip(axes, (mode_specs, band_specs)):
        for label, modes, color in specs:
            amplitude = mode_amplitude(data.early["efy"], modes)
            reference = float(np.median(mode_amplitude(steady, modes)))
            axis.plot(data.early_time_us, amplitude / max(reference, 1.0e-300), label=label, color=color)
        axis.axhline(1.0, color="0.35", linestyle="--", linewidth=1, label="steady median")
        axis.set_yscale("log")
        axis.set_ylabel("amplitude / steady median")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", frameon=True)
    axes[0].set_title("B30 initial mode growth and chronology")
    axes[-1].set_xlabel("time (us)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_triad_confirmation(selected: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    epochs = ["discovery", "confirmation", "late"]
    x = np.arange(len(epochs))
    for axis, field_name in zip(axes, ("efy", "electron_den")):
        local = [row for row in selected if row["field"] == field_name]
        values = [next(row for row in local if row["epoch"] == epoch) for epoch in epochs]
        axis.plot(x, [row["bicoherence_squared"] for row in values], "o-", label="bicoherence squared")
        axis.plot(x, [row["familywise_null_q95"] for row in values], "s--", label="familywise null q95")
        axis.plot(x, [row["weighted_phase_locking"] for row in values], "^:", label="phase locking")
        pair = f"n={values[0]['parent_a_n']}-{values[0]['parent_b_n']} -> 1"
        axis.set_title(f"{field_name}: discovery-selected {pair}")
        axis.set_ylabel("coupling statistic")
        axis.set_xticks(x, epochs)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_epochwise_triads(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    epochs = ["discovery", "confirmation", "late"]
    x = np.arange(len(epochs))
    for axis, field_name in zip(axes, ("efy", "electron_den")):
        local = [row for row in rows if row["field"] == field_name]
        ordered = [next(row for row in local if row["epoch"] == epoch) for epoch in epochs]
        values = [row["bicoherence_squared"] for row in ordered]
        null = [row["familywise_null_q95"] for row in ordered]
        labels = [f"{row['parent_a_n']}-{row['parent_b_n']}" for row in ordered]
        axis.plot(x, values, "o-", label="epoch-wise best bicoherence squared")
        axis.plot(x, null, "s--", label="familywise null q95")
        for index, (value, label) in enumerate(zip(values, labels)):
            axis.annotate(label, (index, value), xytext=(0, 8), textcoords="offset points", ha="center")
        axis.set_title(f"{field_name}: best frequency-matched parent pair changes by epoch")
        axis.set_ylabel("coupling statistic")
        axis.set_xticks(x, epochs)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aligned_vector(vector: np.ndarray) -> np.ndarray:
    index = int(np.argmax(np.abs(vector)))
    return vector * np.exp(-1j * np.angle(vector[index]))


def plot_radial_structure(
    x_m: np.ndarray,
    vectors: dict[str, np.ndarray],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = {"growth": "#0072B2", "post_growth": "#D55E00", "steady": "#009E73", "growth_dmd": "#CC79A7"}
    for label, vector in vectors.items():
        local = aligned_vector(vector)
        amplitude = np.abs(local) / max(float(np.max(np.abs(local))), 1.0e-300)
        phase = np.unwrap(np.angle(local))
        phase -= phase[int(np.argmax(np.abs(local)))]
        axes[0].plot(x_m * 100.0, amplitude, label=label, color=colors[label])
        visible = amplitude >= 0.10
        axes[1].plot(x_m[visible] * 100.0, phase[visible], label=label, color=colors[label])
    axes[0].set_ylabel("normalized |radial mode|")
    axes[1].set_ylabel("relative phase (rad)")
    axes[1].set_xlabel("radial coordinate (cm)")
    axes[0].set_title("B30 n=1 radial structure")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_background_profiles(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fields = ("electron_den", "electron_Temp", "electron_ud", "electron_wd")
    labels = ("electron density", "electron scalar Temp", "electron_ud", "electron_wd")
    epochs = ("pre_onset", "growth", "steady")
    colors = {"pre_onset": "#0072B2", "growth": "#D55E00", "steady": "#009E73"}
    for axis, field, label in zip(axes.flat, fields, labels):
        for epoch in epochs:
            local = [row for row in rows if row["epoch"] == epoch]
            axis.plot(
                [row["radius_cm"] for row in local],
                [row[field] for row in local],
                label=epoch,
                color=colors[epoch],
            )
        axis.set_title(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", frameon=True)
    axes[1, 0].set_xlabel("radial coordinate (cm)")
    axes[1, 1].set_xlabel("radial coordinate (cm)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not SOURCE_H5.exists():
        raise FileNotFoundError(SOURCE_H5)
    rng = np.random.default_rng(RNG_SEED)
    data = load_fourier_data()
    chronology, growth_rows, metadata = chronology_and_growth(data)

    triad_rows: list[dict] = []
    for field_name in ("efy", "electron_den"):
        for epoch, start, stop in TRIAD_WINDOWS:
            mask = (
                (data.early_time_us >= start - 1.0e-9)
                & (data.early_time_us <= stop + 1.0e-9)
            )
            triad_rows.extend(
                difference_bicoherence_scan(
                    data.early[field_name][mask],
                    data.early_time_us[mask],
                    field_name,
                    epoch,
                    rng,
                )
            )
    selected_triads = select_and_confirm_triads(triad_rows)
    epochwise_triads = epochwise_best_triads(triad_rows)

    radial_rows, radial_summary, radial_vectors = radial_structure(data)
    profile_rows = background_profiles(data)

    dt_us = float(np.median(np.diff(data.early_time_us)))
    ly_m = float(len(data.y_m) * np.median(np.diff(data.y_m)))
    chronology_map = {row["component"]: row for row in chronology}
    long_key = f"long_n{metadata['long_mode_n']}"
    mtsi_key = f"mtsi_n{metadata['mtsi_mode_n']}"
    high_key = f"startup_high_n{metadata['startup_high_mode_n']}"
    long_onset = chronology_map[long_key]["persistent_10pct_time_us"]
    mtsi_onset = chronology_map[mtsi_key]["persistent_10pct_time_us"]
    high_onset = chronology_map[high_key]["persistent_10pct_time_us"]

    long_growth = [row for row in growth_rows if row["component"] == long_key]
    long_growth_median = float(np.median([row["gamma_us_inv"] for row in long_growth])) if long_growth else float("nan")
    long_growth_r2_median = float(np.median([row["r_squared"] for row in long_growth])) if long_growth else float("nan")
    confirmatory_rows = [row for row in selected_triads if row["confirmatory"]]
    confirmed_beat = any(
        row["epoch"] == "confirmation"
        and row["familywise_max_shift_p"] <= 0.05
        and row["frequency_match_within_one_bin"]
        for row in confirmatory_rows
    )
    epochwise_pairs = {
        field_name: [
            (row["parent_a_n"], row["parent_b_n"])
            for row in epochwise_triads
            if row["field"] == field_name
        ]
        for field_name in ("efy", "electron_den")
    }
    stable_epochwise_pair = any(
        len(set(pairs)) == 1 for pairs in epochwise_pairs.values()
    )
    significant_dynamic_network = any(
        row["familywise_max_shift_p"] <= 0.05
        and row["frequency_match_within_one_bin"]
        for row in epochwise_triads
    )

    evidence = {
        "independent_finite_time_growth": {
            "positive_growth_fit": bool(long_growth_median > 0 and long_growth_r2_median >= 0.90),
            "median_gamma_us_inv": long_growth_median,
            "median_r_squared": long_growth_r2_median,
            "long_10pct_onset_us": long_onset,
            "mtsi_10pct_onset_us": mtsi_onset,
            "startup_high_10pct_onset_us": high_onset,
            "long_precedes_both_reference_modes": bool(
                math.isfinite(long_onset)
                and long_onset <= min(mtsi_onset, high_onset) + dt_us
            ),
            "interpretation": "Finite-time exponential-looking growth is necessary but not sufficient for a linear eigenmode because the startup background evolves.",
        },
        "specific_difference_frequency_generation": {
            "discovery_selected_pairs": [
                {
                    "field": row["field"],
                    "parent_a_n": row["parent_a_n"],
                    "parent_b_n": row["parent_b_n"],
                }
                for row in selected_triads
                if row["epoch"] == "discovery"
            ],
            "confirmed_in_held_out_epoch": confirmed_beat,
            "same_epochwise_best_pair_across_all_epochs": stable_epochwise_pair,
            "epochwise_best_pairs": epochwise_pairs,
            "at_least_one_exploratory_epochwise_pair_is_familywise_significant": significant_dynamic_network,
            "interpretation": "Confirmation requires frequency matching and familywise-significant coupling in the fixed 1.5-3.0 us epoch.",
        },
        "global_or_box_scale": {
            "dominant_long_mode_n": metadata["long_mode_n"],
            "azimuthal_length_m": ly_m,
            "long_wavelength_m": ly_m / metadata["long_mode_n"],
            "wavelength_over_domain_length": 1.0 / metadata["long_mode_n"],
            "n1_is_box_scale": metadata["long_mode_n"] == 1,
            "growth_to_steady_radial_overlap": radial_summary[
                "growth_to_steady_pod1_overlap"
            ],
            "interpretation": "A dominant n=1 mode is system-scale, but one domain length cannot distinguish a physical global eigenmode from box selection.",
        },
    }

    write_csv(OUTPUT / "onset_chronology.csv", chronology)
    write_csv(OUTPUT / "finite_time_growth_sensitivity.csv", growth_rows)
    write_csv(OUTPUT / "difference_triad_all_pairs.csv", triad_rows)
    write_csv(OUTPUT / "difference_triad_discovery_confirmation.csv", selected_triads)
    write_csv(OUTPUT / "difference_triad_epochwise_best_exploratory.csv", epochwise_triads)
    write_csv(OUTPUT / "n1_radial_structure.csv", radial_rows)
    write_csv(OUTPUT / "background_profiles_for_linear_stability.csv", profile_rows)

    plot_chronology(data, metadata, chronology, OUTPUT / "b30_mode_onset_chronology.png")
    plot_triad_confirmation(
        selected_triads, OUTPUT / "b30_difference_triad_confirmation.png"
    )
    plot_epochwise_triads(
        epochwise_triads, OUTPUT / "b30_difference_triad_epochwise_best.png"
    )
    plot_radial_structure(
        data.x_m, radial_vectors, OUTPUT / "b30_n1_radial_eigenstructure.png"
    )
    plot_background_profiles(
        profile_rows, OUTPUT / "b30_background_profiles_for_linear_stability.png"
    )

    with h5py.File(OUTPUT / "b30_longwave_compact_diagnostics.h5", "w") as target:
        target.create_dataset("early_time_us", data=data.early_time_us)
        target.create_dataset("steady_time_us", data=data.steady_time_us)
        target.create_dataset("x_m", data=data.x_m)
        for field_name in ("efy", "electron_den"):
            target.create_dataset(
                f"early_{field_name}_mode_amplitude",
                data=np.sqrt(np.mean(np.abs(data.early[field_name]) ** 2, axis=1)),
            )
            target.create_dataset(
                f"steady_{field_name}_mode_amplitude",
                data=np.sqrt(np.mean(np.abs(data.steady[field_name]) ** 2, axis=1)),
            )
            target.create_dataset(
                f"early_{field_name}_n1_radial", data=data.early[field_name][..., 1]
            )
            target.create_dataset(
                f"steady_{field_name}_n1_radial", data=data.steady[field_name][..., 1]
            )
        target.attrs["source_h5"] = str(SOURCE_H5)
        target.attrs["azimuthal_duplicate_endpoint_excluded"] = True

    summary = {
        "status": "PASS",
        "question": "If not a simple inverse cascade, what excites the B30 long-wavelength mode?",
        "shared_design": SHARE_URL,
        "source_h5": str(SOURCE_H5),
        "sampling": {
            "frame_interval_ns": dt_us * 1000.0,
            "long_mode_n": metadata["long_mode_n"],
            "long_mode_wavelength_mm": ly_m * 1000.0 / metadata["long_mode_n"],
            "azimuthal_domain_length_mm": ly_m * 1000.0,
        },
        "selected_modes": {
            "long_n": metadata["long_mode_n"],
            "steady_mtsi_n": metadata["mtsi_mode_n"],
            "startup_high_n": metadata["startup_high_mode_n"],
        },
        "chronology": chronology,
        "finite_time_growth_sensitivity": growth_rows,
        "selected_difference_triads": selected_triads,
        "epochwise_best_difference_triads": epochwise_triads,
        "radial_structure": radial_rows,
        "radial_and_dmd_summary": radial_summary,
        "evidence": evidence,
        "limitations": [
            "Finite-time startup growth is not a linear eigenvalue because the PIC background evolves.",
            "Bicoherence and triad phase locking do not provide signed energy-transfer direction.",
            "The 15 ns cadence resolves the approximately 1.5 MHz long mode but only coarsely samples 20 MHz-scale parents.",
            "A domain-length scan is required to distinguish a physical global eigenmode from numerical box selection.",
            "A candidate dispersion relation must be solved against the extracted background before naming the instability.",
        ],
        "next_targeted_tests": [
            "Compare candidate linear dispersion roots with PIC n, frequency, growth and radial structure.",
            "Repeat B30 with Ly=1.5 and 2.0 times the current length while keeping dy fixed.",
            "If a fixed parent triad confirms, change the parent-mode content rather than only increasing diagnostic detail.",
        ],
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    confirmation_lines = []
    for row in selected_triads:
        if row["epoch"] == "confirmation":
            confirmation_lines.append(
                f"- {row['field']}: discovery-selected n={row['parent_a_n']}-{row['parent_b_n']}->1, "
                f"confirmation b2={row['bicoherence_squared']:.3f}, familywise p={row['familywise_max_shift_p']:.3f}, "
                f"frequency mismatch={row['frequency_mismatch_mhz']:.3f} MHz "
                f"(resolution {row['frequency_resolution_mhz']:.3f} MHz)."
            )
    readme = f"""# B30 long-wavelength-mode origin diagnostic

This is an existing-data diagnostic for `{CASE_NAME}`. It tests finite-time
growth, chronology, adjacent-mode difference-frequency coupling, and radial
eigenstructure. It does not claim signed spectral energy transfer.

## Main numerical facts

- Dominant long mode: n={metadata['long_mode_n']} (wavelength/domain length = {1.0 / metadata['long_mode_n']:.3f}).
- Steady MTSI-side reference mode: n={metadata['mtsi_mode_n']}.
- Strongest startup high-n mode over 0.3--2.0 us: n={metadata['startup_high_mode_n']}.
- Long n={metadata['long_mode_n']} 10% onset: {long_onset:.3f} us.
- MTSI n={metadata['mtsi_mode_n']} 10% onset: {mtsi_onset:.3f} us.
- Startup high-n n={metadata['startup_high_mode_n']} 10% onset: {high_onset:.3f} us.
- Long-mode median finite-time gamma: {long_growth_median:.3f} us^-1; median R2={long_growth_r2_median:.3f}.
- Growth-to-steady radial POD overlap: {radial_summary['growth_to_steady_pod1_overlap']:.3f}.
- Post-growth-to-steady radial POD overlap: {radial_summary['post_growth_to_steady_pod1_overlap']:.3f}.
- Growth-window DMD frequency: {radial_summary['selected_frequency_mhz']:.3f} MHz.

## Discovery/confirmation triads

{chr(10).join(confirmation_lines)}

## Interpretation guardrails

- Exponential-looking startup growth is compatible with, but does not prove, a primary linear instability.
- A specific beat explanation requires a discovery-selected pair to retain frequency matching and familywise-significant coupling in the confirmation epoch.
- Epoch-wise significant pairs may still indicate a changing nonlinear network; because they are selected within each epoch, they are exploratory and are not a fixed causal parent pair.
- n=1 is exactly system-scale here. Distinguishing a physical global mode from a box-selected mode requires changing Ly at fixed dy.
- The next linear-stability calculation can use `background_profiles_for_linear_stability.csv`.

## Files

- `b30_mode_onset_chronology.png`
- `b30_difference_triad_confirmation.png`
- `b30_difference_triad_epochwise_best.png`
- `b30_n1_radial_eigenstructure.png`
- `b30_background_profiles_for_linear_stability.png`
- `onset_chronology.csv`
- `finite_time_growth_sensitivity.csv`
- `difference_triad_all_pairs.csv`
- `difference_triad_discovery_confirmation.csv`
- `difference_triad_epochwise_best_exploratory.csv`
- `n1_radial_structure.csv`
- `background_profiles_for_linear_stability.csv`
- `b30_longwave_compact_diagnostics.h5`
- `analysis_summary.json`
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"PASS: wrote {OUTPUT}")


if __name__ == "__main__":
    main()
