#!/usr/bin/env python3
"""Evaluate bidirectional RadAz regime transfer without saving prediction tensors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = (
    ROOT.parent
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
)
WORKDIR = ROOT / "workdirs" / "2D_RadAz"
OUTPUT = WORKDIR / "compare_radaz_regime_generalization_direct10"
CONFIG = ROOT / "configs" / "custom" / "pepapic" / "SimVP_gSTA_radaz_direct.py"
CHANNELS = ("electron_den", "ion_den", "phi")
CHANNEL_LABELS = {
    "electron_den": "Electron density",
    "ion_den": "Ion density",
    "phi": "Potential",
    "ey": "Azimuthal electric field",
}
DATA_ONLY_MODEL_SPECS = {
    "low_E10_E20": {
        "label": "Low-E model (E10+E20)",
        "source_E": (10.0, 20.0),
        "manifest": WORKDIR
        / "radaz_regime_generalization_manifests"
        / "radaz_low_source_manifest.json",
        "experiment": WORKDIR
        / "radaz_bx20mt_lowE_E10_E20_mixed_direct10_sourcepool_noclip_disjoint811_bs1_100ep",
        "final_target": 40.0,
    },
    "high_E30_E40": {
        "label": "High-E model (E30+E40)",
        "source_E": (30.0, 40.0),
        "manifest": WORKDIR
        / "radaz_regime_generalization_manifests"
        / "radaz_high_source_manifest.json",
        "experiment": WORKDIR
        / "radaz_bx20mt_highE_E30_E40_mixed_direct10_sourcepool_noclip_disjoint811_bs1_100ep",
        "final_target": 10.0,
    },
}
SPECTRAL_MODEL_SPECS = {
    "low_E10_E20": {
        "label": "Low-E spectral model (E10+E20)",
        "source_E": (10.0, 20.0),
        "manifest": WORKDIR
        / "radaz_regime_generalization_manifests"
        / "radaz_low_source_manifest.json",
        "experiment": WORKDIR
        / (
            "radaz_bx20mt_lowE_E10_E20_mixed_direct10_sourcepool_noclip_"
            "disjoint811_bs1_spectral_full_50ep"
        ),
        "final_target": 40.0,
    },
    "high_E30_E40": {
        "label": "High-E spectral model (E30+E40)",
        "source_E": (30.0, 40.0),
        "manifest": WORKDIR
        / "radaz_regime_generalization_manifests"
        / "radaz_high_source_manifest.json",
        "experiment": WORKDIR
        / (
            "radaz_bx20mt_highE_E30_E40_mixed_direct10_sourcepool_noclip_"
            "disjoint811_bs1_spectral_full_50ep"
        ),
        "final_target": 10.0,
    },
}
MODEL_SPECS = DATA_ONLY_MODEL_SPECS
E_VALUES = (10.0, 20.0, 22.5, 25.0, 30.0, 40.0)
VALID_H = 257
VALID_W = 256
MODEL_H = 260
MODEL_W = 256
PRE = 10
AFT = 10
FRAME_NS = 15.0
B_T = 0.020
BANDS = {
    "MTSI": np.arange(1, 7, dtype=int),
    "ECDI": np.arange(9, 22, dtype=int),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model-family",
        choices=("data_only", "spectral_full_50ep"),
        default="data_only",
        help="Checkpoint pair to evaluate with the common transfer protocol.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--skip-calibrated",
        action="store_true",
        help="Skip input-window mean/std calibration on the two final targets.",
    )
    return parser.parse_args()


def token(ez: float) -> str:
    return str(int(ez)) if float(ez).is_integer() else str(ez)


def case_name(ez: float) -> str:
    return f"2D_RadAz_Xe1p_Bx20mT_Ez{token(ez)}kVm_dt15ps_out15ns"


def case_h5(ez: float) -> Path:
    name = case_name(ez)
    return RESULTS / name / name / "analysis_fields_uncompressed.h5"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(spec: dict, device: torch.device) -> torch.nn.Module:
    from openstl.models.simvp_model import SimVP_Model

    checkpoint = spec["experiment"] / "checkpoints" / "best.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = runpy.run_path(str(CONFIG))
    model = SimVP_Model(
        in_shape=(PRE, len(CHANNELS), MODEL_H, MODEL_W),
        hid_S=int(cfg.get("hid_S", 64)),
        hid_T=int(cfg.get("hid_T", 512)),
        N_S=int(cfg.get("N_S", 4)),
        N_T=int(cfg.get("N_T", 8)),
        model_type=cfg.get("model_type", "gSTA"),
        drop_path=float(cfg.get("drop_path", 0.0)),
        spatio_kernel_enc=int(cfg.get("spatio_kernel_enc", 3)),
        spatio_kernel_dec=int(cfg.get("spatio_kernel_dec", 3)),
        aft_seq_length=AFT,
        simvp_direct_aft_seq=bool(cfg.get("simvp_direct_aft_seq", True)),
    )
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = loaded.get("state_dict", loaded)
    state = {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint}: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return model.to(device).eval()


def read_test_segment(ez: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = case_h5(ez)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        times = np.asarray(handle["axes/time_s"][1800:2001], dtype=np.float64)
        x_m = np.asarray(handle["axes/x_m"][:VALID_H], dtype=np.float64)
        y_m = np.asarray(handle["axes/y_m"][:VALID_W], dtype=np.float64)
        data = np.empty((201, len(CHANNELS), VALID_H, VALID_W), dtype=np.float32)
        for channel_index, channel in enumerate(CHANNELS):
            data[:, channel_index] = np.asarray(
                handle[f"fields/{channel}"][1800:2001, :VALID_H, :VALID_W],
                dtype=np.float32,
            )
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite test data in {path}")
    return data, times, x_m, y_m


def source_reference_moments(spec: dict, chunk: int = 16) -> dict:
    cache = spec["experiment"] / "source_reference_moments.json"
    if cache.is_file():
        result = load_json(cache)
        if result.get("source_Ez_kVm") == list(spec["source_E"]):
            return result

    total = np.zeros(len(CHANNELS), dtype=np.float64)
    total_sq = np.zeros(len(CHANNELS), dtype=np.float64)
    count = np.zeros(len(CHANNELS), dtype=np.int64)
    for ez in spec["source_E"]:
        with h5py.File(case_h5(ez), "r") as handle:
            for start in range(0, 1600, chunk):
                stop = min(start + chunk, 1600)
                for channel_index, channel in enumerate(CHANNELS):
                    values = np.asarray(
                        handle[f"fields/{channel}"][start:stop, :VALID_H, :VALID_W],
                        dtype=np.float64,
                    )
                    total[channel_index] += np.sum(values, dtype=np.float64)
                    total_sq[channel_index] += np.sum(values * values, dtype=np.float64)
                    count[channel_index] += values.size
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 0.0)
    std = np.sqrt(variance)
    if np.any(std <= 0.0):
        raise ValueError(f"Invalid source std for {spec['label']}: {std}")
    result = {
        "source_Ez_kVm": list(spec["source_E"]),
        "frames_per_case": 1600,
        "channels": list(CHANNELS),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count.tolist(),
    }
    cache.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def pad_normalized(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    normalized = (values.astype(np.float64) - low[None, :, None, None]) / (
        high - low
    )[None, :, None, None]
    output = np.empty((len(values), len(CHANNELS), MODEL_H, MODEL_W), dtype=np.float32)
    output[:, :, :VALID_H, :VALID_W] = normalized.astype(np.float32)
    output[:, :, VALID_H:, :VALID_W] = output[:, :, VALID_H - 1 : VALID_H, :VALID_W]
    return output


def spatial_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    af = a.reshape(len(a), -1).astype(np.float64)
    bf = b.reshape(len(b), -1).astype(np.float64)
    af -= np.mean(af, axis=1, keepdims=True)
    bf -= np.mean(bf, axis=1, keepdims=True)
    numerator = np.sum(af * bf, axis=1)
    denominator = np.sqrt(np.sum(af * af, axis=1) * np.sum(bf * bf, axis=1))
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )


def wrapped_angle_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (np.angle(a) - np.angle(b)))))


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 2 or np.std(a[mask]) == 0.0 or np.std(b[mask]) == 0.0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def complex_coherence(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    denominator = math.sqrt(float(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2)))
    return float(np.abs(np.sum(a * np.conj(b))) / denominator) if denominator else float("nan")


def observables(values: np.ndarray, x_mask: np.ndarray, dy_m: float) -> dict:
    ne = np.asarray(values[:, 0], dtype=np.float64)
    phi = np.asarray(values[:, 2], dtype=np.float64)
    ey = -(np.roll(phi, -1, axis=-1) - np.roll(phi, 1, axis=-1)) / (2.0 * dy_m)
    ne = ne[:, x_mask]
    phi = phi[:, x_mask]
    ey_region = ey[:, x_mask]
    dne = ne - np.mean(ne, axis=-1, keepdims=True)
    dphi = phi - np.mean(phi, axis=-1, keepdims=True)
    dey = ey_region - np.mean(ey_region, axis=-1, keepdims=True)
    ne_fft = np.fft.rfft(dne, axis=-1, norm="forward")
    phi_fft = np.fft.rfft(dphi, axis=-1, norm="forward")
    ey_fft = np.fft.rfft(dey, axis=-1, norm="forward")
    cross_by_mode = np.mean(ne_fft * np.conj(ey_fft), axis=1)
    phi_power_by_mode = np.mean(np.abs(phi_fft) ** 2, axis=1)
    amplitude = []
    cross = []
    transport = []
    for modes in BANDS.values():
        amplitude.append(np.sqrt(np.sum(phi_power_by_mode[:, modes], axis=-1)))
        band_cross = np.sum(cross_by_mode[:, modes], axis=-1)
        cross.append(band_cross)
        transport.append(-2.0 * np.real(band_cross) / B_T)
    return {
        "ey": ey,
        "amplitude": np.stack(amplitude, axis=-1),
        "cross": np.stack(cross, axis=-1),
        "transport": np.stack(transport, axis=-1),
    }


def role_for(spec: dict, ez: float) -> str:
    if ez in spec["source_E"]:
        return "source_in_domain"
    if ez == spec["final_target"]:
        return "opposite_regime_final"
    return "intermediate_diagnostic"


@torch.inference_mode()
def evaluate_case(
    model: torch.nn.Module,
    spec_key: str,
    spec: dict,
    ez: float,
    variant: str,
    device: torch.device,
    progress_every: int,
    source_moments: dict | None = None,
) -> dict:
    manifest = load_json(spec["manifest"])
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    data, time_s, x_m, y_m = read_test_segment(ez)
    x_mask = (x_m >= 0.09e-2 - 1.0e-15) & (x_m <= 1.19e-2 + 1.0e-15)
    dy_m = float(np.median(np.diff(y_m)))
    sample_count = len(data) - PRE - AFT + 1

    field_model = np.empty((sample_count, AFT, len(CHANNELS)), dtype=np.float64)
    field_copy = np.empty_like(field_model)
    field_corr = np.empty_like(field_model)
    field_copy_corr = np.empty_like(field_model)
    ey_model = np.empty((sample_count, AFT), dtype=np.float64)
    ey_copy = np.empty_like(ey_model)
    amp_model = np.empty((sample_count, AFT, len(BANDS)), dtype=np.float64)
    amp_copy = np.empty_like(amp_model)
    amp_true = np.empty_like(amp_model)
    cross_model = np.empty((sample_count, AFT, len(BANDS)), dtype=np.complex128)
    cross_copy = np.empty_like(cross_model)
    cross_true = np.empty_like(cross_model)
    transport_model = np.empty((sample_count, AFT, len(BANDS)), dtype=np.float64)
    transport_copy = np.empty_like(transport_model)
    transport_true = np.empty_like(transport_model)
    calibration_rows = []
    snapshot = None

    if variant == "input_window_calibrated":
        if source_moments is None:
            raise ValueError("source_moments are required for calibrated evaluation")
        source_mean = np.asarray(source_moments["mean"], dtype=np.float64)
        source_std = np.asarray(source_moments["std"], dtype=np.float64)
    else:
        source_mean = source_std = None

    started = time.time()
    for sample in range(sample_count):
        input_phys = np.asarray(data[sample : sample + PRE], dtype=np.float64)
        truth = np.asarray(data[sample + PRE : sample + PRE + AFT], dtype=np.float64)
        copy = np.repeat(input_phys[-1:], AFT, axis=0)

        if variant == "strict_source_normalization":
            model_input = pad_normalized(input_phys, low, high)
            target_mean = target_std = None
        elif variant == "input_window_calibrated":
            target_mean = np.mean(input_phys, axis=(0, 2, 3), dtype=np.float64)
            target_std = np.std(input_phys, axis=(0, 2, 3), dtype=np.float64)
            target_std = np.maximum(target_std, np.maximum(np.abs(target_mean) * 1.0e-8, 1.0e-12))
            mapped = (
                (input_phys - target_mean[None, :, None, None])
                / target_std[None, :, None, None]
                * source_std[None, :, None, None]
                + source_mean[None, :, None, None]
            )
            model_input = pad_normalized(mapped, low, high)
            calibration_rows.append(
                {
                    "sample": sample,
                    **{f"{name}_target_mean": float(target_mean[i]) for i, name in enumerate(CHANNELS)},
                    **{f"{name}_target_std": float(target_std[i]) for i, name in enumerate(CHANNELS)},
                }
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

        tensor = torch.from_numpy(model_input[None]).to(device)
        prediction_norm = model(tensor)[0, :, :, :VALID_H, :VALID_W].float().cpu().numpy()
        prediction_source = (
            prediction_norm.astype(np.float64) * (high - low)[None, :, None, None]
            + low[None, :, None, None]
        )
        if variant == "input_window_calibrated":
            prediction = (
                (prediction_source - source_mean[None, :, None, None])
                / source_std[None, :, None, None]
                * target_std[None, :, None, None]
                + target_mean[None, :, None, None]
            )
        else:
            prediction = prediction_source

        for channel_index in range(len(CHANNELS)):
            difference = prediction[:, channel_index] - truth[:, channel_index]
            copy_difference = copy[:, channel_index] - truth[:, channel_index]
            field_model[sample, :, channel_index] = np.mean(difference * difference, axis=(-2, -1))
            field_copy[sample, :, channel_index] = np.mean(copy_difference * copy_difference, axis=(-2, -1))
            field_corr[sample, :, channel_index] = spatial_corr(
                prediction[:, channel_index], truth[:, channel_index]
            )
            field_copy_corr[sample, :, channel_index] = spatial_corr(
                copy[:, channel_index], truth[:, channel_index]
            )

        pred_obs = observables(prediction, x_mask, dy_m)
        true_obs = observables(truth, x_mask, dy_m)
        copy_obs = observables(copy, x_mask, dy_m)
        ey_model[sample] = np.mean((pred_obs["ey"] - true_obs["ey"]) ** 2, axis=(-2, -1))
        ey_copy[sample] = np.mean((copy_obs["ey"] - true_obs["ey"]) ** 2, axis=(-2, -1))
        amp_model[sample] = pred_obs["amplitude"]
        amp_true[sample] = true_obs["amplitude"]
        amp_copy[sample] = copy_obs["amplitude"]
        cross_model[sample] = pred_obs["cross"]
        cross_true[sample] = true_obs["cross"]
        cross_copy[sample] = copy_obs["cross"]
        transport_model[sample] = pred_obs["transport"]
        transport_true[sample] = true_obs["transport"]
        transport_copy[sample] = copy_obs["transport"]

        if sample == sample_count // 2:
            snapshot = {
                "target_time_us": float(time_s[sample + PRE + AFT - 1] * 1.0e6),
                "input_phi": input_phys[-1, 2].copy(),
                "truth_phi": truth[-1, 2].copy(),
                "prediction_phi": prediction[-1, 2].copy(),
                "copy_phi": copy[-1, 2].copy(),
            }
        if progress_every and (sample + 1) % progress_every == 0:
            elapsed = time.time() - started
            print(
                f"[EVAL] model={spec_key} target=E{token(ez)} variant={variant} "
                f"sample={sample + 1}/{sample_count} elapsed={elapsed:.1f}s",
                flush=True,
            )

    result = {
        "model_key": spec_key,
        "model_label": spec["label"],
        "source_Ez_kVm": list(spec["source_E"]),
        "target_Ez_kVm": ez,
        "role": role_for(spec, ez),
        "variant": variant,
        "sample_count": sample_count,
        "time_s": time_s,
        "field_model": field_model,
        "field_copy": field_copy,
        "field_corr": field_corr,
        "field_copy_corr": field_copy_corr,
        "ey_model": ey_model,
        "ey_copy": ey_copy,
        "amp_model": amp_model,
        "amp_copy": amp_copy,
        "amp_true": amp_true,
        "cross_model": cross_model,
        "cross_copy": cross_copy,
        "cross_true": cross_true,
        "transport_model": transport_model,
        "transport_copy": transport_copy,
        "transport_true": transport_true,
        "snapshot": snapshot,
        "calibration_rows": calibration_rows,
        "normalized_below_zero_fraction": np.mean(
            data.astype(np.float64) < low[None, :, None, None], axis=(0, 2, 3)
        ),
        "normalized_above_one_fraction": np.mean(
            data.astype(np.float64) > high[None, :, None, None], axis=(0, 2, 3)
        ),
        "elapsed_sec": time.time() - started,
    }
    return result


def summarize_result(result: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    horizons = []
    common = {
        "model_key": result["model_key"],
        "source_Ez_kVm": "+".join(token(v) for v in result["source_Ez_kVm"]),
        "target_Ez_kVm": result["target_Ez_kVm"],
        "role": result["role"],
        "variant": result["variant"],
    }

    for channel_index, channel in enumerate(CHANNELS):
        model = result["field_model"][:, :, channel_index]
        copy = result["field_copy"][:, :, channel_index]
        rows.append(
            {
                **common,
                "metric": "field_mse",
                "component": channel,
                "model_error": float(np.mean(model)),
                "copy_error": float(np.mean(copy)),
                "model_over_copy": float(np.mean(model) / np.mean(copy)),
                "skill_vs_copy": float(1.0 - np.mean(model) / np.mean(copy)),
                "model_correlation": float(np.nanmean(result["field_corr"][:, :, channel_index])),
                "copy_correlation": float(np.nanmean(result["field_copy_corr"][:, :, channel_index])),
                "complex_coherence": "",
                "copy_complex_coherence": "",
            }
        )
        for horizon in range(AFT):
            horizons.append(
                {
                    **common,
                    "horizon_frame": horizon + 1,
                    "horizon_ns": (horizon + 1) * FRAME_NS,
                    "metric": "field_mse",
                    "component": channel,
                    "model_error": float(np.mean(model[:, horizon])),
                    "copy_error": float(np.mean(copy[:, horizon])),
                    "model_over_copy": float(np.mean(model[:, horizon]) / np.mean(copy[:, horizon])),
                }
            )

    rows.append(
        {
            **common,
            "metric": "field_mse",
            "component": "ey",
            "model_error": float(np.mean(result["ey_model"])),
            "copy_error": float(np.mean(result["ey_copy"])),
            "model_over_copy": float(np.mean(result["ey_model"]) / np.mean(result["ey_copy"])),
            "skill_vs_copy": float(1.0 - np.mean(result["ey_model"]) / np.mean(result["ey_copy"])),
            "model_correlation": "",
            "copy_correlation": "",
            "complex_coherence": "",
            "copy_complex_coherence": "",
        }
    )
    for horizon in range(AFT):
        horizons.append(
            {
                **common,
                "horizon_frame": horizon + 1,
                "horizon_ns": (horizon + 1) * FRAME_NS,
                "metric": "field_mse",
                "component": "ey",
                "model_error": float(np.mean(result["ey_model"][:, horizon])),
                "copy_error": float(np.mean(result["ey_copy"][:, horizon])),
                "model_over_copy": float(
                    np.mean(result["ey_model"][:, horizon])
                    / np.mean(result["ey_copy"][:, horizon])
                ),
            }
        )

    for band_index, band in enumerate(BANDS):
        comparisons = (
            ("phi_band_amplitude_mae", result["amp_model"], result["amp_copy"], result["amp_true"]),
            ("modal_transport_mae", result["transport_model"], result["transport_copy"], result["transport_true"]),
        )
        for metric, model_values, copy_values, true_values in comparisons:
            model_error = np.abs(model_values[:, :, band_index] - true_values[:, :, band_index])
            copy_error = np.abs(copy_values[:, :, band_index] - true_values[:, :, band_index])
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "component": band,
                    "model_error": float(np.mean(model_error)),
                    "copy_error": float(np.mean(copy_error)),
                    "model_over_copy": float(np.mean(model_error) / np.mean(copy_error)),
                    "skill_vs_copy": float(1.0 - np.mean(model_error) / np.mean(copy_error)),
                    "model_correlation": corrcoef(model_values[:, :, band_index], true_values[:, :, band_index]),
                    "copy_correlation": corrcoef(copy_values[:, :, band_index], true_values[:, :, band_index]),
                    "complex_coherence": "",
                    "copy_complex_coherence": "",
                }
            )
            for horizon in range(AFT):
                horizons.append(
                    {
                        **common,
                        "horizon_frame": horizon + 1,
                        "horizon_ns": (horizon + 1) * FRAME_NS,
                        "metric": metric,
                        "component": band,
                        "model_error": float(np.mean(model_error[:, horizon])),
                        "copy_error": float(np.mean(copy_error[:, horizon])),
                        "model_over_copy": float(
                            np.mean(model_error[:, horizon]) / np.mean(copy_error[:, horizon])
                        ),
                    }
                )

        true_cross = result["cross_true"][:, :, band_index]
        model_cross = result["cross_model"][:, :, band_index]
        copy_cross = result["cross_copy"][:, :, band_index]
        weights = np.abs(true_cross)
        model_phase = wrapped_angle_difference(model_cross, true_cross)
        copy_phase = wrapped_angle_difference(copy_cross, true_cross)
        model_weighted = float(np.sum(weights * model_phase) / np.sum(weights))
        copy_weighted = float(np.sum(weights * copy_phase) / np.sum(weights))
        rows.append(
            {
                **common,
                "metric": "cross_phase_weighted_mae_rad",
                "component": band,
                "model_error": model_weighted,
                "copy_error": copy_weighted,
                "model_over_copy": model_weighted / copy_weighted,
                "skill_vs_copy": 1.0 - model_weighted / copy_weighted,
                "model_correlation": "",
                "copy_correlation": "",
                "complex_coherence": complex_coherence(model_cross, true_cross),
                "copy_complex_coherence": complex_coherence(copy_cross, true_cross),
            }
        )
        for horizon in range(AFT):
            horizon_weights = weights[:, horizon]
            model_value = float(
                np.sum(horizon_weights * model_phase[:, horizon]) / np.sum(horizon_weights)
            )
            copy_value = float(
                np.sum(horizon_weights * copy_phase[:, horizon]) / np.sum(horizon_weights)
            )
            horizons.append(
                {
                    **common,
                    "horizon_frame": horizon + 1,
                    "horizon_ns": (horizon + 1) * FRAME_NS,
                    "metric": "cross_phase_weighted_mae_rad",
                    "component": band,
                    "model_error": model_value,
                    "copy_error": copy_value,
                    "model_over_copy": model_value / copy_value,
                }
            )
    return rows, horizons


def plot_heatmap(summary_rows: list[dict], metric: str, components: list[str], output: Path) -> None:
    strict = [
        row
        for row in summary_rows
        if row["variant"] == "strict_source_normalization" and row["metric"] == metric
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 5.6), sharey=True, layout="constrained"
    )
    for axis, model_key in zip(axes, MODEL_SPECS):
        matrix = np.full((len(components), len(E_VALUES)), np.nan)
        for row in strict:
            if row["model_key"] != model_key or row["component"] not in components:
                continue
            i = components.index(row["component"])
            j = E_VALUES.index(float(row["target_Ez_kVm"]))
            matrix[i, j] = float(row["model_over_copy"])
        displayed = np.log10(np.maximum(matrix, 1.0e-3))
        image = axis.imshow(displayed, cmap="RdYlBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        axis.set_xticks(range(len(E_VALUES)), [token(value) for value in E_VALUES])
        axis.set_yticks(range(len(components)), components)
        axis.set_xlabel("Target Ez [kV/m]")
        axis.set_title(MODEL_SPECS[model_key]["label"])
        for i in range(len(components)):
            for j in range(len(E_VALUES)):
                value = matrix[i, j]
                color = "white" if value > 5.0 or value < 0.2 else "black"
                axis.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)
    axes[0].set_ylabel("Observable")
    colorbar = fig.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02
    )
    colorbar.set_label("log10(model error / copy error)")
    fig.suptitle(f"Strict transfer: {metric} (values below 1 beat copy)")
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_final_horizons(horizon_rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    pairs = (("low_E10_E20", 40.0), ("high_E30_E40", 10.0))
    for row_index, (model_key, target) in enumerate(pairs):
        for column, component in enumerate(("phi", "ey")):
            axis = axes[row_index, column]
            for variant, label, color in (
                ("strict_source_normalization", "strict zero-shot", "#2563eb"),
                ("input_window_calibrated", "input-window calibrated", "#d97706"),
            ):
                selected = sorted(
                    (
                        row
                        for row in horizon_rows
                        if row["model_key"] == model_key
                        and float(row["target_Ez_kVm"]) == target
                        and row["variant"] == variant
                        and row["metric"] == "field_mse"
                        and row["component"] == component
                    ),
                    key=lambda row: float(row["horizon_ns"]),
                )
                if selected:
                    axis.plot(
                        [float(row["horizon_ns"]) for row in selected],
                        [float(row["model_over_copy"]) for row in selected],
                        marker="o",
                        color=color,
                        label=label,
                    )
            axis.axhline(1.0, color="black", linestyle=":", label="equal to copy")
            axis.set_yscale("log")
            axis.set_title(f"{MODEL_SPECS[model_key]['label']} -> E{token(target)}: {component}")
            axis.set_ylabel("Model MSE / copy MSE")
            axis.legend(loc="lower right")
    for axis in axes[-1]:
        axis.set_xlabel("Prediction horizon [ns]")
    fig.suptitle("Final opposite-regime transfer")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_snapshots(results: list[dict], output: Path) -> None:
    targets = [
        result
        for result in results
        if result["variant"] == "strict_source_normalization"
        and result["role"] == "opposite_regime_final"
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 9))
    for row_index, result in enumerate(targets):
        snapshot = result["snapshot"]
        fields = (
            ("Last input", snapshot["input_phi"]),
            ("PIC truth", snapshot["truth_phi"]),
            ("Zero-shot prediction", snapshot["prediction_phi"]),
            ("Absolute error", np.abs(snapshot["prediction_phi"] - snapshot["truth_phi"])),
        )
        vmin = min(np.min(snapshot["truth_phi"]), np.min(snapshot["prediction_phi"]), np.min(snapshot["input_phi"]))
        vmax = max(np.max(snapshot["truth_phi"]), np.max(snapshot["prediction_phi"]), np.max(snapshot["input_phi"]))
        for column, (title, values) in enumerate(fields):
            cmap = "magma" if column == 3 else "viridis"
            kwargs = {} if column == 3 else {"vmin": vmin, "vmax": vmax}
            image = axes[row_index, column].imshow(values, origin="lower", aspect="auto", cmap=cmap, **kwargs)
            axes[row_index, column].set_title(title)
            axes[row_index, column].set_xlabel("Azimuthal index")
            if column == 0:
                axes[row_index, column].set_ylabel(
                    f"{result['model_label']} -> E{token(result['target_Ez_kVm'])}\nRadial index"
                )
            fig.colorbar(image, ax=axes[row_index, column], fraction=0.046, pad=0.04)
    fig.suptitle("Strict opposite-regime phi prediction near 28.6 us at 150 ns horizon")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def metric_lookup(rows: list[dict], model: str, target: float, variant: str, metric: str, component: str) -> dict:
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


def make_readme(
    rows: list[dict], output: Path, runtime_sec: float, model_family: str
) -> None:
    objective = (
        "Data-only loss"
        if model_family == "data_only"
        else "Data + azimuthal Fourier amplitude/phase/cross-phase loss"
    )
    lines = [
        "# RadAz bidirectional regime-generalization evaluation",
        "",
        "## Design",
        "",
        "- Low-E model: trained on E10 and E20 kV/m.",
        "- High-E model: trained on E30 and E40 kV/m.",
        f"- Objective: {objective}.",
        "- SimVPv2 gSTA, direct10, stride1, 15 ns/frame.",
        "- Each case was split by frames into 8:1:1 before source windows were mixed.",
        "- Test evaluation uses true 10-frame target histories and predicts the next 10 frames. It is teacher-forced direct prediction, not rollout.",
        "- Strict transfer uses only source-pool train normalization and no clipping.",
        "- Input-window calibration uses only each target input history's mean/std; no future target frame is used.",
        "",
        "## Final opposite-regime results",
        "",
        "| Direction | Variant | phi/copy | Ey/copy | MTSI transport/copy | ECDI transport/copy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, target, direction in (
        ("low_E10_E20", 40.0, "E10+E20 -> E40"),
        ("high_E30_E40", 10.0, "E30+E40 -> E10"),
    ):
        for variant in ("strict_source_normalization", "input_window_calibrated"):
            available = any(
                row["model_key"] == model
                and float(row["target_Ez_kVm"]) == target
                and row["variant"] == variant
                for row in rows
            )
            if not available:
                continue
            phi = metric_lookup(rows, model, target, variant, "field_mse", "phi")
            ey = metric_lookup(rows, model, target, variant, "field_mse", "ey")
            mtsi = metric_lookup(rows, model, target, variant, "modal_transport_mae", "MTSI")
            ecdi = metric_lookup(rows, model, target, variant, "modal_transport_mae", "ECDI")
            lines.append(
                f"| {direction} | {variant} | {float(phi['model_over_copy']):.3f} | "
                f"{float(ey['model_over_copy']):.3f} | {float(mtsi['model_over_copy']):.3f} | "
                f"{float(ecdi['model_over_copy']):.3f} |"
            )
    low_e10_phi = metric_lookup(
        rows, "low_E10_E20", 10.0, "strict_source_normalization", "field_mse", "phi"
    )
    low_e20_phi = metric_lookup(
        rows, "low_E10_E20", 20.0, "strict_source_normalization", "field_mse", "phi"
    )
    high_e30_phi = metric_lookup(
        rows, "high_E30_E40", 30.0, "strict_source_normalization", "field_mse", "phi"
    )
    high_e40_phi = metric_lookup(
        rows, "high_E30_E40", 40.0, "strict_source_normalization", "field_mse", "phi"
    )
    low_e40_phi = metric_lookup(
        rows, "low_E10_E20", 40.0, "strict_source_normalization", "field_mse", "phi"
    )
    high_e10_phi = metric_lookup(
        rows, "high_E30_E40", 10.0, "strict_source_normalization", "field_mse", "phi"
    )
    low_e20_mtsi = metric_lookup(
        rows,
        "low_E10_E20",
        20.0,
        "strict_source_normalization",
        "modal_transport_mae",
        "MTSI",
    )
    high_e40_ecdi = metric_lookup(
        rows,
        "high_E30_E40",
        40.0,
        "strict_source_normalization",
        "modal_transport_mae",
        "ECDI",
    )
    lines.extend(
        [
            "",
            "A ratio below 1 means SimVPv2 beats persistence/copy.",
            "",
            "## Interpretation",
            "",
            "- Field prediction was learned in-domain. The phi/copy ratios were "
            f"E20 `{float(low_e20_phi['model_over_copy']):.3f}`, E30 "
            f"`{float(high_e30_phi['model_over_copy']):.3f}`, and E40 "
            f"`{float(high_e40_phi['model_over_copy']):.3f}`. E10 phi alone was "
            f"`{float(low_e10_phi['model_over_copy']):.3f}`, because persistence is "
            "especially strong there, although its density and Ey errors improved.",
            "- Strict opposite-regime phi transfer failed in both directions: "
            f"low-to-high `{float(low_e40_phi['model_over_copy']):.1f}` and "
            f"high-to-low `{float(high_e10_phi['model_over_copy']):.1f}` times copy.",
            "- High spatial correlation is not sufficient evidence of mode recovery. "
            "The radial mean profile dominates correlation while azimuthal mode "
            "amplitude, cross-phase, and transport can be wrong.",
            "- Input-window calibration greatly reduced phi scale error, but did not "
            "make phi or modal transport beat copy. Distribution shift is therefore "
            "only part of the failure; the learned dynamics are regime-specific.",
            "- Two in-domain transport references are low-model E20 MTSI "
            f"(`{float(low_e20_mtsi['model_over_copy']):.3f}`) and high-model E40 ECDI "
            f"(`{float(high_e40_ecdi['model_over_copy']):.3f}`). These must be compared "
            "with the data-only run before attributing a change to the spectral objective.",
            "- Regime-level generalization requires mode amplitude, cross-phase, and "
            "modal transport to improve together; field MSE alone is not sufficient.",
            "",
            "## Files",
            "",
            "- `overall_metrics.csv`: all aggregate field and physics metrics.",
            "- `metrics_by_horizon.csv`: horizon-resolved errors.",
            "- `normalization_diagnostics.csv`: fractions outside source normalization bounds.",
            "- `strict_field_model_over_copy_by_Ez.png`: field-error transfer map.",
            "- `strict_transport_model_over_copy_by_Ez.png`: modal-transport transfer map.",
            "- `strict_cross_phase_model_over_copy_by_Ez.png`: cross-phase transfer map.",
            "- `final_zero_shot_by_horizon.png`: strict and input-window calibrated final transfer.",
            "- `final_zero_shot_phi_snapshots.png`: representative strict phi predictions.",
            "- `evaluation_summary.json`: machine-readable protocol and key metrics.",
            "",
            f"Runtime: {runtime_sec / 60.0:.1f} min.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global MODEL_SPECS
    args = parse_args()
    MODEL_SPECS = (
        DATA_ONLY_MODEL_SPECS
        if args.model_family == "data_only"
        else SPECTRAL_MODEL_SPECS
    )
    default_output = (
        OUTPUT
        if args.model_family == "data_only"
        else WORKDIR
        / "compare_radaz_regime_generalization_direct10_spectral_full_50ep"
    )
    output = (args.output or default_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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

    started = time.time()
    results = []
    normalization_rows = []
    for model_key, spec in MODEL_SPECS.items():
        print(f"[MODEL] loading {model_key}", flush=True)
        model = load_model(spec, device)
        for ez in E_VALUES:
            result = evaluate_case(
                model,
                model_key,
                spec,
                ez,
                "strict_source_normalization",
                device,
                args.progress_every,
            )
            results.append(result)
            for channel_index, channel in enumerate(CHANNELS):
                normalization_rows.append(
                    {
                        "model_key": model_key,
                        "target_Ez_kVm": ez,
                        "variant": result["variant"],
                        "channel": channel,
                        "below_source_range_fraction": float(result["normalized_below_zero_fraction"][channel_index]),
                        "above_source_range_fraction": float(result["normalized_above_one_fraction"][channel_index]),
                    }
                )
        if not args.skip_calibrated:
            moments = source_reference_moments(spec)
            result = evaluate_case(
                model,
                model_key,
                spec,
                float(spec["final_target"]),
                "input_window_calibrated",
                device,
                args.progress_every,
                source_moments=moments,
            )
            results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    overall_rows = []
    horizon_rows = []
    for result in results:
        current_overall, current_horizons = summarize_result(result)
        overall_rows.extend(current_overall)
        horizon_rows.extend(current_horizons)
    write_csv(output / "overall_metrics.csv", overall_rows)
    write_csv(output / "metrics_by_horizon.csv", horizon_rows)
    write_csv(output / "normalization_diagnostics.csv", normalization_rows)

    plot_heatmap(
        overall_rows,
        "field_mse",
        ["electron_den", "ion_den", "phi", "ey"],
        output / "strict_field_model_over_copy_by_Ez.png",
    )
    plot_heatmap(
        overall_rows,
        "modal_transport_mae",
        ["MTSI", "ECDI"],
        output / "strict_transport_model_over_copy_by_Ez.png",
    )
    plot_heatmap(
        overall_rows,
        "cross_phase_weighted_mae_rad",
        ["MTSI", "ECDI"],
        output / "strict_cross_phase_model_over_copy_by_Ez.png",
    )
    plot_final_horizons(horizon_rows, output / "final_zero_shot_by_horizon.png")
    plot_snapshots(results, output / "final_zero_shot_phi_snapshots.png")

    key_metrics = {}
    for model, target, direction in (
        ("low_E10_E20", 40.0, "low_to_high"),
        ("high_E30_E40", 10.0, "high_to_low"),
    ):
        key_metrics[direction] = {}
        for variant in ("strict_source_normalization", "input_window_calibrated"):
            if not any(
                row["model_key"] == model
                and float(row["target_Ez_kVm"]) == target
                and row["variant"] == variant
                for row in overall_rows
            ):
                continue
            key_metrics[direction][variant] = {
                output_key: float(
                    metric_lookup(
                        overall_rows,
                        model,
                        target,
                        variant,
                        metric,
                        component,
                    )["model_over_copy"]
                )
                for output_key, metric, component in (
                    ("electron_den", "field_mse", "electron_den"),
                    ("ion_den", "field_mse", "ion_den"),
                    ("phi", "field_mse", "phi"),
                    ("ey", "field_mse", "ey"),
                    ("MTSI_transport", "modal_transport_mae", "MTSI"),
                    ("ECDI_transport", "modal_transport_mae", "ECDI"),
                    ("MTSI_cross_phase", "cross_phase_weighted_mae_rad", "MTSI"),
                    ("ECDI_cross_phase", "cross_phase_weighted_mae_rad", "ECDI"),
                )
            }
    runtime = time.time() - started
    summary = {
        "status": "PASS",
        "device": str(device),
        "model_family": args.model_family,
        "runtime_sec": runtime,
        "protocol": {
            "low_source_Ez_kVm": [10.0, 20.0],
            "high_source_Ez_kVm": [30.0, 40.0],
            "evaluated_Ez_kVm": list(E_VALUES),
            "input_frames": PRE,
            "output_frames": AFT,
            "frame_interval_ns": FRAME_NS,
            "test_frame_range": [1800, 2000],
            "test_time_us": [27.0, 30.0],
            "teacher_forced_direct_not_rollout": True,
            "strict_normalization": "source-pool train-only, no clipping",
            "calibrated_normalization": "per-window target-input mean/std mapped to source train mean/std; no future frames",
        },
        "key_final_transfer_model_over_copy": key_metrics,
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    make_readme(overall_rows, output, runtime, args.model_family)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[PASS] output={output}", flush=True)


if __name__ == "__main__":
    main()
