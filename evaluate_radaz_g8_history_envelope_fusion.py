#!/usr/bin/env python3
"""Fuse the E-history hidden-band envelope ROM with the frozen E25 G8 SimVP.

This is an integration test on the held-out final tenth of the E25 stationary
trajectory.  SimVP supplies a complex high-mode carrier; the independently
trained electric-history ROM replaces only its n=17--21 amplitudes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from openstl.models.simvp_model import SimVP_Model
from radaz_electric_history_hidden_band_rom import (
    HiddenBandEnvelopeBundle,
    apply_hidden_amplitude_to_carrier,
    blend_carrier_and_rom_amplitude,
    electric_history_controls,
    electric_history_blend_weight,
)
from train_radaz_g2_residual_superresolution import (
    DEFAULT_H5,
    make_grid_interpolated,
    segment_starts,
)
import train_radaz_electric_history_hidden_band_rom as complex_rom
import train_radaz_electric_history_hidden_band_envelope_rom as envelope_rom


ROOT = Path(__file__).resolve().parent
DEFAULT_G8 = ROOT / "workdirs/radaz_e25_g8_simvp_residual_sr_sync10_20to30us"
DEFAULT_ENVELOPE = ROOT / "workdirs/radaz_electric_history_hidden_band_envelope_rom"
DEFAULT_OUTPUT = ROOT / "workdirs/radaz_e25_g8_history_envelope_fusion"
CHANNELS = ("electron_den", "ion_den", "phi")
FOURIER_FIELDS = ("phi", "electron_den", "ion_den")
TINY = np.finfo(np.float64).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--g8-workdir", type=Path, default=DEFAULT_G8)
    parser.add_argument("--envelope-workdir", type=Path, default=DEFAULT_ENVELOPE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-us", type=float, default=20.0)
    parser.add_argument("--end-us", type=float, default=30.0)
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Evaluate every complete G8 window (used for a new transition case).",
    )
    parser.add_argument("--current-ez-kvm", type=float, default=25.0)
    parser.add_argument("--source-ez-kvm", type=float, default=25.0)
    parser.add_argument("--transition", action="store_true")
    return parser.parse_args()


def predict_unique_frames_full_radial(
    fine: np.ndarray,
    baseline: np.ndarray,
    starts: np.ndarray,
    checkpoint: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = SimVP_Model(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    length = int(checkpoint["model_kwargs"]["in_shape"][0])
    scale = torch.as_tensor(
        checkpoint["residual_rms"], dtype=torch.float32, device=device
    ).view(1, 1, 3, 1, 1)
    first = int(starts[0])
    final = int(starts[-1]) + length
    sums = np.zeros((final-first, 3, 257, 256), dtype=np.float64)
    counts = np.zeros(final-first, dtype=np.int64)
    with torch.inference_mode():
        for start in starts:
            inputs = torch.from_numpy(
                baseline[int(start):int(start)+length]
            ).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                prediction = inputs + model(inputs) * scale
            values = prediction[0, :, :, :257].float().cpu().numpy()
            local = int(start) - first
            sums[local:local+length] += values
            counts[local:local+length] += 1
    covered = counts > 0
    indices = np.arange(first, final, dtype=np.int64)[covered]
    prediction = (sums[covered] / counts[covered, None, None, None]).astype(np.float32)
    return prediction, fine[indices, :, :257].copy(), baseline[indices, :, :257].copy(), indices


def physical_fourier(
    normalized: np.ndarray, norm_low: np.ndarray, norm_high: np.ndarray
) -> np.ndarray:
    physical = (
        normalized.astype(np.float64)
        * (norm_high - norm_low)[None, :, None, None]
        + norm_low[None, :, None, None]
    )
    reordered = physical[:, (2, 0, 1)]
    groups = np.array_split(np.arange(257, dtype=np.int64), 8)
    coefficients = np.empty((len(physical), 3, 8, 22), dtype=np.complex128)
    for field in range(3):
        for radial, group in enumerate(groups):
            radial_mean = np.mean(reordered[:, field, group, :], axis=1)
            coefficients[:, field, radial] = np.fft.rfft(
                radial_mean, axis=-1, norm="forward"
            )[:, :22]
    return coefficients


def metrics_for(
    truth: np.ndarray,
    candidate: np.ndarray,
    radial_weights: np.ndarray,
    time_us: np.ndarray,
) -> dict:
    return envelope_rom.envelope_metrics(
        truth,
        np.abs(candidate),
        candidate,
        radial_weights,
        time_us,
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.g8_workdir / "checkpoint_best.pth",
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint.get("metadata", {}).get("grid_factor", -1)) != 8:
        raise ValueError("checkpoint is not the frozen G8 model")
    with h5py.File(args.h5, "r") as handle:
        all_time_s = np.asarray(handle["time_s"], dtype=np.float64)
        selected = np.flatnonzero(
            (all_time_s * 1.0e6 >= args.start_us - 1.0e-9)
            & (all_time_s * 1.0e6 <= args.end_us + 1.0e-9)
        )
        fine = np.asarray(
            handle["data_tchw"][int(selected[0]):int(selected[-1])+1],
            dtype=np.float32,
        )
        time_s = all_time_s[selected]
        norm_low = np.asarray(handle["norm_low"], dtype=np.float64)
        norm_high = np.asarray(handle["norm_high"], dtype=np.float64)
    baseline_all = make_grid_interpolated(fine, 8)
    evaluation_begin = 0 if args.all_frames else int(math.floor(0.9 * len(fine)))
    length = int(checkpoint["model_kwargs"]["in_shape"][0])
    hop = int(checkpoint["metadata"]["window_hop"])
    starts = segment_starts(evaluation_begin, len(fine), length, hop)
    prediction, truth, baseline, indices = predict_unique_frames_full_radial(
        fine, baseline_all, starts, checkpoint, device
    )
    evaluation_time_us = time_s[indices] * 1.0e6
    truth_coefficients = physical_fourier(truth, norm_low, norm_high)
    prediction_coefficients = physical_fourier(prediction, norm_low, norm_high)
    baseline_coefficients = physical_fourier(baseline, norm_low, norm_high)
    bundle = HiddenBandEnvelopeBundle.load(args.envelope_workdir, device=device)
    controls = electric_history_controls(
        evaluation_time_us,
        current_ez_kvm=args.current_ez_kvm,
        source_ez_kvm=args.source_ez_kvm,
        transition=args.transition,
    )
    input_sources = {
        "coarse_visible": baseline_coefficients[..., :17],
        "simvp_visible": prediction_coefficients[..., :17],
    }
    truth_hidden = truth_coefficients[..., 17:22]
    g8_hidden = prediction_coefficients[..., 17:22]
    radial_weights = np.asarray([len(group) for group in np.array_split(np.arange(257), 8)], dtype=np.float64)
    radial_weights /= np.sum(radial_weights)
    rows = []
    series = {}
    common_indices = None
    candidates = {}
    for label, visible in input_sources.items():
        local_indices, predicted_amplitude = bundle.predict_amplitude(visible, controls)
        fused = apply_hidden_amplitude_to_carrier(
            g8_hidden[local_indices], predicted_amplitude[:, :3]
        )
        candidates[f"history_envelope_{label}"] = (local_indices, fused)
        gated = blend_carrier_and_rom_amplitude(
            g8_hidden[local_indices],
            predicted_amplitude[:, :3],
            electric_history_blend_weight(controls[local_indices]),
        )
        candidates[f"gated_history_envelope_{label}"] = (local_indices, gated)
        common_indices = local_indices
    candidates["g8_simvp"] = (common_indices, g8_hidden[common_indices])
    candidates["g8_phase_amplitude_oracle"] = (
        common_indices,
        apply_hidden_amplitude_to_carrier(
            g8_hidden[common_indices], np.abs(truth_hidden[common_indices])
        ),
    )
    candidates["ideal_g8_zero_hidden"] = (
        common_indices,
        np.zeros_like(truth_hidden[common_indices]),
    )
    for label, (local_indices, candidate) in candidates.items():
        current_truth = truth_hidden[local_indices]
        current_time = evaluation_time_us[local_indices]
        aggregate = metrics_for(current_truth, candidate, radial_weights, current_time)
        rows.append({"source": label, "field": "all", "mode": "17-21", "frames": len(local_indices), **aggregate})
        for field_index, field in enumerate(FOURIER_FIELDS):
            for local_mode, mode in enumerate(range(17, 22)):
                item = metrics_for(
                    current_truth[:, field_index:field_index+1, :, local_mode:local_mode+1],
                    candidate[:, field_index:field_index+1, :, local_mode:local_mode+1],
                    radial_weights,
                    current_time,
                )
                rows.append({"source": label, "field": field, "mode": mode, "frames": len(local_indices), **item})
        weights = radial_weights[None, None, :, None]
        series[label] = {
            "time": current_time,
            "truth": np.sum(np.abs(current_truth)**2 * weights, axis=(1,2,3)),
            "candidate": np.sum(np.abs(candidate)**2 * weights, axis=(1,2,3)),
        }
    complex_rom.write_csv(output / "fusion_metrics.csv", rows)
    aggregate = {row["source"]: {k:v for k,v in row.items() if k not in ("source","field","mode")} for row in rows if row["field"] == "all"}
    baseline_metrics = aggregate["g8_simvp"]
    fused_metrics = aggregate["history_envelope_simvp_visible"]
    gated_metrics = aggregate["gated_history_envelope_simvp_visible"]
    summary = {
        "status": "g8_history_envelope_integration_test_complete",
        "evaluation_time_us": [float(evaluation_time_us[common_indices][0]), float(evaluation_time_us[common_indices][-1])],
        "g8_training_condition": "E25 stationary",
        "evaluated_electric_history": {
            "source_ez_kvm": args.source_ez_kvm,
            "current_ez_kvm": args.current_ez_kvm,
            "transition": args.transition,
            "all_frames": args.all_frames,
        },
        "envelope_rom_e25_used_for_training": False,
        "aggregate": aggregate,
        "fusion_increment": {
            "power_series_error_reduction": baseline_metrics["power_series_relative_l2"] - fused_metrics["power_series_relative_l2"],
            "complex_error_reduction": baseline_metrics["complex_relative_l2"] - fused_metrics["complex_relative_l2"],
            "complex_coherence_gain": fused_metrics["complex_coherence"] - baseline_metrics["complex_coherence"],
        },
        "conservative_gated_fusion_increment": {
            "power_series_error_reduction": baseline_metrics["power_series_relative_l2"] - gated_metrics["power_series_relative_l2"],
            "complex_error_reduction": baseline_metrics["complex_relative_l2"] - gated_metrics["complex_relative_l2"],
            "complex_coherence_gain": gated_metrics["complex_coherence"] - baseline_metrics["complex_coherence"],
            "gate_weight_range": [
                float(np.min(electric_history_blend_weight(controls[common_indices]))),
                float(np.max(electric_history_blend_weight(controls[common_indices]))),
            ],
        },
        "claim_boundary": (
            "The E25->E22.5 result is confirmatory only when --transition, "
            "--source-ez-kvm 25, --current-ez-kvm 22.5, and its untouched "
            "normalized H5 are supplied."
        ),
    }
    (output / "summary.json").write_text(json.dumps(complex_rom.json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 5))
    truth_drawn = False
    for label in ("g8_simvp", "history_envelope_coarse_visible", "history_envelope_simvp_visible", "gated_history_envelope_simvp_visible"):
        item = series[label]
        if not truth_drawn:
            ax.plot(item["time"], item["truth"], color="black", lw=1.5, label="PIC truth"); truth_drawn=True
        ax.plot(item["time"], item["candidate"], lw=1.1, label=label)
    ax.set(xlabel="time [us]", ylabel="physical n=17-21 power", title="G8 SimVP + electric-history envelope ROM")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(output / "g8_history_envelope_power.png", dpi=180); plt.close(fig)
    print(json.dumps(complex_rom.json_safe(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
