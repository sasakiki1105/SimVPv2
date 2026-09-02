#!/usr/bin/env python3
"""Measure the upper bound of mode-selective G8/history amplitude fusion."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

import evaluate_radaz_g8_history_envelope_fusion as fusion
from radaz_electric_history_hidden_band_rom import (
    HiddenBandEnvelopeBundle,
    apply_hidden_amplitude_to_carrier,
    electric_history_controls,
)
from train_radaz_g2_residual_superresolution import make_grid_interpolated, segment_starts


ROOT = Path(__file__).resolve().parent
FIELDS = ("phi", "electron_den", "ion_den")
MODES = tuple(range(17, 22))


@dataclass(frozen=True)
class Case:
    name: str
    h5: Path
    start_us: float
    end_us: float
    source_ez: float
    current_ez: float
    transition: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--g8-workdir", type=Path, default=ROOT / "workdirs/radaz_e25_g8_simvp_residual_sr_sync10_20to30us")
    p.add_argument("--envelope-workdir", type=Path, default=ROOT / "workdirs/radaz_electric_history_hidden_band_envelope_rom")
    return p.parse_args()


def cases() -> list[Case]:
    return [
        Case("E25_stationary", fusion.DEFAULT_H5, 20.0, 30.0, 25.0, 25.0, False),
        Case("E20_to_E22p5", ROOT / "workdirs/radaz_e20_to_e22p5_transition/radaz_3ch_e25targetnorm_native257x256_pad260x256.h5", 30.0, 40.0, 20.0, 22.5, True),
        Case("E22p5_to_E20", ROOT / "workdirs/radaz_e22p5_to_e20_transition/radaz_3ch_e25targetnorm_native257x256_pad260x256.h5", 30.0, 35.0, 22.5, 20.0, True),
    ]


def optimal_static_weights(truth_amp: np.ndarray, g8_amp: np.ndarray, rom_amp: np.ndarray) -> np.ndarray:
    delta = rom_amp - g8_amp
    numerator = np.sum(delta * (truth_amp - g8_amp), axis=0)
    denominator = np.sum(delta * delta, axis=0)
    weight = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=weight, where=denominator > np.finfo(np.float64).tiny)
    return np.clip(weight, 0.0, 1.0)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate_case(case: Case, checkpoint: dict, bundle: HiddenBandEnvelopeBundle, device: torch.device):
    with h5py.File(case.h5, "r") as handle:
        time_s = np.asarray(handle["time_s"], dtype=np.float64)
        selected = np.flatnonzero((time_s * 1e6 >= case.start_us - 1e-9) & (time_s * 1e6 <= case.end_us + 1e-9))
        fine = np.asarray(handle["data_tchw"][selected[0]:selected[-1]+1], dtype=np.float32)
        local_time_us = time_s[selected] * 1e6
        norm_low = np.asarray(handle["norm_low"], dtype=np.float64)
        norm_high = np.asarray(handle["norm_high"], dtype=np.float64)
    baseline = make_grid_interpolated(fine, 8)
    length = int(checkpoint["model_kwargs"]["in_shape"][0])
    starts = segment_starts(0, len(fine), length, int(checkpoint["metadata"]["window_hop"]))
    prediction, truth, _, indices = fusion.predict_unique_frames_full_radial(fine, baseline, starts, checkpoint, device)
    evaluation_time = local_time_us[indices]
    truth_coeff = fusion.physical_fourier(truth, norm_low, norm_high)[..., 17:22]
    g8_coeff = fusion.physical_fourier(prediction, norm_low, norm_high)[..., 17:22]
    visible = fusion.physical_fourier(prediction, norm_low, norm_high)[..., :17]
    controls = electric_history_controls(evaluation_time, case.current_ez, case.source_ez, case.transition)
    rom_indices, rom_amp4 = bundle.predict_amplitude(visible, controls)
    truth_coeff = truth_coeff[rom_indices]
    g8_coeff = g8_coeff[rom_indices]
    evaluation_time = evaluation_time[rom_indices]
    rom_amp = rom_amp4[:, :3]
    truth_amp = np.abs(truth_coeff)
    g8_amp = np.abs(g8_coeff)
    weights = optimal_static_weights(truth_amp, g8_amp, rom_amp)
    oracle_amp = g8_amp + weights[None] * (rom_amp - g8_amp)
    oracle = apply_hidden_amplitude_to_carrier(g8_coeff, oracle_amp)
    rom = apply_hidden_amplitude_to_carrier(g8_coeff, rom_amp)
    radial_weights = np.asarray([len(g) for g in np.array_split(np.arange(257), 8)], dtype=np.float64)
    radial_weights /= radial_weights.sum()
    metrics = {
        "g8_simvp": fusion.metrics_for(truth_coeff, g8_coeff, radial_weights, evaluation_time),
        "history_rom": fusion.metrics_for(truth_coeff, rom, radial_weights, evaluation_time),
        "static_component_oracle": fusion.metrics_for(truth_coeff, oracle, radial_weights, evaluation_time),
    }
    weight_rows = []
    for fi, field in enumerate(FIELDS):
        for radial in range(8):
            for mi, mode in enumerate(MODES):
                weight_rows.append({"case": case.name, "field": field, "radial": radial, "mode": mode, "oracle_weight": float(weights[fi, radial, mi])})
    payload = {
        "truth_coeff": truth_coeff,
        "g8_coeff": g8_coeff,
        "rom_amplitude": rom_amp,
        "visible_coeff": visible[rom_indices],
        "controls": controls[rom_indices],
        "time_us": evaluation_time,
        "radial_weights": radial_weights,
    }
    return metrics, weight_rows, (float(evaluation_time[0]), float(evaluation_time[-1]), len(evaluation_time)), payload


def main() -> None:
    args = parse_args(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.g8_workdir / "checkpoint_best.pth", map_location="cpu", weights_only=False)
    bundle = HiddenBandEnvelopeBundle.load(args.envelope_workdir, device=device)
    summary = {"status": "complete", "primary_E25_to_E22p5_used": False, "oracle_uses_truth": True, "cases": {}}
    all_weights = []
    for case in cases():
        metrics, weights, interval, _ = evaluate_case(case, checkpoint, bundle, device)
        all_weights.extend(weights)
        g8 = metrics["g8_simvp"]["power_series_relative_l2"]
        oracle = metrics["static_component_oracle"]["power_series_relative_l2"]
        summary["cases"][case.name] = {
            "evaluation_time_us": [interval[0], interval[1]], "frames": interval[2],
            "metrics": metrics, "oracle_power_error_reduction": g8 - oracle,
            "oracle_relative_power_error_reduction": (g8 - oracle) / max(g8, np.finfo(float).tiny),
        }
        print(case.name, json.dumps(summary["cases"][case.name], indent=2), flush=True)
    reductions = [v["oracle_relative_power_error_reduction"] for v in summary["cases"].values()]
    summary["go_no_go"] = {
        "mean_relative_power_error_reduction": float(np.mean(reductions)),
        "all_cases_improved": bool(all(v > 0 for v in reductions)),
        "proceed_to_truth_free_gate": bool(np.mean(reductions) >= 0.10 and all(v > 0 for v in reductions)),
        "threshold": "mean >= 10% and every development case improves",
    }
    write_csv(output / "oracle_weights.csv", all_weights)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["go_no_go"], indent=2), flush=True)


if __name__ == "__main__":
    main()
