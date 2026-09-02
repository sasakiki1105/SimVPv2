#!/usr/bin/env python3
"""Build the matched axis-interpolation manifest for conditioned SimVPv2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = (
    ROOT.parent
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
)
OUTPUT = ROOT / "workdirs" / "2D_RadAz" / "radaz_conditioned_factorial_manifests"
CHANNELS = ("electron_den", "ion_den", "phi")
CONDITIONS = ("log_vE", "log_n0")
LY_M = 1.28e-2
E_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31

# The L-shaped parameter domain has one shared corner, (E10, B20), which is
# deliberately registered once.
TRAIN_CONDITIONS = (
    (10.0, 20.0),
    (20.0, 20.0),
    (30.0, 20.0),
    (40.0, 20.0),
    (10.0, 10.0),
    (10.0, 30.0),
)
HOLDOUT_CONDITIONS = (
    (22.5, 20.0),
    (25.0, 20.0),
    (10.0, 15.0),
    (10.0, 25.0),
)


def token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def case_name(ez_kvm: float, b_mT: float) -> str:
    return (
        f"2D_RadAz_Xe1p_Bx{token(b_mT)}mT_Ez{token(ez_kvm)}kVm_"
        "dt15ps_out15ns"
    )


def source_h5(ez_kvm: float, b_mT: float) -> Path:
    name = case_name(ez_kvm, b_mT)
    return RESULTS / name / name / "analysis_fields_uncompressed.h5"


def raw_condition(ez_kvm: float, b_mT: float) -> np.ndarray:
    electric_v_m = float(ez_kvm) * 1.0e3
    magnetic_t = float(b_mT) * 1.0e-3
    drift_velocity = electric_v_m / magnetic_t
    mode_n0 = (
        E_CHARGE * magnetic_t**2 * LY_M
        / (2.0 * np.pi * ELECTRON_MASS * electric_v_m)
    )
    return np.log(np.asarray([drift_velocity, mode_n0], dtype=np.float64))


def scan_training_fields(train_ratio: float, chunk_frames: int):
    train_min = np.full(len(CHANNELS), np.inf, dtype=np.float64)
    train_max = np.full(len(CHANNELS), -np.inf, dtype=np.float64)
    frame_counts = {}
    for ez_kvm, b_mT in TRAIN_CONDITIONS:
        path = source_h5(ez_kvm, b_mT)
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            nt = int(len(handle["axes/time_s"]))
            train_end = int(np.floor(nt * train_ratio))
            key = f"E{token(ez_kvm)}_B{token(b_mT)}"
            frame_counts[key] = {
                "frames": nt,
                "train_end_exclusive": train_end,
            }
            print(f"[SCAN] {key} frames=0..{train_end - 1}", flush=True)
            for start in range(0, train_end, chunk_frames):
                stop = min(start + chunk_frames, train_end)
                for channel_index, channel in enumerate(CHANNELS):
                    values = np.asarray(
                        handle[f"fields/{channel}"][start:stop, :257, :256]
                    )
                    if not np.all(np.isfinite(values)):
                        raise ValueError(
                            f"Non-finite {channel} in {path} frames {start}:{stop}"
                        )
                    train_min[channel_index] = min(
                        train_min[channel_index], float(np.min(values))
                    )
                    train_max[channel_index] = max(
                        train_max[channel_index], float(np.max(values))
                    )
    return train_min, train_max, frame_counts


def case_entry(ez_kvm, b_mT, norm_low, norm_high, splits, role):
    path = source_h5(ez_kvm, b_mT)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "case_key": f"E{token(ez_kvm)}_B{token(b_mT)}",
        "label": f"Bx={b_mT:g} mT, Ez={ez_kvm:g} kV/m",
        "role": role,
        "B_mT": float(b_mT),
        "Ez_kVm": float(ez_kvm),
        "Ly_m": LY_M,
        "dt_frame_ns": 15.0,
        "path": str(path),
        "format": "radaz_consolidated",
        "channels": list(CHANNELS),
        "spatial_stride": 1,
        "model_height": 260,
        "model_width": 256,
        "normalization_low": norm_low.tolist(),
        "normalization_high": norm_high.tolist(),
        "normalization_clip": False,
        "splits": list(splits),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--chunk-frames", type=int, default=16)
    args = parser.parse_args()
    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to one")

    train_min, train_max, frame_counts = scan_training_fields(
        args.train_ratio, args.chunk_frames
    )
    span = train_max - train_min
    if np.any(span <= 0.0):
        raise ValueError(f"Non-positive channel span: {span}")
    norm_low = train_min - args.margin * span
    norm_high = train_max + args.margin * span

    raw_conditions = np.stack(
        [raw_condition(ez, b) for ez, b in TRAIN_CONDITIONS], axis=0
    )
    condition_mean = np.mean(raw_conditions, axis=0)
    condition_std = np.std(raw_conditions, axis=0)
    if np.any(condition_std <= 0.0):
        raise ValueError(f"Degenerate condition std: {condition_std}")

    cases = [
        case_entry(ez, b, norm_low, norm_high, ("train", "val"), "source")
        for ez, b in TRAIN_CONDITIONS
    ]
    cases.extend(
        case_entry(ez, b, norm_low, norm_high, ("test",), "axis_holdout")
        for ez, b in HOLDOUT_CONDITIONS
    )
    manifest = {
        "name": "radaz_axis_interpolation_conditioned_factorial_direct10",
        "description": (
            "Matched 2x2 factorial training manifest. Six unique L-axis cases "
            "provide train/validation windows; E22.5, E25, B15 and B25 are "
            "condition-level test holdouts. The off-axis condition is deferred."
        ),
        "pre_seq_length": 10,
        "aft_seq_length": 10,
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "policy": "per-case frame-disjoint; condition holdouts excluded from train/val",
        },
        "normalization": {
            "policy": "all-source train-only per-channel affine",
            "channels": list(CHANNELS),
            "train_min": train_min.tolist(),
            "train_max": train_max.tolist(),
            "margin": args.margin,
            "low": norm_low.tolist(),
            "high": norm_high.tolist(),
            "clip": False,
            "frame_counts": frame_counts,
        },
        "condition_channels": list(CONDITIONS),
        "condition_normalization": {
            "policy": "source-condition population mean/std",
            "names": list(CONDITIONS),
            "mean": condition_mean.tolist(),
            "std": condition_std.tolist(),
            "clip": False,
            "raw_source_values": raw_conditions.tolist(),
        },
        "source_conditions": [list(value) for value in TRAIN_CONDITIONS],
        "axis_holdout_conditions": [list(value) for value in HOLDOUT_CONDITIONS],
        "off_axis_condition": "deferred and never used for model selection",
        "cases": cases,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "radaz_axis_factorial_manifest.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
