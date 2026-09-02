#!/usr/bin/env python3
"""Create train-only normalization manifests for bidirectional RadAz transfer."""

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
OUTPUT = ROOT / "workdirs" / "2D_RadAz" / "radaz_regime_generalization_manifests"
CHANNELS = ("electron_den", "ion_den", "phi")
E_VALUES = (10.0, 20.0, 22.5, 25.0, 30.0, 40.0)
POOLS = {
    "low": (10.0, 20.0),
    "high": (30.0, 40.0),
}


def case_token(ez_kvm: float) -> str:
    return str(int(ez_kvm)) if float(ez_kvm).is_integer() else str(ez_kvm)


def case_name(ez_kvm: float) -> str:
    return f"2D_RadAz_Xe1p_Bx20mT_Ez{case_token(ez_kvm)}kVm_dt15ps_out15ns"


def source_h5(ez_kvm: float) -> Path:
    name = case_name(ez_kvm)
    return RESULTS / name / name / "analysis_fields_uncompressed.h5"


def scan_pool(e_values, train_ratio: float, chunk_frames: int):
    train_min = np.full(len(CHANNELS), np.inf, dtype=np.float64)
    train_max = np.full(len(CHANNELS), -np.inf, dtype=np.float64)
    frame_counts = {}
    for ez_kvm in e_values:
        path = source_h5(ez_kvm)
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            nt = int(len(handle["axes/time_s"]))
            train_end = int(np.floor(nt * train_ratio))
            frame_counts[f"E{case_token(ez_kvm)}"] = {
                "frames": nt,
                "train_end_exclusive": train_end,
            }
            print(
                f"[SCAN] pool cases={list(e_values)} E={ez_kvm:g} "
                f"train_frames=0..{train_end - 1}",
                flush=True,
            )
            for start in range(0, train_end, chunk_frames):
                stop = min(start + chunk_frames, train_end)
                for channel_index, channel in enumerate(CHANNELS):
                    values = np.asarray(
                        handle[f"fields/{channel}"][start:stop, :257, :256]
                    )
                    if not np.all(np.isfinite(values)):
                        raise ValueError(
                            f"Non-finite {channel} values in {path} frames {start}:{stop}"
                        )
                    train_min[channel_index] = min(
                        train_min[channel_index], float(np.min(values))
                    )
                    train_max[channel_index] = max(
                        train_max[channel_index], float(np.max(values))
                    )
    return train_min, train_max, frame_counts


def case_entry(ez_kvm, norm_low, norm_high):
    return {
        "case_key": f"E{case_token(ez_kvm)}",
        "label": f"Bx=20 mT, Ez={ez_kvm:g} kV/m",
        "B_mT": 20.0,
        "Ez_kVm": float(ez_kvm),
        "dt_frame_ns": 15.0,
        "path": str(source_h5(ez_kvm)),
        "format": "radaz_consolidated",
        "channels": list(CHANNELS),
        "spatial_stride": 1,
        "model_height": 260,
        "model_width": 256,
        "normalization_low": norm_low.tolist(),
        "normalization_high": norm_high.tolist(),
        "normalization_clip": False,
        "splits": ["train", "val", "test"],
    }


def build_manifest(pool_name, train_ratio, val_ratio, test_ratio, margin, chunk_frames):
    source_values = POOLS[pool_name]
    train_min, train_max, frame_counts = scan_pool(
        source_values, train_ratio, chunk_frames
    )
    span = train_max - train_min
    if np.any(span <= 0.0):
        raise ValueError(f"Non-positive source-pool span for {pool_name}: {span}")
    norm_low = train_min - margin * span
    norm_high = train_max + margin * span
    return {
        "name": f"radaz_bx20mt_{pool_name}_regime_direct10",
        "description": (
            "Label-free RadAz source-pool training. Each case is split 8:1:1 "
            "before windows are mixed. Normalization uses only the source "
            "cases' training frames and is applied without clipping."
        ),
        "pool": pool_name,
        "source_Ez_kVm": list(source_values),
        "pre_seq_length": 10,
        "aft_seq_length": 10,
        "split": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "policy": "per-case frame-disjoint",
        },
        "normalization": {
            "policy": "source-pool train-only per-channel affine",
            "channels": list(CHANNELS),
            "train_min": train_min.tolist(),
            "train_max": train_max.tolist(),
            "margin": margin,
            "low": norm_low.tolist(),
            "high": norm_high.tolist(),
            "clip": False,
            "frame_counts": frame_counts,
        },
        "cases": [case_entry(ez, norm_low, norm_high) for ez in source_values],
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

    args.output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for pool_name in POOLS:
        manifest = build_manifest(
            pool_name,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.margin,
            args.chunk_frames,
        )
        output_path = args.output / f"radaz_{pool_name}_source_manifest.json"
        output_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        manifests[pool_name] = str(output_path)
        print(f"[PASS] wrote {output_path}", flush=True)

    protocol = {
        "research_question": (
            "Can a label-free SimVPv2 trained on one side of the electric-field "
            "sweep predict unseen dynamics on the other instability-regime side?"
        ),
        "model": "SimVPv2 gSTA data-only direct10, stride1, 15 ns/frame",
        "channels": list(CHANNELS),
        "forward": {
            "train": [10.0, 20.0],
            "intermediate_diagnostics": [22.5, 25.0, 30.0],
            "final_zero_shot": 40.0,
        },
        "reverse": {
            "train": [30.0, 40.0],
            "intermediate_diagnostics": [25.0, 22.5, 20.0],
            "final_zero_shot": 10.0,
        },
        "bridge_models_deferred_until_base_models_are_evaluated": {
            "forward": [10.0, 20.0, 25.0],
            "reverse": [30.0, 40.0, 25.0],
            "requirement": "match the number of source training windows",
        },
        "available_cases_Ez_kVm": list(E_VALUES),
        "manifests": manifests,
    }
    protocol_path = args.output / "experiment_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] wrote {protocol_path}", flush=True)


if __name__ == "__main__":
    main()
