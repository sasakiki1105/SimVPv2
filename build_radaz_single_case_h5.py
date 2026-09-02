#!/usr/bin/env python3
"""Build a train-normalized SimVPv2 H5 from consolidated RadAz PIC fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


CHANNELS = ("electron_den", "ion_den", "phi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_h5", type=Path)
    parser.add_argument("output_h5", type=Path)
    parser.add_argument("--spatial-stride", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=2001,
        help="Expected source length; use 0 to accept an arbitrary transition length.",
    )
    parser.add_argument(
        "--normalization-h5",
        type=Path,
        default=None,
        help="Reuse normalization arrays from a training H5 for zero-shot transfer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source_h5 = args.source_h5.resolve()
    args.output_h5 = args.output_h5.resolve()
    if args.normalization_h5 is not None:
        args.normalization_h5 = args.normalization_h5.resolve()
    if args.spatial_stride <= 0:
        raise ValueError("--spatial-stride must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between zero and one")

    with h5py.File(args.source_h5, "r") as source:
        missing = [name for name in CHANNELS if f"fields/{name}" not in source]
        if missing:
            raise KeyError(f"Missing fields in {args.source_h5}: {missing}")
        time_s = np.asarray(source["axes/time_s"], dtype=np.float64)
        x_m = np.asarray(source["axes/x_m"], dtype=np.float64)
        y_m = np.asarray(source["axes/y_m"], dtype=np.float64)
        nt = len(time_s)
        if args.expected_frames > 0 and nt != args.expected_frames:
            raise ValueError(
                f"Expected {args.expected_frames} frames, got {nt}"
            )
        if len(x_m) != 257 or len(y_m) != 257:
            raise ValueError(f"Expected a 257x257 stitched grid, got {len(x_m)}x{len(y_m)}")
        if not np.allclose(
            source["fields/phi"][:, :, -1],
            source["fields/phi"][:, :, 0],
            rtol=1.0e-5,
            atol=1.0e-8,
        ):
            raise ValueError("The last y endpoint is not the expected duplicate of y=0")

        if 256 % args.spatial_stride:
            raise ValueError("The unique 256-point azimuthal grid must be divisible by stride")
        native_grid = args.spatial_stride == 1
        valid_height = 257 if native_grid else 256 // args.spatial_stride
        valid_width = 256 // args.spatial_stride
        height = 260 if native_grid else valid_height
        width = valid_width
        train_end = int(np.floor(nt * args.train_ratio))
        normalization_source = "target training interval"
        if args.normalization_h5 is not None:
            with h5py.File(args.normalization_h5, "r") as normalization:
                source_channels = tuple(
                    item.decode() if isinstance(item, bytes) else str(item)
                    for item in np.asarray(normalization["props"])
                )
                if source_channels != CHANNELS:
                    raise ValueError(
                        f"Normalization channels {source_channels} != {CHANNELS}"
                    )
                source_valid_shape = tuple(
                    np.asarray(normalization["valid_spatial_shape"], dtype=int)
                )
                source_model_shape = tuple(
                    np.asarray(normalization["model_spatial_shape"], dtype=int)
                )
                if source_valid_shape != (valid_height, valid_width):
                    raise ValueError(
                        f"Normalization valid shape {source_valid_shape} != "
                        f"{(valid_height, valid_width)}"
                    )
                if source_model_shape != (height, width):
                    raise ValueError(
                        f"Normalization model shape {source_model_shape} != "
                        f"{(height, width)}"
                    )
                train_min = np.asarray(
                    normalization["train_min"], dtype=np.float64
                )
                train_max = np.asarray(
                    normalization["train_max"], dtype=np.float64
                )
                norm_low = np.asarray(
                    normalization["norm_low"], dtype=np.float64
                )
                norm_high = np.asarray(
                    normalization["norm_high"], dtype=np.float64
                )
            normalization_source = str(args.normalization_h5)
        else:
            train_min = np.full(len(CHANNELS), np.inf, dtype=np.float64)
            train_max = np.full(len(CHANNELS), -np.inf, dtype=np.float64)

            for start in range(0, train_end, args.chunk_frames):
                stop = min(start + args.chunk_frames, train_end)
                for channel_index, name in enumerate(CHANNELS):
                    if native_grid:
                        values = np.asarray(
                            source[f"fields/{name}"][start:stop, :257, :256]
                        )
                    else:
                        raw = np.asarray(
                            source[f"fields/{name}"][start:stop, :256, :256]
                        )
                        values = raw.reshape(
                            stop - start,
                            valid_height,
                            args.spatial_stride,
                            valid_width,
                            args.spatial_stride,
                        ).mean(axis=(2, 4))
                    train_min[channel_index] = min(
                        train_min[channel_index], float(np.min(values))
                    )
                    train_max[channel_index] = max(
                        train_max[channel_index], float(np.max(values))
                    )

            span = train_max - train_min
            if np.any(span <= 0.0):
                raise ValueError(f"Non-positive training span: {span}")
            norm_low = train_min - args.margin * span
            norm_high = train_max + args.margin * span

        if np.any(norm_high <= norm_low):
            raise ValueError("Normalization bounds must have positive spans")

        args.output_h5.parent.mkdir(parents=True, exist_ok=True)
        clipped_low_count = np.zeros(len(CHANNELS), dtype=np.int64)
        clipped_high_count = np.zeros(len(CHANNELS), dtype=np.int64)
        value_count = np.zeros(len(CHANNELS), dtype=np.int64)
        with h5py.File(args.output_h5, "w") as output:
            data = output.create_dataset(
                "data_tchw",
                shape=(nt, len(CHANNELS), height, width),
                dtype=np.float32,
                chunks=(1, len(CHANNELS), height, width),
                compression="gzip",
                compression_opts=4,
            )
            for start in range(0, nt, args.chunk_frames):
                stop = min(start + args.chunk_frames, nt)
                chunk = np.empty(
                    (stop - start, len(CHANNELS), height, width), dtype=np.float32
                )
                for channel_index, name in enumerate(CHANNELS):
                    if native_grid:
                        valid = np.asarray(
                            source[f"fields/{name}"][start:stop, :257, :256],
                            dtype=np.float64,
                        )
                        raw = np.pad(
                            valid,
                            ((0, 0), (0, height - valid_height), (0, 0)),
                            mode="edge",
                        )
                    else:
                        source_values = np.asarray(
                            source[f"fields/{name}"][start:stop, :256, :256],
                            dtype=np.float64,
                        )
                        raw = source_values.reshape(
                            stop - start,
                            valid_height,
                            args.spatial_stride,
                            valid_width,
                            args.spatial_stride,
                        ).mean(axis=(2, 4))
                    clipped_low_count[channel_index] += np.count_nonzero(
                        raw < norm_low[channel_index]
                    )
                    clipped_high_count[channel_index] += np.count_nonzero(
                        raw > norm_high[channel_index]
                    )
                    value_count[channel_index] += raw.size
                    chunk[:, channel_index] = np.clip(
                        (raw - norm_low[channel_index])
                        / (norm_high[channel_index] - norm_low[channel_index]),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                data[start:stop] = chunk
                if stop == nt or stop % 200 < args.chunk_frames:
                    print(f"[BUILD] {stop}/{nt} frames", flush=True)

            if "axes/frame_id" in source:
                timesteps = np.asarray(source["axes/frame_id"], dtype=np.int64)
            else:
                timesteps = np.arange(nt, dtype=np.int64)
            output.create_dataset("timesteps", data=timesteps)
            output.create_dataset("time_s", data=time_s)
            if native_grid:
                output.create_dataset("x_m", data=x_m)
                output.create_dataset(
                    "x_m_model",
                    data=np.pad(x_m, (0, height - valid_height), mode="edge"),
                )
            else:
                output.create_dataset(
                    "x_m",
                    data=x_m[:256]
                    .reshape(valid_height, args.spatial_stride)
                    .mean(axis=1),
                )
            output.create_dataset(
                "y_m",
                data=y_m[:256]
                .reshape(valid_width, args.spatial_stride)
                .mean(axis=1),
            )
            output.create_dataset("props", data=np.asarray(CHANNELS, dtype="S"))
            output.create_dataset("train_min", data=train_min.astype(np.float32))
            output.create_dataset("train_max", data=train_max.astype(np.float32))
            output.create_dataset("norm_low", data=norm_low.astype(np.float32))
            output.create_dataset("norm_high", data=norm_high.astype(np.float32))
            output.create_dataset("margin", data=np.float32(args.margin))
            output.create_dataset("train_frame_end_exclusive", data=np.int32(train_end))
            output.create_dataset("spatial_stride", data=np.int32(args.spatial_stride))
            output.create_dataset(
                "valid_spatial_shape", data=np.asarray([valid_height, valid_width])
            )
            output.create_dataset(
                "model_spatial_shape", data=np.asarray([height, width])
            )
            output.create_dataset("pre_seq_length", data=np.int32(10))
            output.create_dataset("aft_seq_length", data=np.int32(10))
            output.create_dataset(
                "layout", data=np.bytes_("data_tchw[T,C,x_radial,y_azimuthal]")
            )
            output.create_dataset("source_h5", data=np.bytes_(str(args.source_h5)))
            output.create_dataset(
                "normalization_source", data=np.bytes_(normalization_source)
            )
            output.create_dataset("clipped_low_count", data=clipped_low_count)
            output.create_dataset("clipped_high_count", data=clipped_high_count)
            output.create_dataset("normalization_value_count", data=value_count)
            output.attrs["periodic_duplicate_endpoint_removed"] = True
            output.attrs["spatial_reduction"] = (
                "none; radial edge padded from 257 to 260 for model divisibility"
                if native_grid
                else "non-overlapping mean pooling"
            )
            output.attrs["normalization"] = "per-channel train-only minmax with margin"

    split = {
        "frames": nt,
        "train_frame_range": [0, train_end - 1],
        "validation_frame_range": [train_end, int(np.floor(nt * 0.9)) - 1],
        "test_frame_range": [int(np.floor(nt * 0.9)), nt - 1],
        "train_time_us": [float(time_s[0] * 1.0e6), float(time_s[train_end - 1] * 1.0e6)],
        "validation_time_us": [
            float(time_s[train_end] * 1.0e6),
            float(time_s[int(np.floor(nt * 0.9)) - 1] * 1.0e6),
        ],
        "test_time_us": [
            float(time_s[int(np.floor(nt * 0.9))] * 1.0e6),
            float(time_s[-1] * 1.0e6),
        ],
        "shape_tchw": [nt, len(CHANNELS), height, width],
        "valid_spatial_shape": [valid_height, valid_width],
        "model_padding": [height - valid_height, width - valid_width],
        "channels": list(CHANNELS),
        "train_min": train_min.tolist(),
        "train_max": train_max.tolist(),
        "normalization_low": norm_low.tolist(),
        "normalization_high": norm_high.tolist(),
        "normalization_source": normalization_source,
        "clipped_low_fraction": (
            clipped_low_count / np.maximum(value_count, 1)
        ).tolist(),
        "clipped_high_fraction": (
            clipped_high_count / np.maximum(value_count, 1)
        ).tolist(),
    }
    args.output_h5.with_suffix(".json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[PASS] wrote {args.output_h5}")
    print(json.dumps(split, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
