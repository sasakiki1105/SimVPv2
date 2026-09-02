"""Create a time-concatenated virtual HDF5 view of RadAz transition segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_h5", type=Path)
    parser.add_argument("source_h5", type=Path, nargs="+")
    return parser.parse_args()


def dataset_names(handle: h5py.File, group: str) -> tuple[str, ...]:
    if group not in handle:
        return ()
    return tuple(
        sorted(
            name
            for name, value in handle[group].items()
            if isinstance(value, h5py.Dataset)
        )
    )


def main() -> None:
    args = parse_args()
    output = args.output_h5.resolve()
    sources = [path.resolve() for path in args.source_h5]
    if len(sources) < 1:
        raise ValueError("At least one source HDF5 is required")
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = []
    field_names = None
    x_reference = None
    y_reference = None
    previous_time = None
    previous_frame = None
    frame_dt = None
    with h5py.File(sources[0], "r") as first:
        field_names = dataset_names(first, "fields")
        static_names = dataset_names(first, "static_fields")
        if not field_names:
            raise ValueError(f"No fields found in {sources[0]}")
        x_reference = np.asarray(first["axes/x_m"], dtype=np.float64)
        y_reference = np.asarray(first["axes/y_m"], dtype=np.float64)
        field_shapes = {
            name: tuple(first[f"fields/{name}"].shape[1:])
            for name in field_names
        }
        field_dtypes = {
            name: first[f"fields/{name}"].dtype for name in field_names
        }

    all_time = []
    all_frame = []
    total = 0
    for path in sources:
        with h5py.File(path, "r") as handle:
            names = dataset_names(handle, "fields")
            if names != field_names:
                raise ValueError(f"Field schema mismatch in {path}")
            x_m = np.asarray(handle["axes/x_m"], dtype=np.float64)
            y_m = np.asarray(handle["axes/y_m"], dtype=np.float64)
            if not np.array_equal(x_m, x_reference) or not np.array_equal(
                y_m, y_reference
            ):
                raise ValueError(f"Spatial axes mismatch in {path}")
            time_s = np.asarray(handle["axes/time_s"], dtype=np.float64)
            frame = np.asarray(handle["axes/frame_id"], dtype=np.int64)
            if len(time_s) != len(frame):
                raise ValueError(f"Time/frame length mismatch in {path}")
            if not np.all(np.diff(time_s) > 0.0) or not np.all(
                np.diff(frame) == 1
            ):
                raise ValueError(f"Non-contiguous source segment in {path}")
            current_dt = float(np.median(np.diff(time_s)))
            if frame_dt is None:
                frame_dt = current_dt
            elif not np.isclose(current_dt, frame_dt, rtol=1.0e-9, atol=1.0e-15):
                raise ValueError(f"Frame interval mismatch in {path}")
            if previous_time is not None:
                if int(frame[0]) != int(previous_frame) + 1:
                    raise ValueError(
                        f"Frame gap between segments: {previous_frame} -> {frame[0]}"
                    )
                if not np.isclose(
                    float(time_s[0] - previous_time),
                    frame_dt,
                    rtol=1.0e-9,
                    atol=1.0e-15,
                ):
                    raise ValueError("Time gap between transition segments")
            for name in field_names:
                dataset = handle[f"fields/{name}"]
                if tuple(dataset.shape[1:]) != field_shapes[name]:
                    raise ValueError(f"Shape mismatch for {name} in {path}")
                if dataset.dtype != field_dtypes[name]:
                    raise ValueError(f"Dtype mismatch for {name} in {path}")
            metadata.append(
                {
                    "path": str(path),
                    "frames": int(len(frame)),
                    "first_frame": int(frame[0]),
                    "last_frame": int(frame[-1]),
                    "first_time_s": float(time_s[0]),
                    "last_time_s": float(time_s[-1]),
                    "offset": total,
                }
            )
            total += len(frame)
            all_time.append(time_s)
            all_frame.append(frame)
            previous_time = float(time_s[-1])
            previous_frame = int(frame[-1])

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w", libver="latest") as target:
        target.attrs["format"] = "PEPAPIC RadAz transition virtual concatenation"
        target.attrs["format_version"] = 1
        target.attrs["completed"] = True
        target.attrs["source_files_json"] = json.dumps(metadata)
        target.attrs["frame_dt_s"] = frame_dt
        axes = target.require_group("axes")
        axes.create_dataset("time_s", data=np.concatenate(all_time))
        axes.create_dataset("frame_id", data=np.concatenate(all_frame))
        axes.create_dataset("x_m", data=x_reference)
        axes.create_dataset("y_m", data=y_reference)
        fields = target.require_group("fields")
        for name in field_names:
            layout = h5py.VirtualLayout(
                shape=(total,) + field_shapes[name], dtype=field_dtypes[name]
            )
            offset = 0
            for path, info in zip(sources, metadata):
                length = int(info["frames"])
                virtual = h5py.VirtualSource(
                    str(path),
                    f"fields/{name}",
                    shape=(length,) + field_shapes[name],
                )
                layout[offset : offset + length] = virtual
                offset += length
            fields.create_virtual_dataset(name, layout)
        if static_names:
            static = target.require_group("static_fields")
            with h5py.File(sources[0], "r") as first:
                for name in static_names:
                    reference = np.asarray(first[f"static_fields/{name}"])
                    for path in sources[1:]:
                        with h5py.File(path, "r") as current:
                            if not np.array_equal(
                                reference,
                                np.asarray(current[f"static_fields/{name}"]),
                            ):
                                raise ValueError(
                                    f"Static field {name} differs in {path}"
                                )
                    static.create_dataset(name, data=reference)

    report = {
        "status": "PASS",
        "output": str(output),
        "virtual": True,
        "frames": total,
        "first_frame": int(all_frame[0][0]),
        "last_frame": int(all_frame[-1][-1]),
        "first_time_us": float(all_time[0][0] * 1.0e6),
        "last_time_us": float(all_time[-1][-1] * 1.0e6),
        "frame_dt_ns": float(frame_dt * 1.0e9),
        "fields": list(field_names),
        "sources": metadata,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"PASS: {total} frames, {report['first_time_us']:.3f}--"
        f"{report['last_time_us']:.3f} us -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
