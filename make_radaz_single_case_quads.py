#!/usr/bin/env python3
"""Create four-panel physical-unit comparison PNGs for the RadAz direct10 run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CASE_NAME = "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
CASE_ROOT = (
    ROOT.parent
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / CASE_NAME
    / CASE_NAME
)
DATA_H5 = (
    CASE_ROOT
    / "SimVPv2_inputs"
    / "radaz_3ch_trainfixed_margin20_native257x256_pad260x256.h5"
)
EXPERIMENT = (
    "radaz_xe1p_bx20mt_ez10kvm_out15ns_native257x256_"
    "direct10_trainfixed_disjoint_811_bs1_100ep"
)
SAVED = ROOT / "workdirs" / EXPERIMENT / "saved"
DEFAULT_OUTPUT = (
    ROOT
    / "workdirs"
    / "compare_radaz_bx20mt_ez10kvm_native_direct10"
    / "comparison_png_phi_out10"
)
CHANNELS = ("electron_den", "ion_den", "phi")
UNITS = (r"$\mathrm{m}^{-3}$", r"$\mathrm{m}^{-3}$", "V")
VALID_H = 257
VALID_W = 256
PRE = 10
AFT = 10
TEST_START = 1800
FRAME_INTERVAL_NS = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, default=SAVED)
    parser.add_argument("--h5", type=Path, default=DATA_H5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--channel", type=int, default=2, choices=range(3))
    parser.add_argument("--output-index", type=int, default=10, choices=range(1, 11))
    parser.add_argument(
        "--sample-step",
        type=int,
        default=10,
        help="Keep every Nth test window. The final test window is always included.",
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def robust_range(values: np.ndarray, low_q: float, high_q: float) -> tuple[float, float]:
    vmin, vmax = np.percentile(values, [low_q, high_q]).astype(float)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if vmax <= vmin:
        vmax = vmin + 1.0e-12
    return vmin, vmax


def save_quad(
    path: Path,
    input_last: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    extent_cm: tuple[float, float, float, float],
    field_range: tuple[float, float],
    error_range: tuple[float, float],
    channel_name: str,
    unit: str,
    target_time_us: float,
    horizon_ns: float,
    dpi: int,
) -> tuple[float, float]:
    error = np.abs(prediction - truth)
    model_mse = float(np.mean((prediction - truth) ** 2, dtype=np.float64))
    copy_mse = float(np.mean((input_last - truth) ** 2, dtype=np.float64))
    ratio = model_mse / copy_mse

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.5), constrained_layout=True)
    common = dict(
        origin="lower",
        extent=extent_cm,
        aspect="equal",
        cmap="viridis",
        vmin=field_range[0],
        vmax=field_range[1],
    )
    images = [
        axes[0].imshow(input_last, **common),
        axes[1].imshow(truth, **common),
        axes[2].imshow(prediction, **common),
        axes[3].imshow(
            error,
            origin="lower",
            extent=extent_cm,
            aspect="equal",
            cmap="magma",
            vmin=error_range[0],
            vmax=error_range[1],
        ),
    ]
    for axis, title in zip(
        axes,
        ("Input last (copy)", "True PIC", "SimVPv2", "|Prediction - truth|"),
    ):
        axis.set_title(title)
        axis.set_xlabel("Azimuthal position [cm]")
    axes[0].set_ylabel("Radial position [cm]")
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)

    fig.colorbar(
        images[0],
        ax=axes[:3],
        shrink=0.82,
        location="bottom",
        pad=0.04,
        label=f"{channel_name} [{unit}]",
    )
    fig.colorbar(
        images[3],
        ax=axes[3],
        shrink=0.82,
        location="bottom",
        pad=0.04,
        label=f"absolute error [{unit}]",
    )
    fig.suptitle(
        f"{CASE_NAME} | {channel_name} | target={target_time_us:.3f} us | "
        f"direct out{int(horizon_ns / FRAME_INTERVAL_NS)} ({horizon_ns:.0f} ns)\n"
        f"MSE: model={model_mse:.4e}, copy={copy_mse:.4e}, "
        f"model/copy={ratio:.3f}"
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return model_mse, copy_mse


def main() -> None:
    args = parse_args()
    if args.sample_step <= 0:
        raise ValueError("--sample-step must be positive")

    inputs = np.load(args.saved / "inputs.npy", mmap_mode="r")
    predictions = np.load(args.saved / "preds.npy", mmap_mode="r")
    truths = np.load(args.saved / "trues.npy", mmap_mode="r")
    expected_tail = (AFT, 3, 260, 256)
    for name, values in (
        ("inputs", inputs),
        ("preds", predictions),
        ("trues", truths),
    ):
        if values.ndim != 5 or tuple(values.shape[1:]) != expected_tail:
            raise ValueError(f"{name} has unexpected shape {values.shape}")

    sample_indices = np.arange(0, len(inputs), args.sample_step, dtype=np.int64)
    if sample_indices[-1] != len(inputs) - 1:
        sample_indices = np.append(sample_indices, len(inputs) - 1)
    output_index0 = args.output_index - 1

    with h5py.File(args.h5, "r") as handle:
        low = np.asarray(handle["norm_low"], dtype=np.float64)
        high = np.asarray(handle["norm_high"], dtype=np.float64)
        time_s = np.asarray(handle["time_s"], dtype=np.float64)
        x_m = np.asarray(handle["x_m"], dtype=np.float64)
        y_m = np.asarray(handle["y_m"], dtype=np.float64)
        valid_shape = tuple(np.asarray(handle["valid_spatial_shape"], dtype=int))
    if valid_shape != (VALID_H, VALID_W):
        raise ValueError(f"Unexpected valid shape {valid_shape}")

    channel = args.channel
    scale = high[channel] - low[channel]

    def physical(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * scale + low[channel]

    input_last = physical(
        inputs[sample_indices, PRE - 1, channel, :VALID_H, :VALID_W]
    )
    truth = physical(
        truths[sample_indices, output_index0, channel, :VALID_H, :VALID_W]
    )
    prediction = physical(
        predictions[sample_indices, output_index0, channel, :VALID_H, :VALID_W]
    )

    field_values = np.concatenate(
        (input_last.ravel(), truth.ravel(), prediction.ravel())
    )
    field_range = robust_range(field_values, 0.5, 99.5)
    error_values = np.abs(prediction - truth)
    error_range = robust_range(error_values, 0.0, 99.5)
    extent_cm = (
        float(y_m[0] * 100.0),
        float((y_m[-1] + np.median(np.diff(y_m))) * 100.0),
        float(x_m[0] * 100.0),
        float(x_m[-1] * 100.0),
    )

    frames_dir = args.output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    starts = TEST_START + sample_indices
    target_frames = starts + PRE + output_index0
    horizon_ns = args.output_index * FRAME_INTERVAL_NS

    rows: list[dict] = []
    for index, sample in enumerate(sample_indices):
        target_frame = int(target_frames[index])
        target_time_us = float(time_s[target_frame] * 1.0e6)
        output_path = frames_dir / (
            f"frame_{index:03d}_sample{sample:03d}_"
            f"t{target_time_us:06.3f}us_out{args.output_index:02d}_"
            f"{CHANNELS[channel]}.png"
        )
        model_mse, copy_mse = save_quad(
            output_path,
            input_last[index],
            truth[index],
            prediction[index],
            extent_cm,
            field_range,
            error_range,
            CHANNELS[channel],
            UNITS[channel],
            target_time_us,
            horizon_ns,
            args.dpi,
        )
        rows.append(
            {
                "png": output_path.name,
                "saved_sample_index": int(sample),
                "window_start_frame": int(starts[index]),
                "target_frame": target_frame,
                "target_time_us": target_time_us,
                "output_index": args.output_index,
                "horizon_ns": horizon_ns,
                "model_mse": model_mse,
                "copy_mse": copy_mse,
                "model_over_copy": model_mse / copy_mse,
            }
        )
        print(f"[PNG] {index + 1}/{len(sample_indices)} {output_path.name}", flush=True)

    with (args.output / "comparison_frames.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    readme = f"""# RadAz direct10 comparison PNG

## 日本語

`{CASE_NAME}` の未学習test区間について、間引きなしで学習した
SimVPv2のdirect10予測を4枚並べで可視化したものです。

```text
最後の入力フレーム（copy baseline） | 真値PIC | SimVPv2予測 | 絶対誤差
```

- 表示物理量: `{CHANNELS[channel]}`
- 予測位置: `out{args.output_index}`、最後の入力から {horizon_ns:g} ns先
- test区間: 27-30 us
- PNG間隔: 原則 {args.sample_step * FRAME_INTERVAL_NS:g} ns
- PNG枚数: {len(rows)}
- 場・誤差は正規化値ではなく物理単位
- radial padding 3行は除外し、有効257x256点だけを表示
- 横軸: azimuthal、縦軸: radial

各予測はtest区間内の真値10枚を入力するteacher-forced direct predictionです。
予測を次の入力へ戻すrolloutではありません。

`comparison_frames.csv`には各PNGのtarget時刻、model/copy MSEとその比を保存しています。
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(f"[PASS] output={args.output}")


if __name__ == "__main__":
    main()
