import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import predict_low_magnet_transfer_stride2 as transfer


DEFAULT_OUTDIR = (
    transfer.DEFAULT_OUTDIR / "quads_phi_out10"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Make 4-panel PNG frames and a GIF for the low-magnet transfer "
            "direct10 prediction."
        )
    )
    parser.add_argument("--h5", default=str(transfer.DEFAULT_H5))
    parser.add_argument("--workdir", default=str(transfer.DEFAULT_WORKDIR))
    parser.add_argument("--config", default=str(transfer.DEFAULT_CFG))
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--case-label", default="transfer case")
    parser.add_argument(
        "--base-dt-ns",
        type=float,
        default=transfer.BASE_DT_NS,
        help="Physical time represented by one raw PIC timestep index.",
    )
    parser.add_argument("--channel", type=int, default=transfer.PHI_INDEX)
    parser.add_argument("--channel-name", default="phi")
    parser.add_argument(
        "--output-index",
        type=int,
        default=10,
        help="1-based future frame index to visualize. For direct10, use 1..10.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=10,
        help=(
            "Use every Nth true-input window. With stride2 data, 10 means one "
            "GIF frame every 250 ns."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum PNG/GIF frames to export. 0 means all selected windows.",
    )
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--gif-duration-ms", type=int, default=90)
    return parser.parse_args()


def selected_starts(n_time, pre, aft, step, max_frames):
    starts = np.arange(0, n_time - pre - aft + 1, step, dtype=np.int64)
    if max_frames and max_frames > 0:
        starts = starts[:max_frames]
    return starts


@torch.inference_mode()
def predict_selected(model, data, starts, output_index0, channel, device, batch_size):
    input_last = []
    trues = []
    preds = []

    for b0 in range(0, len(starts), batch_size):
        b_starts = starts[b0:b0 + batch_size]
        x_np, y_np = transfer.batch_windows(data, b_starts, transfer.AFT)
        x = torch.from_numpy(x_np).to(device)
        pred = model(x).detach().cpu().numpy()
        input_last.append(x_np[:, transfer.PRE - 1, channel])
        trues.append(y_np[:, output_index0, channel])
        preds.append(pred[:, output_index0, channel])
        print(f"[PRED] {min(b0 + batch_size, len(starts))}/{len(starts)} frames", flush=True)

    return (
        np.concatenate(input_last, axis=0),
        np.concatenate(trues, axis=0),
        np.concatenate(preds, axis=0),
    )


def robust_range(x, q_low=1.0, q_high=99.0):
    vmin, vmax = np.percentile(np.asarray(x), [q_low, q_high]).astype(float)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(x))
        vmax = float(np.max(x))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def save_quad(path, in_last, true, pred, title, vmin, vmax, evmin, evmax, cmap, dpi):
    err = np.abs(pred - true)
    mse = float(np.mean((pred - true) ** 2))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    ims = [
        axes[0].imshow(in_last, vmin=vmin, vmax=vmax, cmap=cmap),
        axes[1].imshow(true, vmin=vmin, vmax=vmax, cmap=cmap),
        axes[2].imshow(pred, vmin=vmin, vmax=vmax, cmap=cmap),
        axes[3].imshow(err, vmin=evmin, vmax=evmax, cmap="magma"),
    ]
    axes[0].set_title("Input last")
    axes[1].set_title("True")
    axes[2].set_title("Pred")
    axes[3].set_title("|Error|")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(ims[0], ax=axes[:3], shrink=0.80, location="bottom", pad=0.03)
    fig.colorbar(ims[3], ax=axes[3], shrink=0.80, location="bottom", pad=0.03)
    fig.suptitle(f"{title} | MSE={mse:.4e}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return mse


def make_gif(png_paths, gif_path, duration_ms):
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in png_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for frame in frames:
        frame.close()


def write_readme(outdir, args, n_frames, gif_path):
    readme = f"""# Low-Magnet Transfer Quad GIF

This folder visualizes the zero-shot transfer experiment:

```text
high-magnet testcase 3b stride2 direct10 model
-> low-magnet testcase 3a stride2 data
```

Each PNG frame has four panels:

```text
Input last | True | Pred | |Error|
```

The model input is always true low-magnet PIC data, so this is teacher-forced
direct prediction, not rollout. The visualized output frame is
`out{args.output_index}`.

With stride2 data, one retained frame is 25 ns. Therefore:

```text
out{args.output_index} horizon = {args.output_index * 25} ns from the last input frame
PNG/GIF frame interval = {args.start_step * 25} ns of simulation time
```

GIF:

```text
{gif_path.name}
```

# 日本語メモ

このフォルダは、high magnet test case 3b で学習した stride2 direct10
モデルを low magnet test case 3a にそのまま適用した結果を、4枚並べで
可視化したものです。

各フレームは以下の4枚です。

```text
最後の入力フレーム | 真値 | 予測 | 絶対誤差
```

入力には常に low magnet 3a の真値 PIC フレームを使っているため、これは
rollout ではありません。予測値を次の入力には戻していません。

今回のGIFでは `out{args.output_index}` を時系列に並べています。stride2 なので、
`out{args.output_index}` は最後の入力フレームから {args.output_index * 25} ns 先の予測です。

出力PNG/GIFフレーム数: {n_frames}
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def write_readme_current(outdir, args, n_frames, gif_path, retained_dt_ns):
    horizon_ns = args.output_index * retained_dt_ns
    gif_interval_ns = args.start_step * retained_dt_ns
    readme = f"""# Low-Magnet Transfer Quad GIF

This folder visualizes the zero-shot transfer experiment:

```text
high-magnet testcase 3b direct10 model
-> low-magnet testcase 3a data
```

Each PNG frame has four panels:

```text
Input last | True | Pred | |Error|
```

The model input is always true low-magnet PIC data, so this is teacher-forced
direct prediction, not rollout. The visualized output frame is
`out{args.output_index}`.

One retained frame is {retained_dt_ns:g} ns. Therefore:

```text
out{args.output_index} horizon = {horizon_ns:g} ns from the last input frame
PNG/GIF frame interval = {gif_interval_ns:g} ns of simulation time
```

GIF:

```text
{gif_path.name}
```

# 日本語メモ

このフォルダは、high magnet test case 3b で学習した direct10 モデルを
low magnet test case 3a にそのまま適用したゼロショット転移結果を、
4枚並べで可視化したものです。

各フレームは次の4枚です。

```text
最後の入力フレーム | 真値 | 予測 | 絶対誤差
```

入力には常に low magnet 3a の真値PICフレームを使っています。
予測結果を次の入力へ戻していないため、これは rollout ではなく
teacher-forced direct prediction です。

保持フレーム間隔は {retained_dt_ns:g} ns、表示対象は `out{args.output_index}` です。
最後の入力から予測対象までの時間は {horizon_ns:g} ns、
PNG/GIFの隣接フレーム間隔は {gif_interval_ns:g} ns です。

出力PNG/GIFフレーム数: {n_frames}
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def main():
    args = parse_args()
    if not (1 <= args.output_index <= transfer.AFT):
        raise ValueError(f"--output-index must be 1..{transfer.AFT}")
    if args.start_step <= 0:
        raise ValueError("--start-step must be positive")

    outdir = Path(args.outdir)
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    device = transfer.resolve_device(args.device)
    h5_path = Path(args.h5)
    workdir = Path(args.workdir)
    ckpt = Path(args.ckpt) if args.ckpt else workdir / "checkpoints" / "best.ckpt"

    data, timesteps = transfer.load_tchw(h5_path)
    starts = selected_starts(
        data.shape[0],
        transfer.PRE,
        transfer.AFT,
        args.start_step,
        args.max_frames,
    )
    target_indices = starts + transfer.PRE + (args.output_index - 1)
    print(f"[DATA] {h5_path}")
    print(f"[DATA] data={data.shape}, selected_frames={len(starts)}")

    model = transfer.build_model(args.config, ckpt, device, transfer.AFT)
    in_last, true, pred = predict_selected(
        model,
        data,
        starts,
        args.output_index - 1,
        args.channel,
        device,
        args.batch_size,
    )

    vmin, vmax = robust_range(np.concatenate([true[:, None], in_last[:, None]], axis=1))
    err = np.abs(pred - true)
    evmin, evmax = robust_range(err, q_low=0.0, q_high=99.0)
    print(f"[SCALE] field vmin={vmin:.6g} vmax={vmax:.6g}")
    print(f"[SCALE] error vmin={evmin:.6g} vmax={evmax:.6g}")

    png_paths = []
    rows = []
    for i, (start, target_idx) in enumerate(zip(starts, target_indices)):
        target_step = int(timesteps[target_idx])
        target_time_us = float(target_step * args.base_dt_ns / 1000.0)
        horizon_ns = float((target_step - timesteps[start + transfer.PRE - 1]) * args.base_dt_ns)
        title = (
            f"{args.case_label} transfer | {args.channel_name} | "
            f"target={target_time_us:.3f} us | out{args.output_index} ({horizon_ns:.0f} ns)"
        )
        png = frames_dir / f"frame_{i:04d}_t{target_time_us:07.3f}us_out{args.output_index:02d}_{args.channel_name}.png"
        mse = save_quad(
            png,
            in_last[i],
            true[i],
            pred[i],
            title,
            vmin,
            vmax,
            evmin,
            evmax,
            args.cmap,
            args.dpi,
        )
        png_paths.append(png)
        rows.append({
            "frame_index": i,
            "window_start": int(start),
            "target_index": int(target_idx),
            "target_timestep": target_step,
            "target_time_us": target_time_us,
            "horizon_ns": horizon_ns,
            "mse": mse,
        })
        if (i + 1) % 25 == 0 or (i + 1) == len(starts):
            print(f"[PNG] {i + 1}/{len(starts)}", flush=True)

    gif_path = outdir / f"low_magnet_transfer_out{args.output_index:02d}_{args.channel_name}_quad.gif"
    make_gif(png_paths, gif_path, args.gif_duration_ms)
    print(f"[GIF] {gif_path}")

    csv_path = outdir / f"low_magnet_transfer_out{args.output_index:02d}_{args.channel_name}_quad_frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {csv_path}")

    retained_dt_ns = float(np.median(np.diff(timesteps))) * args.base_dt_ns
    write_readme_current(
        outdir,
        args,
        len(png_paths),
        gif_path,
        retained_dt_ns,
    )
    print(f"[README] {outdir / 'README.md'}")


if __name__ == "__main__":
    main()
