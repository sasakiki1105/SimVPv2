import argparse
from pathlib import Path

import numpy as np
import torch

import predict_low_magnet_transfer_stride2 as transfer


def selected_starts(n_time, pre, aft, step, max_frames):
    starts = np.arange(0, n_time - pre - aft + 1, step, dtype=np.int64)
    if max_frames and max_frames > 0:
        starts = starts[:max_frames]
    return starts


def mse(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def corr(a, b):
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return np.nan
    return float(np.sum(x * y) / denom)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Check whether predicted frames are closer to earlier/later true PIC frames."
    )
    parser.add_argument("--h5", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--config", default=str(transfer.DEFAULT_CFG))
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-index", type=int, default=10)
    parser.add_argument("--channel", type=int, default=transfer.PHI_INDEX)
    parser.add_argument("--start-step", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--lag-min", type=int, default=-10)
    parser.add_argument("--lag-max", type=int, default=10)
    args = parser.parse_args()

    output_index0 = args.output_index - 1
    if output_index0 < 0 or output_index0 >= transfer.AFT:
        raise ValueError(f"--output-index must be 1..{transfer.AFT}")

    device = transfer.resolve_device(args.device)
    data, timesteps = transfer.load_tchw(args.h5)
    starts = selected_starts(data.shape[0], transfer.PRE, transfer.AFT, args.start_step, args.max_frames)
    target_indices = starts + transfer.PRE + output_index0
    ckpt = Path(args.ckpt) if args.ckpt else Path(args.workdir) / "checkpoints" / "best.ckpt"
    model = transfer.build_model(args.config, ckpt, device)

    results = []
    lags = np.arange(args.lag_min, args.lag_max + 1, dtype=np.int64)
    mse_by_lag = {int(lag): [] for lag in lags}
    corr_by_lag = {int(lag): [] for lag in lags}

    for b0 in range(0, len(starts), args.batch_size):
        b_starts = starts[b0:b0 + args.batch_size]
        b_targets = target_indices[b0:b0 + args.batch_size]
        x_np, _ = transfer.batch_windows(data, b_starts)
        pred = model(torch.from_numpy(x_np).to(device)).detach().cpu().numpy()

        for bi, target_idx in enumerate(b_targets):
            p = pred[bi, output_index0, args.channel]
            per_lag = []
            for lag in lags:
                true_idx = int(target_idx + lag)
                if true_idx < 0 or true_idx >= data.shape[0]:
                    continue
                t = data[true_idx, args.channel]
                lag_i = int(lag)
                m = mse(p, t)
                c = corr(p, t)
                mse_by_lag[lag_i].append(m)
                corr_by_lag[lag_i].append(c)
                per_lag.append((lag_i, true_idx, m, c))

            best_mse = min(per_lag, key=lambda row: row[2])
            best_corr = max(per_lag, key=lambda row: row[3])
            results.append({
                "start": int(b_starts[bi]),
                "target_index": int(target_idx),
                "target_timestep": int(timesteps[target_idx]),
                "best_mse_lag": best_mse[0],
                "best_mse_true_index": best_mse[1],
                "best_mse": best_mse[2],
                "best_corr_lag": best_corr[0],
                "best_corr_true_index": best_corr[1],
                "best_corr": best_corr[3],
            })
        print(f"[PRED] {min(b0 + args.batch_size, len(starts))}/{len(starts)}", flush=True)

    print("CONFIG")
    print(f"h5={args.h5}")
    print(f"workdir={args.workdir}")
    print(f"output_index={args.output_index}")
    print(f"start_step={args.start_step}")
    print(f"n_windows={len(results)}")
    if len(timesteps) >= 2:
        print(f"retained_dt_ns={(timesteps[1] - timesteps[0]) * transfer.BASE_DT_NS:g}")
    print()

    best_mse_lags = np.asarray([r["best_mse_lag"] for r in results], dtype=np.int64)
    best_corr_lags = np.asarray([r["best_corr_lag"] for r in results], dtype=np.int64)
    print("BEST_LAG_COUNTS_BY_MSE")
    for lag, count in sorted(zip(*np.unique(best_mse_lags, return_counts=True))):
        print(f"{int(lag):+d} {int(count)}")
    print(f"mean_best_mse_lag={float(np.mean(best_mse_lags)):.4g}")
    print(f"median_best_mse_lag={float(np.median(best_mse_lags)):.4g}")
    print()

    print("BEST_LAG_COUNTS_BY_CORR")
    for lag, count in sorted(zip(*np.unique(best_corr_lags, return_counts=True))):
        print(f"{int(lag):+d} {int(count)}")
    print(f"mean_best_corr_lag={float(np.mean(best_corr_lags)):.4g}")
    print(f"median_best_corr_lag={float(np.median(best_corr_lags)):.4g}")
    print()

    print("MEAN_BY_LAG")
    print("lag mean_mse mean_corr n")
    for lag in lags:
        vals = np.asarray(mse_by_lag[int(lag)], dtype=np.float64)
        cors = np.asarray(corr_by_lag[int(lag)], dtype=np.float64)
        if vals.size:
            print(f"{int(lag):+d} {float(np.mean(vals)):.8g} {float(np.nanmean(cors)):.8g} {int(vals.size)}")


if __name__ == "__main__":
    main()
