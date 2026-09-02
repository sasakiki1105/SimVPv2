import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def parse_names(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_frames(text):
    return [int(x) for x in text.split(",") if x.strip()]


def parse_offsets(text):
    if ":" in text:
        start, stop = [int(x) for x in text.split(":")]
        step = 1 if stop >= start else -1
        return list(range(start, stop + step, step))
    return [int(x) for x in text.split(",") if x.strip()]


def load_tchw(path, key):
    with h5py.File(path, "r") as f:
        x = np.asarray(f[key][()], dtype=np.float64)

    if x.ndim != 4:
        raise ValueError(f"expected 4D array in {path}, got {x.shape}")

    s = x.shape
    if s[0] >= 20 and s[1] <= 20 and s[2] >= 16 and s[3] >= 16:
        return np.ascontiguousarray(x)
    if s[0] >= 16 and s[1] >= 16 and s[2] <= 20 and s[3] >= 20:
        return np.ascontiguousarray(np.transpose(x, (3, 2, 0, 1)))
    if s[0] <= 20 and s[1] >= 16 and s[2] >= 16 and s[3] >= 20:
        return np.ascontiguousarray(np.transpose(x, (3, 0, 1, 2)))
    if s[0] >= 20 and s[1] >= 16 and s[2] >= 16 and s[3] <= 20:
        return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))

    raise ValueError(f"cannot infer H5 layout from shape {s} in {path}")


def load_minmax_meta(path):
    with h5py.File(path, "r") as f:
        if "train_min" not in f or "train_max" not in f:
            return None
        margin = float(f["margin"][()]) if "margin" in f else 0.0
        return {
            "train_min": np.asarray(f["train_min"][()], dtype=np.float64),
            "train_max": np.asarray(f["train_max"][()], dtype=np.float64),
            "margin": margin,
        }


def denorm_minmax_channel(x, meta, ch):
    mn = float(meta["train_min"][ch])
    mx = float(meta["train_max"][ch])
    margin = float(meta["margin"])
    r = mx - mn
    mn2 = mn - margin * r
    mx2 = mx + margin * r
    return x * (mx2 - mn2) + mn2


def corr2d(a, b):
    aa = a.ravel() - float(np.mean(a))
    bb = b.ravel() - float(np.mean(b))
    den = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if den <= 0:
        return np.nan
    return float(np.sum(aa * bb) / den)


def normalized_frame_diff_stats(x_tchw, n_frames):
    n = min(n_frames, x_tchw.shape[0] - 1)
    d = np.diff(x_tchw[: n + 1], axis=0)
    rms_d = np.sqrt(np.mean(d * d, axis=(2, 3)))
    rms_x = np.sqrt(np.mean(x_tchw[:n] * x_tchw[:n], axis=(2, 3)))
    ratio = rms_d / np.maximum(rms_x, 1.0e-12)
    return ratio


def main():
    ap = argparse.ArgumentParser(description="Compare two H5 time series frame-by-frame at matching physical times.")
    ap.add_argument("--a", required=True, help="First H5 path.")
    ap.add_argument("--b", required=True, help="Second H5 path.")
    ap.add_argument("--key", default="data_tchw")
    ap.add_argument("--channel-names", default="electron_den,ion_den,phi")
    ap.add_argument("--dt-ns", type=float, default=50.0)
    ap.add_argument("--frames", default="0,1,2,10,100,500,800,900,950,999")
    ap.add_argument("--summary-ranges", default="train:0:800,test_like:900:1000,all_overlap:0:1000")
    ap.add_argument("--b-index-scale", type=int, default=1, help="Compare a[k] to b[k * scale + offset].")
    ap.add_argument("--scan-b-offsets", default=None, help="Comma list or START:STOP offsets for b index scan.")
    ap.add_argument("--charge-imbalance", action="store_true", help="Also compare denormalized ion_den - electron_den.")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)
    a = load_tchw(a_path, args.key)
    b = load_tchw(b_path, args.key)
    a_minmax = load_minmax_meta(a_path)
    b_minmax = load_minmax_meta(b_path)
    n = min(a.shape[0], b.shape[0] // max(args.b_index_scale, 1))
    c = min(a.shape[1], b.shape[1])
    names = parse_names(args.channel_names)

    print(f"[INFO] a: {a_path}")
    print(f"[INFO] b: {b_path}")
    print(f"[INFO] shape a={a.shape}, b={b.shape}, overlap_frames={n}, overlap_channels={c}")
    print(f"[INFO] b_index = a_frame * {args.b_index_scale} + offset")

    rows = []
    print("\n[FRAME_CORR]")
    for k in parse_frames(args.frames):
        if not (0 <= k < n):
            continue
        j = k * args.b_index_scale
        if not (0 <= j < b.shape[0]):
            continue
        values = []
        for ch in range(c):
            name = names[ch] if ch < len(names) else f"ch{ch}"
            r = corr2d(a[k, ch], b[j, ch])
            values.append(f"{name}={r: .4f}")
            rows.append({
                "kind": "frame",
                "label": "",
                "frame": k,
                "time_us": k * args.dt_ns / 1000.0,
                "channel": ch,
                "channel_name": name,
                "corr": r,
                "mean": "",
                "min": "",
                "max": "",
            })
        print(f"frame {k:4d} time_us={k * args.dt_ns / 1000.0:7.3f}  " + "  ".join(values))

    print("\n[RANGE_CORR]")
    for spec in [x.strip() for x in args.summary_ranges.split(",") if x.strip()]:
        label, start, stop = spec.split(":")
        start = int(start)
        stop = min(int(stop), n)
        for ch in range(c):
            name = names[ch] if ch < len(names) else f"ch{ch}"
            vals = np.asarray(
                [
                    corr2d(a[k, ch], b[k * args.b_index_scale, ch])
                    for k in range(start, stop)
                    if 0 <= k * args.b_index_scale < b.shape[0]
                ],
                dtype=np.float64,
            )
            row = {
                "kind": "range",
                "label": label,
                "frame": "",
                "time_us": "",
                "channel": ch,
                "channel_name": name,
                "corr": "",
                "mean": float(np.nanmean(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            }
            rows.append(row)
            print(
                f"{label:11s} {name:12s} "
                f"mean={row['mean']: .4f} min={row['min']: .4f} max={row['max']: .4f}"
            )

    if args.scan_b_offsets:
        offsets = parse_offsets(args.scan_b_offsets)
        print("\n[OFFSET_SCAN] mean frame correlation for a[k] vs b[k * scale + offset]")
        for spec in [x.strip() for x in args.summary_ranges.split(",") if x.strip()]:
            label, start, stop = spec.split(":")
            start = int(start)
            stop = min(int(stop), n)
            print(f"\nrange={label} frames={start}:{stop}")
            for ch in range(c):
                name = names[ch] if ch < len(names) else f"ch{ch}"
                best = None
                for off in offsets:
                    vals = []
                    for k in range(start, stop):
                        j = k * args.b_index_scale + off
                        if 0 <= j < b.shape[0]:
                            vals.append(corr2d(a[k, ch], b[j, ch]))
                    if not vals:
                        continue
                    m = float(np.nanmean(vals))
                    if best is None or m > best[1]:
                        best = (off, m)
                if best is not None:
                    print(f"{name:12s} best_offset={best[0]:4d} mean_corr={best[1]: .4f}")

    print("\n[CONSECUTIVE_DIFF] normalized RMS change per frame")
    for label, x in [("a", a), ("b", b)]:
        ratio = normalized_frame_diff_stats(x[:, :c], min(n, 1000))
        for ch in range(c):
            name = names[ch] if ch < len(names) else f"ch{ch}"
            r = ratio[:, ch]
            print(
                f"{label:2s} {name:12s} "
                f"mean={np.mean(r):.4g} median={np.median(r):.4g} p95={np.quantile(r, 0.95):.4g}"
            )

    if args.charge_imbalance:
        if a_minmax is None or b_minmax is None:
            print("\n[CHARGE_IMBALANCE] skipped: train_min/train_max missing in one of the H5 files")
        elif c < 2:
            print("\n[CHARGE_IMBALANCE] skipped: need electron_den and ion_den channels")
        else:
            a_ne = denorm_minmax_channel(a[:, 0], a_minmax, 0)
            a_ni = denorm_minmax_channel(a[:, 1], a_minmax, 1)
            b_ne = denorm_minmax_channel(b[:, 0], b_minmax, 0)
            b_ni = denorm_minmax_channel(b[:, 1], b_minmax, 1)
            a_rho = a_ni - a_ne
            b_rho = b_ni - b_ne
            print("\n[CHARGE_IMBALANCE] corr of denormalized ion_den - electron_den")
            for spec in [x.strip() for x in args.summary_ranges.split(",") if x.strip()]:
                label, start, stop = spec.split(":")
                start = int(start)
                stop = min(int(stop), n)
                vals = np.asarray(
                    [
                        corr2d(a_rho[k], b_rho[k * args.b_index_scale])
                        for k in range(start, stop)
                        if 0 <= k * args.b_index_scale < b_rho.shape[0]
                    ],
                    dtype=np.float64,
                )
                print(
                    f"{label:11s} rho_i_minus_e "
                    f"mean={np.nanmean(vals): .4f} min={np.nanmin(vals): .4f} max={np.nanmax(vals): .4f}"
                )

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["kind", "label", "frame", "time_us", "channel", "channel_name", "corr", "mean", "min", "max"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[CSV] {args.out_csv}")


if __name__ == "__main__":
    main()
