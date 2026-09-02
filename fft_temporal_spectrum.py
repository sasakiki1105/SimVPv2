import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def parse_names(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_signals(text):
    signals = [x.strip() for x in text.split(",") if x.strip()]
    if signals == ["all"]:
        return ["pixel_power", "spatial_mean", "spatial_rms"]
    allowed = {"pixel_power", "spatial_mean", "spatial_rms"}
    unknown = [x for x in signals if x not in allowed]
    if unknown:
        raise ValueError(f"unknown signal(s): {unknown}; allowed={sorted(allowed)}")
    return signals


def as_str_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    try:
        return [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in x]
    except Exception:
        return [str(x)]


def decode_scalar(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode()
    if hasattr(x, "shape") and x.shape == ():
        return decode_scalar(x.item())
    return x


def canon_to_tchw(f, key):
    props = as_str_list(f["props"][()]) if "props" in f else None
    timesteps = f["timesteps"][()] if "timesteps" in f else None
    layout = decode_scalar(f["layout"][()]) if "layout" in f else None

    if key in f:
        raw_key = key
    elif "data_tchw" in f:
        raw_key = "data_tchw"
    elif "data" in f:
        raw_key = "data"
    else:
        raise KeyError(f"H5 must contain '{key}', 'data_tchw', or 'data'")

    x = np.asarray(f[raw_key][()], dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"{raw_key} must be 4D, got {x.shape}")

    if timesteps is not None:
        t_len = len(timesteps)
        if x.shape[0] == t_len and x.shape[1] <= 20:
            out = x
        elif x.shape[3] == t_len and x.shape[2] <= 20:
            out = np.transpose(x, (3, 2, 0, 1))
        elif x.shape[0] == t_len and x.shape[3] <= 20:
            out = np.transpose(x, (0, 3, 1, 2))
        else:
            raise ValueError(f"cannot infer layout from shape {x.shape} with {t_len} timesteps")
    else:
        s = x.shape
        if s[0] >= 20 and s[1] <= 20 and s[2] >= 16 and s[3] >= 16:
            out = x
        elif s[0] >= 16 and s[1] >= 16 and s[2] <= 20 and s[3] >= 20:
            out = np.transpose(x, (3, 2, 0, 1))
        elif s[0] <= 20 and s[1] >= 16 and s[2] >= 16 and s[3] >= 20:
            out = np.transpose(x, (3, 0, 1, 2))
        elif s[0] >= 20 and s[1] >= 16 and s[2] >= 16 and s[3] <= 20:
            out = np.transpose(x, (0, 3, 1, 2))
        else:
            raise ValueError(f"cannot infer layout from shape {s}")

    timestep_meta = None
    if timesteps is not None:
        ts = np.asarray(timesteps)
        timestep_meta = {
            "length": int(len(ts)),
            "first": decode_scalar(ts[0]) if len(ts) else None,
            "last": decode_scalar(ts[-1]) if len(ts) else None,
        }

    return np.ascontiguousarray(out), {
        "raw_key": raw_key,
        "raw_shape": list(x.shape),
        "props": props,
        "timesteps": timestep_meta,
        "layout": layout,
    }


def load_tchw_from_h5(path, key):
    with h5py.File(path, "r") as f:
        return canon_to_tchw(f, key)


def select_frame_range(data, frame_range, train_ratio, val_ratio):
    t = data.shape[0]
    train_end = int(np.floor(t * train_ratio))
    val_end = int(np.floor(t * (train_ratio + val_ratio)))

    if frame_range == "all":
        start, stop = 0, t
    elif frame_range == "train":
        start, stop = 0, train_end
    elif frame_range == "val":
        start, stop = train_end, val_end
    elif frame_range == "test":
        start, stop = val_end, t
    else:
        parts = frame_range.split(":")
        if len(parts) != 2:
            raise ValueError("custom frame range must be START:STOP")
        start = 0 if parts[0] == "" else int(parts[0])
        stop = t if parts[1] == "" else int(parts[1])

    if not (0 <= start < stop <= t):
        raise ValueError(f"invalid frame range {start}:{stop} for T={t}")
    return data[start:stop], {
        "frame_range": frame_range,
        "frame_start": start,
        "frame_stop_exclusive": stop,
        "n_frames": stop - start,
        "split_train_stop_exclusive": train_end,
        "split_val_stop_exclusive": val_end,
    }


def read_constant_b(b_file):
    if not b_file:
        return None
    values = []
    with open(b_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                values.append(float(parts[3]))
    if not values:
        return None
    return float(np.mean(values))


def load_reference_frequencies(sim_input=None, b_file=None, n0=None, b_tesla=None):
    refs = []

    if sim_input:
        with open(sim_input, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        b_mean = read_constant_b(b_file)
        for particle in cfg.get("Particles", []):
            name = particle.get("name", "particle")
            particle_n0 = float(particle.get("n0", 0.0))
            q = float(particle.get("q", 0.0))
            m = float(particle.get("m", 0.0))
            append_species_refs(refs, name, particle_n0, q, m, b_mean)

    b_from_file = read_constant_b(b_file)
    b_value = b_tesla if b_tesla is not None else b_from_file
    if n0 is not None:
        append_species_refs(refs, "electron", n0, -E_CHARGE, M_E, b_value)
        append_species_refs(refs, "ion", n0, E_CHARGE, M_P, b_value)

    return refs


def append_species_refs(refs, name, n0, q, m, b_tesla):
    if n0 > 0 and q != 0 and m > 0:
        f_plasma = np.sqrt(n0 * q * q / (EPS0 * m)) / (2.0 * np.pi)
        refs.append({
            "name": f"{name} plasma",
            "frequency_hz": float(f_plasma),
            "frequency_mhz": float(f_plasma / 1.0e6),
            "period_us": float(1.0e6 / f_plasma),
        })
    if b_tesla is not None and q != 0 and m > 0:
        f_cyclotron = abs(q) * abs(b_tesla) / (2.0 * np.pi * m)
        refs.append({
            "name": f"{name} cyclotron",
            "frequency_hz": float(f_cyclotron),
            "frequency_mhz": float(f_cyclotron / 1.0e6),
            "period_us": float(1.0e6 / f_cyclotron),
            "B_T": float(b_tesla),
        })


def apply_window(signal, use_window):
    if use_window:
        window = np.hanning(signal.shape[0]).astype(np.float64)
        norm = float(np.sum(window * window))
        return signal * window.reshape((-1,) + (1,) * (signal.ndim - 1)), norm
    return signal, float(signal.shape[0])


def temporal_power_spectrum(field_tyx, dt_s, signal_kind, use_window=True):
    field = field_tyx.astype(np.float64, copy=False)

    if signal_kind == "pixel_power":
        x = field - np.mean(field, axis=0, keepdims=True)
        x, norm = apply_window(x, use_window)
        fft = np.fft.rfft(x, axis=0)
        power = np.mean(np.abs(fft) ** 2, axis=(1, 2)) / max(norm, 1.0)
    elif signal_kind == "spatial_mean":
        x = np.mean(field, axis=(1, 2))
        x = x - np.mean(x)
        x, norm = apply_window(x, use_window)
        fft = np.fft.rfft(x)
        power = np.abs(fft) ** 2 / max(norm, 1.0)
    elif signal_kind == "spatial_rms":
        centered = field - np.mean(field, axis=(1, 2), keepdims=True)
        x = np.sqrt(np.mean(centered * centered, axis=(1, 2)))
        x = x - np.mean(x)
        x, norm = apply_window(x, use_window)
        fft = np.fft.rfft(x)
        power = np.abs(fft) ** 2 / max(norm, 1.0)
    else:
        raise ValueError(f"unknown signal kind: {signal_kind}")

    freqs = np.fft.rfftfreq(field.shape[0], d=dt_s)
    return freqs, power


def normalize_power(power, mode):
    if mode == "none":
        return power.copy()
    positive = power[1:] if len(power) > 1 else power
    if mode == "max":
        denom = float(np.nanmax(positive)) if positive.size else float(np.nanmax(power))
    elif mode == "sum":
        denom = float(np.nansum(positive)) if positive.size else float(np.nansum(power))
    else:
        raise ValueError(f"unknown normalize mode: {mode}")
    if not np.isfinite(denom) or denom <= 0:
        return np.full_like(power, np.nan, dtype=np.float64)
    return power / denom


def candidate_mask(freqs, min_freq_hz=None, max_freq_hz=None):
    mask = freqs > 0
    if min_freq_hz is not None:
        mask &= freqs >= min_freq_hz
    if max_freq_hz is not None:
        mask &= freqs <= max_freq_hz
    return mask


def find_top_peaks(freqs, power, top_n, min_separation_bins, min_freq_hz=None, max_freq_hz=None):
    valid = candidate_mask(freqs, min_freq_hz, max_freq_hz)
    candidates = []
    for i in range(1, len(power) - 1):
        if not valid[i]:
            continue
        if power[i] >= power[i - 1] and power[i] >= power[i + 1]:
            candidates.append(i)

    if not candidates:
        candidates = [i for i in range(len(power)) if valid[i]]

    selected = []
    for idx in sorted(candidates, key=lambda i: power[i], reverse=True):
        if all(abs(idx - j) >= min_separation_bins for j in selected):
            selected.append(idx)
        if len(selected) >= top_n:
            break
    return selected


def nearest_reference(freq_hz, refs):
    if not refs:
        return None
    ref = min(refs, key=lambda r: abs(np.log(freq_hz / r["frequency_hz"])))
    return {
        "nearest_reference": ref["name"],
        "nearest_reference_mhz": ref["frequency_hz"] / 1.0e6,
        "ratio_to_nearest_reference": freq_hz / ref["frequency_hz"],
    }


def write_spectrum_csv(out_csv, freqs, power, power_norm):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "frequency_mhz", "period_us", "power", "power_norm"])
        for f_hz, p, pn in zip(freqs, power, power_norm):
            period_us = np.inf if f_hz == 0 else 1.0e6 / f_hz
            writer.writerow([float(f_hz), float(f_hz / 1.0e6), float(period_us), float(p), float(pn)])


def plot_spectrum(
    out_png,
    freqs,
    power_to_plot,
    peak_indices,
    channel_name,
    signal_kind,
    refs,
    min_freq_mhz=None,
    max_freq_mhz=None,
    log_y=True,
    normalize_label="normalized power",
):
    mask = candidate_mask(
        freqs,
        None if min_freq_mhz is None else min_freq_mhz * 1.0e6,
        None if max_freq_mhz is None else max_freq_mhz * 1.0e6,
    )

    plt.figure(figsize=(10, 5.5))
    x_mhz = freqs[mask] / 1.0e6
    y = power_to_plot[mask]
    if log_y:
        y = np.maximum(y, np.finfo(np.float64).tiny)
        plt.semilogy(x_mhz, y, linewidth=1.2)
    else:
        plt.plot(x_mhz, y, linewidth=1.2)

    for idx in peak_indices:
        if idx >= len(freqs):
            continue
        if not mask[idx]:
            continue
        yy = max(power_to_plot[idx], np.finfo(np.float64).tiny) if log_y else power_to_plot[idx]
        plt.scatter(freqs[idx] / 1.0e6, yy, s=28, zorder=3)
        plt.annotate(
            f"{freqs[idx] / 1.0e6:.3g} MHz",
            (freqs[idx] / 1.0e6, yy),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ymax = float(np.nanmax(y)) if y.size else 1.0
    ymin = float(np.nanmin(y)) if y.size else 0.0
    for ref in refs:
        f_mhz = ref["frequency_hz"] / 1.0e6
        if min_freq_mhz is not None and f_mhz < min_freq_mhz:
            continue
        if max_freq_mhz is not None and f_mhz > max_freq_mhz:
            continue
        plt.axvline(f_mhz, linestyle="--", linewidth=1.0, alpha=0.6)
        plt.text(f_mhz, ymax, ref["name"], rotation=90, va="top", ha="right", fontsize=8)

    plt.xlabel("Frequency (MHz)")
    plt.ylabel(normalize_label)
    plt.title(f"Temporal FFT spectrum: {channel_name} / {signal_kind}")
    if not log_y and ymin < ymax:
        plt.ylim(bottom=max(0.0, ymin))
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Compute temporal FFT spectra from PEPAPIC H5 data.")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--key", default="data_tchw")
    ap.add_argument("--dt-ns", type=float, required=True, help="Physical time between retained frames in ns.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--channels", default="0,1,2")
    ap.add_argument("--chan-names", default="electron_den,ion_den,phi")
    ap.add_argument("--signals", default="pixel_power", help="pixel_power, spatial_mean, spatial_rms, or all")
    ap.add_argument("--range", dest="frame_range", default="train", help="train, val, test, all, or START:STOP")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--min-separation-bins", type=int, default=3)
    ap.add_argument("--min-freq-mhz", type=float, default=None)
    ap.add_argument("--max-freq-mhz", type=float, default=None)
    ap.add_argument("--normalize", choices=["max", "sum", "none"], default="max")
    ap.add_argument("--linear-y", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--sim-input", default=None)
    ap.add_argument("--b-file", default=None)
    ap.add_argument("--n0", type=float, default=None, help="Reference density for electron/ion plasma frequencies.")
    ap.add_argument("--b-tesla", type=float, default=None, help="Reference B for cyclotron frequencies.")
    return ap


def main():
    args = build_arg_parser().parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_all, h5_meta = load_tchw_from_h5(args.h5, args.key)
    data, range_meta = select_frame_range(data_all, args.frame_range, args.train_ratio, args.val_ratio)
    channels = parse_int_list(args.channels)
    names = parse_names(args.chan_names)
    signals = parse_signals(args.signals)
    dt_s = args.dt_ns * 1.0e-9
    min_freq_hz = None if args.min_freq_mhz is None else args.min_freq_mhz * 1.0e6
    max_freq_hz = None if args.max_freq_mhz is None else args.max_freq_mhz * 1.0e6
    refs = load_reference_frequencies(args.sim_input, args.b_file, args.n0, args.b_tesla)

    print(f"[INFO] raw H5 shape={h5_meta['raw_shape']} -> TCHW shape={list(data_all.shape)}")
    print(f"[INFO] selected frames {range_meta['frame_start']}:{range_meta['frame_stop_exclusive']} -> {list(data.shape)}")
    print(f"[INFO] dt={args.dt_ns} ns, Nyquist={0.5 / dt_s / 1.0e6:.6g} MHz")

    summary_rows = []
    spectra_meta = {
        "h5": args.h5,
        "key": args.key,
        "h5_meta": h5_meta,
        "data_shape_tchw": list(data_all.shape),
        "selected_shape_tchw": list(data.shape),
        "range": range_meta,
        "dt_ns": args.dt_ns,
        "duration_us_between_first_last": float((data.shape[0] - 1) * args.dt_ns / 1000.0),
        "frequency_resolution_mhz": float(1.0 / (data.shape[0] * dt_s) / 1.0e6),
        "nyquist_mhz": float(0.5 / dt_s / 1.0e6),
        "channels": channels,
        "channel_names": names,
        "signals": signals,
        "normalize": args.normalize,
        "use_hanning_window": not args.no_window,
        "references": refs,
    }

    for c in channels:
        if c < 0 or c >= data.shape[1]:
            raise ValueError(f"channel {c} out of range for data shape {data.shape}")
        cname = names[c] if c < len(names) else f"ch{c}"

        for signal_kind in signals:
            print(f"[INFO] FFT channel {c}: {cname}, signal={signal_kind}")
            freqs, power = temporal_power_spectrum(
                data[:, c],
                dt_s,
                signal_kind=signal_kind,
                use_window=not args.no_window,
            )
            power_norm = normalize_power(power, args.normalize)
            peak_indices = find_top_peaks(
                freqs,
                power,
                args.top_n,
                args.min_separation_bins,
                min_freq_hz=min_freq_hz,
                max_freq_hz=max_freq_hz,
            )

            stem = f"temporal_fft_c{c:02d}_{cname}_{signal_kind}"
            spectrum_csv = outdir / f"{stem}.csv"
            peaks_csv = outdir / f"{stem}_peaks.csv"
            out_png = outdir / f"{stem}.png"

            write_spectrum_csv(spectrum_csv, freqs, power, power_norm)

            peak_rows = []
            pos_power_sum = float(np.sum(power[candidate_mask(freqs, min_freq_hz, max_freq_hz)]))
            for rank, idx in enumerate(peak_indices, start=1):
                f_hz = float(freqs[idx])
                row = {
                    "channel": c,
                    "channel_name": cname,
                    "signal": signal_kind,
                    "rank": rank,
                    "frequency_hz": f_hz,
                    "frequency_mhz": f_hz / 1.0e6,
                    "period_us": 1.0e6 / f_hz,
                    "period_frames": 1.0 / (f_hz * dt_s),
                    "power": float(power[idx]),
                    "power_norm": float(power_norm[idx]),
                    "relative_power_in_selected_band": float(power[idx] / pos_power_sum) if pos_power_sum > 0 else np.nan,
                }
                nearest = nearest_reference(f_hz, refs)
                if nearest is not None:
                    row.update(nearest)
                peak_rows.append(row)
                summary_rows.append(row)

            with open(peaks_csv, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "channel",
                    "channel_name",
                    "signal",
                    "rank",
                    "frequency_hz",
                    "frequency_mhz",
                    "period_us",
                    "period_frames",
                    "power",
                    "power_norm",
                    "relative_power_in_selected_band",
                    "nearest_reference",
                    "nearest_reference_mhz",
                    "ratio_to_nearest_reference",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(peak_rows)

            plot_spectrum(
                out_png,
                freqs,
                power_norm if args.normalize != "none" else power,
                peak_indices,
                cname,
                signal_kind,
                refs,
                min_freq_mhz=args.min_freq_mhz,
                max_freq_mhz=args.max_freq_mhz,
                log_y=not args.linear_y,
                normalize_label="Power / max positive-bin power" if args.normalize == "max" else (
                    "Power / positive-bin power sum" if args.normalize == "sum" else "Power"
                ),
            )
            print(f"[PLOT] {out_png}")
            print(f"[CSV]  {spectrum_csv}")
            print(f"[CSV]  {peaks_csv}")

    summary_csv = outdir / "temporal_fft_top_peaks.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "channel",
            "channel_name",
            "signal",
            "rank",
            "frequency_hz",
            "frequency_mhz",
            "period_us",
            "period_frames",
            "power",
            "power_norm",
            "relative_power_in_selected_band",
            "nearest_reference",
            "nearest_reference_mhz",
            "ratio_to_nearest_reference",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(outdir / "temporal_fft_metadata.json", "w", encoding="utf-8") as f:
        json.dump(spectra_meta, f, indent=2)

    print(f"[DONE] wrote {summary_csv}")


if __name__ == "__main__":
    main()
