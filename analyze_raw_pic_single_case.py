import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_low_magnet_h5_from_high_stats import (
    build_global_frame,
    build_layout,
    parse_timestep_spec,
)
from fft_temporal_spectrum import (
    find_top_peaks,
    normalize_power,
    temporal_power_spectrum,
)


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
RANKS = [0, 1, 2, 3]


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}", flush=True)


def load_phi_stack(case_folder, timesteps):
    domains, nx_tile, ny_tile, xmins, ymins = build_layout(
        case_folder,
        RANKS,
        trim_mode="training_compatible",
    )
    sample = build_global_frame(
        case_folder,
        domains,
        RANKS,
        timesteps[0],
        "phi",
        nx_tile,
        ny_tile,
        xmins,
        ymins,
        trim_mode="training_compatible",
    )
    phi = np.empty((len(timesteps),) + sample.shape, dtype=np.float32)
    phi[0] = sample
    for i, timestep in enumerate(timesteps[1:], start=1):
        if (i + 1) % 100 == 0 or i + 1 == len(timesteps):
            print(f"[LOAD] phi timestep {timestep} ({i + 1}/{len(timesteps)})", flush=True)
        phi[i] = build_global_frame(
            case_folder,
            domains,
            RANKS,
            timestep,
            "phi",
            nx_tile,
            ny_tile,
            xmins,
            ymins,
            trim_mode="training_compatible",
        )
    return phi


def compute_timeseries(case_key, label, phi, timesteps, dt_ns):
    rows = []
    efield_rms = []
    for i, timestep in enumerate(timesteps):
        frame = phi[i].astype(np.float64)
        gy, gx = np.gradient(frame)
        e_rms = float(np.sqrt(np.mean(gx * gx + gy * gy)))
        efield_rms.append(e_rms)
        rows.append({
            "case": case_key,
            "label": label,
            "timestep": int(timestep),
            "time_us": float(timestep * dt_ns / 1000.0),
            "phi_spatial_mean": float(np.mean(frame)),
            "phi_spatial_std": float(np.std(frame)),
            "phi_spatial_rms": float(np.sqrt(np.mean(frame * frame))),
            "phi_spatial_min": float(np.min(frame)),
            "phi_spatial_max": float(np.max(frame)),
            "efield_rms_raw_efx_efy": e_rms,
        })
    return rows, np.asarray(efield_rms, dtype=np.float64)


def compute_copy_horizon(case_key, label, phi, dt_ns, mean_std, mean_rms):
    candidate_frames = list(range(1, 41)) + [60, 80, 100, 160, 400, 800, 1600, 3200]
    candidate_frames = [h for h in candidate_frames if h < phi.shape[0]]
    rows = []
    for horizon in candidate_frames:
        diff = phi[horizon:].astype(np.float64) - phi[:-horizon].astype(np.float64)
        mse = float(np.mean(diff * diff))
        rmse = float(np.sqrt(mse))
        rows.append({
            "case": case_key,
            "label": label,
            "horizon_frames": int(horizon),
            "horizon_ns": float(horizon * dt_ns),
            "horizon_us": float(horizon * dt_ns / 1000.0),
            "phi_copy_mse": mse,
            "phi_copy_rmse": rmse,
            "phi_copy_nrmse_by_mean_std": float(rmse / mean_std) if mean_std > 0 else np.nan,
            "phi_copy_nrmse_by_mean_rms": float(rmse / mean_rms) if mean_rms > 0 else np.nan,
        })
    return rows


def write_fft(case_key, label, phi, dt_ns, outdir, top_n):
    dt_s = dt_ns * 1.0e-9
    freqs, power = temporal_power_spectrum(phi, dt_s, "pixel_power", use_window=True)
    power_norm = normalize_power(power, "max")
    peak_indices = find_top_peaks(freqs, power, top_n, min_separation_bins=2)

    spectrum_rows = []
    for freq, p, pn in zip(freqs, power, power_norm):
        spectrum_rows.append({
            "frequency_hz": float(freq),
            "frequency_mhz": float(freq / 1.0e6),
            "period_us": float(np.inf if freq == 0 else 1.0e6 / freq),
            "power": float(p),
            "power_norm": float(pn),
        })
    write_csv(spectrum_rows, outdir / "raw_pic_phi_fft_pixel_power.csv")

    peak_rows = []
    for rank, idx in enumerate(peak_indices, start=1):
        freq = float(freqs[idx])
        peak_rows.append({
            "case": case_key,
            "label": label,
            "rank": rank,
            "frequency_hz": freq,
            "frequency_mhz": freq / 1.0e6,
            "period_us": float(np.inf if freq == 0 else 1.0e6 / freq),
            "period_ns": float(np.inf if freq == 0 else 1.0e9 / freq),
            "period_frames": float(np.inf if freq == 0 else 1.0 / (freq * dt_s)),
            "power": float(power[idx]),
            "power_norm": float(power_norm[idx]),
        })
    write_csv(peak_rows, outdir / "raw_pic_phi_fft_top_peaks.csv")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    mask = freqs > 0
    ax.semilogy(freqs[mask] / 1.0e6, np.maximum(power_norm[mask], np.finfo(np.float64).tiny), linewidth=1.2)
    for row, idx in zip(peak_rows, peak_indices):
        ax.scatter(freqs[idx] / 1.0e6, max(power_norm[idx], np.finfo(np.float64).tiny), s=28)
        ax.annotate(
            f"{row['frequency_mhz']:.3g} MHz",
            (freqs[idx] / 1.0e6, max(power_norm[idx], np.finfo(np.float64).tiny)),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power / max positive-bin power")
    ax.set_title(f"{label}: temporal FFT of phi / pixel power")
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    fig.tight_layout()
    path = outdir / "raw_pic_phi_fft_pixel_power.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)
    return peak_rows


def plot_timeseries(label, rows, outdir):
    t = np.asarray([row["time_us"] for row in rows], dtype=np.float64)
    std = np.asarray([row["phi_spatial_std"] for row in rows], dtype=np.float64)
    rms = np.asarray([row["phi_spatial_rms"] for row in rows], dtype=np.float64)
    e = np.asarray([row["efield_rms_raw_efx_efy"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.2), sharex=True)
    axes[0].plot(t, rms, label="phi RMS", color="#2563eb", linewidth=1.4)
    axes[0].plot(t, std, label="phi spatial std", color="#dc2626", linewidth=1.4)
    axes[0].set_ylabel("phi value")
    axes[0].set_title(f"{label}: phi RMS/std and electric-field RMS")
    axes[0].grid(True, linestyle=":", alpha=0.55)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t, e, label="E RMS from -grad(phi)", color="#16a34a", linewidth=1.4)
    axes[1].set_xlabel("Simulation time (us)")
    axes[1].set_ylabel("raw gradient units")
    axes[1].grid(True, linestyle=":", alpha=0.55)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = outdir / "raw_pic_phi_rms_std_and_efield_rms.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def plot_copy(label, rows, outdir):
    x = np.asarray([row["horizon_ns"] for row in rows], dtype=np.float64)
    y = np.asarray([row["phi_copy_nrmse_by_mean_std"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(x, y, marker="o", color="#dc2626", linewidth=1.7)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Horizon (ns)")
    ax.set_ylabel("Copy RMSE / mean spatial std(phi)")
    ax.set_title(f"{label}: copy baseline collapse")
    ax.grid(True, linestyle=":", alpha=0.55)
    fig.tight_layout()
    path = outdir / "raw_pic_phi_copy_baseline_horizon.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] {path}", flush=True)


def first_collapse(rows, threshold):
    for row in rows:
        if row["phi_copy_nrmse_by_mean_std"] >= threshold:
            return {
                "horizon_frames": row["horizon_frames"],
                "horizon_ns": row["horizon_ns"],
                "horizon_us": row["horizon_us"],
                "copy_mse": row["phi_copy_mse"],
                "copy_nrmse_by_mean_std": row["phi_copy_nrmse_by_mean_std"],
            }
    return None


def write_readme(label, outdir):
    text = f"""# Raw PIC Analysis: {label}

This folder contains raw PIC diagnostics using the training-compatible stitching convention.

Main files:

- `raw_pic_phi_efield_timeseries.csv`
- `raw_pic_phi_rms_std_and_efield_rms.png`
- `raw_pic_phi_fft_pixel_power.csv`
- `raw_pic_phi_fft_pixel_power.png`
- `raw_pic_phi_fft_top_peaks.csv`
- `raw_pic_phi_copy_baseline_horizon.csv`
- `raw_pic_phi_copy_baseline_horizon.png`
- `raw_pic_metrics_summary.json`
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")
    print(f"[README] {outdir / 'README.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-folder", required=True)
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--b-mt", type=float, default=np.nan)
    parser.add_argument("--dt-ns", type=float, default=12.5)
    parser.add_argument("--timesteps", default="0:1:4000")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    case_folder = Path(args.case_folder)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timesteps = parse_timestep_spec(args.timesteps)

    phi = load_phi_stack(case_folder, timesteps)
    ts_rows, efield_rms = compute_timeseries(args.case_key, args.label, phi, timesteps, args.dt_ns)
    write_csv(ts_rows, outdir / "raw_pic_phi_efield_timeseries.csv")
    plot_timeseries(args.label, ts_rows, outdir)

    phi_std = np.asarray([row["phi_spatial_std"] for row in ts_rows], dtype=np.float64)
    phi_rms = np.asarray([row["phi_spatial_rms"] for row in ts_rows], dtype=np.float64)
    mean_std = float(np.mean(phi_std))
    mean_rms = float(np.mean(phi_rms))

    copy_rows = compute_copy_horizon(args.case_key, args.label, phi, args.dt_ns, mean_std, mean_rms)
    write_csv(copy_rows, outdir / "raw_pic_phi_copy_baseline_horizon.csv")
    plot_copy(args.label, copy_rows, outdir)

    peak_rows = write_fft(args.case_key, args.label, phi, args.dt_ns, outdir, args.top_n)

    summary = {
        "description": f"Raw PIC characterization for {args.label} using training_compatible stitching.",
        "case_key": args.case_key,
        "label": args.label,
        "case_folder": str(case_folder),
        "B_mT": args.b_mt,
        "dt_ns": args.dt_ns,
        "frame_count": int(phi.shape[0]),
        "duration_us": float((timesteps[-1] - timesteps[0]) * args.dt_ns / 1000.0),
        "grid_shape_xy": [int(phi.shape[1]), int(phi.shape[2])],
        "stitch_mode": "training_compatible",
        "transpose_ranks": RANKS,
        "rotate_ranks": [],
        "phi_std_mean": mean_std,
        "phi_std_max": float(np.max(phi_std)),
        "phi_rms_mean": mean_rms,
        "phi_rms_max": float(np.max(phi_rms)),
        "efield_rms_mean_raw_efx_efy": float(np.mean(efield_rms)),
        "efield_rms_max_raw_efx_efy": float(np.max(efield_rms)),
        "copy_collapse_by_nrmse_std": {
            str(th): first_collapse(copy_rows, th)
            for th in [0.25, 0.5, 1.0]
        },
        "fft_top_peaks_mhz": [row["frequency_mhz"] for row in peak_rows],
    }
    (outdir / "raw_pic_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[JSON] {outdir / 'raw_pic_metrics_summary.json'}", flush=True)
    write_readme(args.label, outdir)


if __name__ == "__main__":
    main()
