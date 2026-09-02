import csv
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "workdirs" / "compare_phi_time_evolution_testcases"
PHI_INDEX = 2
SNAPSHOT_FRACTIONS = np.linspace(0.0, 1.0, 6)

CASES = [
    {
        "slug": "low_magnet_3a",
        "label": "Test case 3a: low magnet",
        "short_label": "3a low B",
        "b_tesla": 2.0e-5,
        "raw_dt_ns": 12.5,
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_low_magnet_dt2.5e-9_maxt50e-6_macro5"
            r"\SimVPv2_inputs"
            r"\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_training_compatible.h5"
        ),
    },
    {
        "slug": "high_magnet_3b",
        "label": "Test case 3b: high magnet",
        "short_label": "3b high B",
        "b_tesla": 2.0e-4,
        "raw_dt_ns": 12.5,
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
            r"\SimVPv2_inputs"
            r"\global_norm_trainfixed_minmax_margin20_t0_to_t4000.h5"
        ),
    },
    {
        "slug": "exhigh_magnet_5us",
        "label": "Ex-high magnet: 5 us, fine PIC timestep",
        "short_label": "10x B, 5 us",
        "b_tesla": 2.0e-3,
        "raw_dt_ns": 1.25,
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_exhigh_magnet_dt2.5e-10_maxt5e-6_macro5"
            r"\SimVPv2_inputs"
            r"\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step20_training_compatible.h5"
        ),
    },
    {
        "slug": "exhigh_magnet_50us",
        "label": "Ex-high magnet: 50 us, original PIC timestep",
        "short_label": "10x B, 50 us",
        "b_tesla": 2.0e-3,
        "raw_dt_ns": 12.5,
        "h5": Path(
            r"C:\Users\astro\research\PEPAPIC\test\results"
            r"\2D_ExB_exhigh_magnet_dt2.5e-9_maxt50e-6_macro5"
            r"\SimVPv2_inputs"
            r"\global_norm_from_high3b_trainfixed_minmax_margin20_t0_to_t4000_step2_training_compatible.h5"
        ),
    },
]

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def decode_props(values):
    return [
        value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
        for value in values
    ]


def denormalize_phi(normalized, train_min, train_max, margin):
    span = train_max - train_min
    low = train_min - margin * span
    high = train_max + margin * span
    return normalized * (high - low) + low


def detect_layout(dataset, timestep_count):
    if (
        dataset.ndim == 4
        and dataset.shape[0] == timestep_count
        and dataset.shape[1] <= 20
    ):
        return "tchw"
    if (
        dataset.ndim == 4
        and dataset.shape[3] == timestep_count
        and dataset.shape[2] <= 20
    ):
        return "xyct"
    raise ValueError(
        f"Cannot infer H5 layout from data shape {dataset.shape} "
        f"and {timestep_count} timesteps"
    )


def read_phi_frames(dataset, layout, phi_index, start, stop):
    if layout == "tchw":
        return np.asarray(dataset[start:stop, phi_index], dtype=np.float64)
    return np.transpose(
        np.asarray(dataset[:, :, phi_index, start:stop], dtype=np.float64),
        (2, 0, 1),
    )


def load_case(case):
    with h5py.File(case["h5"], "r") as handle:
        props = decode_props(handle["props"][()])
        phi_index = props.index("phi") if "phi" in props else PHI_INDEX
        timesteps = np.asarray(handle["timesteps"][()], dtype=np.int64)
        train_min = float(handle["train_min"][phi_index])
        train_max = float(handle["train_max"][phi_index])
        margin = float(handle["margin"][()])
        dataset = handle["data_tchw"]
        frame_count = len(timesteps)
        layout = detect_layout(dataset, frame_count)
        snapshot_indices = np.rint(SNAPSHOT_FRACTIONS * (frame_count - 1)).astype(int)
        snapshots = []
        for index in snapshot_indices:
            normalized = read_phi_frames(
                dataset,
                layout,
                phi_index,
                index,
                index + 1,
            )[0]
            snapshots.append(
                denormalize_phi(normalized, train_min, train_max, margin)
            )

        means = np.empty(frame_count, dtype=np.float64)
        stds = np.empty(frame_count, dtype=np.float64)
        rms = np.empty(frame_count, dtype=np.float64)
        minima = np.empty(frame_count, dtype=np.float64)
        maxima = np.empty(frame_count, dtype=np.float64)
        for start in range(0, frame_count, 64):
            stop = min(start + 64, frame_count)
            normalized = read_phi_frames(
                dataset,
                layout,
                phi_index,
                start,
                stop,
            )
            phi = denormalize_phi(normalized, train_min, train_max, margin)
            means[start:stop] = np.mean(phi, axis=(1, 2))
            stds[start:stop] = np.std(phi, axis=(1, 2))
            rms[start:stop] = np.sqrt(np.mean(phi * phi, axis=(1, 2)))
            minima[start:stop] = np.min(phi, axis=(1, 2))
            maxima[start:stop] = np.max(phi, axis=(1, 2))

    time_us = timesteps.astype(np.float64) * case["raw_dt_ns"] / 1000.0
    duration_us = float(time_us[-1] - time_us[0])
    retained_dt_ns = float(np.median(np.diff(timesteps))) * case["raw_dt_ns"]
    progress = (time_us - time_us[0]) / duration_us if duration_us else np.zeros_like(time_us)

    return {
        **case,
        "timesteps": timesteps,
        "time_us": time_us,
        "progress": progress,
        "duration_us": duration_us,
        "retained_dt_ns": retained_dt_ns,
        "snapshot_indices": snapshot_indices,
        "snapshot_times_us": time_us[snapshot_indices],
        "snapshots": snapshots,
        "mean": means,
        "std": stds,
        "rms": rms,
        "min": minima,
        "max": maxima,
    }


def common_norm(loaded_cases):
    values = np.concatenate(
        [snapshot.ravel() for case in loaded_cases for snapshot in case["snapshots"]]
    )
    vmin, vmax = np.percentile(values, [1.0, 99.0])
    if vmin < 0.0 < vmax:
        return TwoSlopeNorm(vmin=float(vmin), vcenter=0.0, vmax=float(vmax))
    return matplotlib.colors.Normalize(vmin=float(vmin), vmax=float(vmax))


def draw_snapshot(ax, frame, norm, title=None):
    image = ax.imshow(
        np.rot90(frame, k=-1),
        origin="lower",
        cmap="coolwarm",
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return image


def plot_case(case, norm):
    fig = plt.figure(figsize=(16, 7.8))
    grid = fig.add_gridspec(
        3,
        6,
        height_ratios=[1.0, 0.58, 0.58],
        hspace=0.42,
        wspace=0.12,
    )
    image = None
    for column, (frame, time_us) in enumerate(
        zip(case["snapshots"], case["snapshot_times_us"])
    ):
        ax = fig.add_subplot(grid[0, column])
        image = draw_snapshot(ax, frame, norm, f"t = {time_us:g} us")

    mean_ax = fig.add_subplot(grid[1, :])
    mean_ax.fill_between(
        case["time_us"],
        case["min"],
        case["max"],
        color="#9ecae1",
        alpha=0.25,
        linewidth=0,
        label="spatial min-max",
    )
    mean_ax.plot(
        case["time_us"],
        case["mean"],
        color="#1f77b4",
        linewidth=1.2,
        label="spatial mean",
    )
    mean_ax.set_ylabel("phi (V)")
    mean_ax.grid(True, linestyle=":", alpha=0.55)
    mean_ax.legend(loc="lower right", ncol=2, frameon=True)

    std_ax = fig.add_subplot(grid[2, :], sharex=mean_ax)
    std_ax.plot(
        case["time_us"],
        case["std"],
        color="#d62728",
        linewidth=1.2,
        label="spatial std",
    )
    std_ax.plot(
        case["time_us"],
        case["rms"],
        color="#2ca02c",
        linewidth=1.0,
        label="spatial RMS",
    )
    std_ax.set_xlabel("Simulation time (us)")
    std_ax.set_ylabel("Amplitude (V)")
    std_ax.grid(True, linestyle=":", alpha=0.55)
    std_ax.legend(loc="lower right", ncol=2, frameon=True)

    fig.suptitle(
        f"{case['label']} | Bz={case['b_tesla']:.1e} T | "
        f"duration={case['duration_us']:g} us | retained frame interval={case['retained_dt_ns']:g} ns",
        fontsize=14,
    )
    colorbar = fig.colorbar(
        image,
        ax=[fig.axes[index] for index in range(6)],
        location="right",
        shrink=0.78,
        pad=0.015,
    )
    colorbar.set_label("phi (V)")
    path = OUTDIR / f"phi_time_evolution_{case['slug']}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_snapshot_comparison(loaded_cases, norm):
    fig, axes = plt.subplots(
        len(loaded_cases),
        len(SNAPSHOT_FRACTIONS),
        figsize=(16, 10.5),
        constrained_layout=True,
    )
    image = None
    for row, case in enumerate(loaded_cases):
        for column, (frame, time_us) in enumerate(
            zip(case["snapshots"], case["snapshot_times_us"])
        ):
            title = f"{SNAPSHOT_FRACTIONS[column] * 100:.0f}%\n{time_us:g} us"
            image = draw_snapshot(axes[row, column], frame, norm, title)
            if column == 0:
                axes[row, column].set_ylabel(
                    f"{case['short_label']}\nB={case['b_tesla']:.1e} T",
                    fontsize=10,
                )
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.82)
    colorbar.set_label("phi (V)")
    fig.suptitle(
        "PIC phi evolution at equal fractions of each simulation",
        fontsize=15,
    )
    path = OUTDIR / "phi_snapshots_all_cases_common_scale.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_statistics_comparison(loaded_cases):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for case, color in zip(loaded_cases, COLORS):
        axes[0].plot(
            case["progress"],
            case["mean"],
            color=color,
            linewidth=1.2,
            label=case["short_label"],
        )
        axes[1].plot(
            case["progress"],
            case["std"],
            color=color,
            linewidth=1.2,
            label=case["short_label"],
        )
    axes[0].set_title("Spatial mean of phi")
    axes[1].set_title("Spatial standard deviation of phi")
    for ax in axes:
        ax.set_xlabel("Normalized simulation progress")
        ax.set_ylabel("phi (V)")
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(loc="lower right", frameon=True)
    path = OUTDIR / "phi_time_statistics_all_cases.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[PLOT] {path}")


def write_statistics_csv(loaded_cases):
    path = OUTDIR / "phi_time_statistics_all_cases.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case",
            "b_tesla",
            "timestep",
            "time_us",
            "normalized_progress",
            "phi_spatial_mean_v",
            "phi_spatial_std_v",
            "phi_spatial_rms_v",
            "phi_spatial_min_v",
            "phi_spatial_max_v",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in loaded_cases:
            for index in range(len(case["time_us"])):
                writer.writerow(
                    {
                        "case": case["slug"],
                        "b_tesla": case["b_tesla"],
                        "timestep": int(case["timesteps"][index]),
                        "time_us": case["time_us"][index],
                        "normalized_progress": case["progress"][index],
                        "phi_spatial_mean_v": case["mean"][index],
                        "phi_spatial_std_v": case["std"][index],
                        "phi_spatial_rms_v": case["rms"][index],
                        "phi_spatial_min_v": case["min"][index],
                        "phi_spatial_max_v": case["max"][index],
                    }
                )
    print(f"[CSV] {path}")


def write_readme(loaded_cases):
    case_lines = "\n".join(
        f"- `{case['slug']}`: Bz={case['b_tesla']:.1e} T, "
        f"duration={case['duration_us']:g} us, retained interval={case['retained_dt_ns']:g} ns"
        for case in loaded_cases
    )
    text = f"""# phiの時間変化

各PIC test caseの真値`phi`について、代表時刻の空間分布と時間統計を可視化した。

対象:

{case_lines}

## 図

- `phi_snapshots_all_cases_common_scale.png`: 各計算時間の0, 20, 40, 60, 80, 100%地点を、全ケース共通の物理カラースケールで比較する。
- `phi_time_statistics_all_cases.png`: 各ケースの空間平均と空間標準偏差を、計算進行率0--1で重ねる。
- `phi_time_evolution_<case>.png`: ケースごとの代表スナップショット、空間平均、min-max、標準偏差、RMS。
- `phi_time_statistics_all_cases.csv`: 各フレームの時間統計。

`phi`はH5の正規化値から、各H5に保存された`train_min`、`train_max`、`margin`を用いて物理値へ戻した。追加ケースには正式な`training_compatible`貼り合わせH5を使用している。

5 us強磁場ケースのH5は生PICの20フレームごと、50 us強磁場ケースは2フレームごとのデータである。このため、個別図の時間統計は各H5に保持された時間解像度で計算している。
"""
    path = OUTDIR / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    loaded_cases = []
    for case in CASES:
        print(f"[LOAD] {case['label']}: {case['h5']}")
        loaded_cases.append(load_case(case))
    norm = common_norm(loaded_cases)
    for case in loaded_cases:
        plot_case(case, norm)
    plot_snapshot_comparison(loaded_cases, norm)
    plot_statistics_comparison(loaded_cases)
    write_statistics_csv(loaded_cases)
    write_readme(loaded_cases)


if __name__ == "__main__":
    main()
