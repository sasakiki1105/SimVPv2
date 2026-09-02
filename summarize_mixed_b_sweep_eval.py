import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WORKDIRS = ROOT / "workdirs"
CHANNELS = ["electron_den", "ion_den", "phi"]


DEFAULT_MANIFEST = (
    WORKDIRS
    / "mixed_b_sweep_manifests"
    / "b_sweep_stride2_all_cases_manifest.json"
)
DEFAULT_EX_NAME = (
    "pepapic_simvp_gsta_mixed_b_sweep_stride2_direct10_"
    "all_cases_data_only_trainfixed_disjoint_811_bs2_100ep"
)
DEFAULT_OUTDIR = WORKDIRS / "compare_mixed_b_sweep_stride2_all_cases_data_only"


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def h5_len(path):
    with h5py.File(path, "r") as f:
        if "timesteps" in f:
            return int(len(f["timesteps"][()]))
        if "data_tchw" in f:
            data = f["data_tchw"]
        else:
            data = f["data"]
        if len(data.shape) != 4:
            raise ValueError(f"Expected 4D data in {path}, got {data.shape}")
        if data.shape[1] <= 16:
            return int(data.shape[0])      # (T,C,H,W)
        if data.shape[2] <= 16:
            return int(data.shape[3])      # (H,W,C,T)
        if data.shape[3] <= 16:
            return int(data.shape[0])      # (T,H,W,C)
        raise ValueError(f"Cannot infer time axis for {path}: {data.shape}")


def disjoint_starts(T, total, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    train_end = int(np.floor(T * train_ratio))
    val_end = int(np.floor(T * (train_ratio + val_ratio)))

    def starts_in_range(frame_start, frame_end):
        last_start = frame_end - total
        if last_start < frame_start:
            return np.empty((0,), dtype=np.int32)
        return np.arange(frame_start, last_start + 1, dtype=np.int32)

    return {
        "train": starts_in_range(0, train_end),
        "val": starts_in_range(train_end, val_end),
        "test": starts_in_range(val_end, T),
    }


def expected_test_samples(manifest, pre, aft):
    samples = []
    split_cfg = manifest.get("split", {})
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    for case in manifest["cases"]:
        splits = case.get("splits", ["train", "val", "test"])
        if isinstance(splits, str):
            splits = [splits]
        splits = set(str(s).lower() for s in splits)
        if "all" not in splits and "test" not in splits:
            continue
        T = h5_len(case["path"])
        starts = disjoint_starts(
            T, pre + aft,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )["test"]
        for st in starts:
            samples.append({
                "case_key": case["case_key"],
                "label": case.get("label", case["case_key"]),
                "B_mT": float(case.get("B_mT", np.nan)),
                "start": int(st),
            })
    return samples


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_legend(ax, ncol=1):
    ax.legend(loc="upper right", frameon=True, fontsize=8, ncol=ncol)


def plot_phi_by_case(rows, outdir):
    labels = sorted({row["label"] for row in rows}, key=lambda x: float(next(r["B_mT"] for r in rows if r["label"] == x)))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for label in labels:
        sub = [r for r in rows if r["label"] == label and r["channel"] == "phi"]
        sub = sorted(sub, key=lambda r: int(r["horizon"]))
        if not sub:
            continue
        horizons = [int(r["horizon"]) for r in sub]
        model = [float(r["model_mse_mean"]) for r in sub]
        copy = [float(r["copy_mse_mean"]) for r in sub]
        ax.plot(horizons, model, marker="o", linewidth=1.8, label=f"{label} model")
        ax.plot(horizons, copy, linestyle="--", linewidth=1.2, alpha=0.7, label=f"{label} copy")
    ax.set_xlabel("Prediction horizon frame")
    ax.set_ylabel("Normalized phi MSE")
    ax.set_title("Mixed B-sweep training: test phi MSE by case")
    ax.grid(True, alpha=0.25)
    add_legend(ax, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "mixed_test_phi_mse_by_case.png", dpi=200)
    plt.close(fig)


def plot_phi_ratio(rows, outdir):
    labels = sorted({row["label"] for row in rows}, key=lambda x: float(next(r["B_mT"] for r in rows if r["label"] == x)))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for label in labels:
        sub = [r for r in rows if r["label"] == label and r["channel"] == "phi"]
        sub = sorted(sub, key=lambda r: int(r["horizon"]))
        if not sub:
            continue
        horizons = [int(r["horizon"]) for r in sub]
        ratio = [float(r["model_over_copy"]) for r in sub]
        ax.plot(horizons, ratio, marker="o", linewidth=1.8, label=label)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("Prediction horizon frame")
    ax.set_ylabel("Model / copy phi MSE")
    ax.set_title("Mixed B-sweep training: phi model-over-copy ratio")
    ax.grid(True, alpha=0.25)
    add_legend(ax)
    fig.tight_layout()
    fig.savefig(outdir / "mixed_test_phi_model_over_copy_by_case.png", dpi=200)
    plt.close(fig)


def write_readme(outdir, manifest_path, ex_name, rows):
    phi_rows = [r for r in rows if r["channel"] == "phi"]
    case_summary = []
    for label in sorted({r["label"] for r in phi_rows}, key=lambda x: float(next(r["B_mT"] for r in phi_rows if r["label"] == x))):
        sub = [r for r in phi_rows if r["label"] == label]
        mean_ratio = np.mean([float(r["model_over_copy"]) for r in sub])
        case_summary.append(f"- {label}: mean phi model/copy = {mean_ratio:.3f}")
    text = f"""# Mixed B-sweep Training Evaluation

This folder summarizes test predictions from a label-free mixed-case SimVPv2/gSTA model.

## Source

- manifest: `{manifest_path}`
- workdir: `{ex_name}`
- split policy: each testcase is split by frame ranges first, then samples are mixed.
- model input/output: 10 frames input, 10 frames direct output.

## Files

- `mixed_test_channel_mse_by_horizon.csv`: per-case, per-channel, per-horizon model and copy MSE.
- `mixed_test_phi_mse_by_case.png`: normalized phi MSE for model and copy baseline.
- `mixed_test_phi_model_over_copy_by_case.png`: phi MSE ratio. Values below 1 mean the model beats copy.

## Phi Summary

{chr(10).join(case_summary)}
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--ex-name", default=DEFAULT_EX_NAME)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--pre", type=int, default=10)
    parser.add_argument("--aft", type=int, default=10)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    outdir = Path(args.outdir)
    saved = WORKDIRS / args.ex_name / "saved"
    inputs = np.load(saved / "inputs.npy", mmap_mode="r")
    preds = np.load(saved / "preds.npy", mmap_mode="r")
    trues = np.load(saved / "trues.npy", mmap_mode="r")
    samples = expected_test_samples(load_manifest(manifest_path), args.pre, args.aft)
    if len(samples) != preds.shape[0]:
        raise ValueError(
            f"Saved prediction count ({preds.shape[0]}) does not match manifest test samples ({len(samples)})."
        )

    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    sample_cases = np.array([s["case_key"] for s in samples])
    label_by_case = {s["case_key"]: s["label"] for s in samples}
    b_by_case = {s["case_key"]: s["B_mT"] for s in samples}
    for case_key in sorted(label_by_case, key=lambda key: b_by_case[key]):
        idx = np.where(sample_cases == case_key)[0]
        for ch, channel in enumerate(CHANNELS):
            for h in range(args.aft):
                model_err = (preds[idx, h, ch] - trues[idx, h, ch]) ** 2
                copy_err = (inputs[idx, -1, ch] - trues[idx, h, ch]) ** 2
                model_mse = float(np.mean(model_err))
                copy_mse = float(np.mean(copy_err))
                rows.append({
                    "case_key": case_key,
                    "label": label_by_case[case_key],
                    "B_mT": b_by_case[case_key],
                    "channel": channel,
                    "horizon": h + 1,
                    "model_mse_mean": model_mse,
                    "copy_mse_mean": copy_mse,
                    "model_over_copy": model_mse / copy_mse if copy_mse > 0 else np.nan,
                    "n_samples": int(len(idx)),
                })

    write_csv(rows, outdir / "mixed_test_channel_mse_by_horizon.csv")
    plot_phi_by_case(rows, outdir)
    plot_phi_ratio(rows, outdir)
    write_readme(outdir, manifest_path, args.ex_name, rows)
    print(outdir)


if __name__ == "__main__":
    main()
