import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\astro\research\SimVPv2")
WORKDIRS = ROOT / "workdirs"
H5_ROOT = Path(
    r"C:\Users\astro\research\PEPAPIC\test\results"
    r"\2D_ExB_high_magnet_dt2.5e-9_maxt50e-6_macro5"
    r"\SimVPv2_inputs"
)
OUTDIR = WORKDIRS / "compare_stride2_direct_horizon_blocks"

BASE_DT_NS = 12.5
PRE = 10
CHANNEL_FALLBACK = ["electron_den", "ion_den", "phi"]


CASES = [
    {
        "name": "stride2_direct10",
        "label": "Direct10",
        "stride": 2,
        "aft": 10,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#9ecae1",
        "marker": "o",
    },
    {
        "name": "stride2_direct20",
        "label": "Direct20",
        "stride": 2,
        "aft": 20,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct20_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#6baed6",
        "marker": "s",
    },
    {
        "name": "stride2_direct40",
        "label": "Direct40",
        "stride": 2,
        "aft": 40,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct40_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#3182bd",
        "marker": "^",
    },
    {
        "name": "stride2_direct80",
        "label": "Direct80",
        "stride": 2,
        "aft": 80,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct80_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#08519c",
        "marker": "D",
    },
    {
        "name": "stride2_direct160",
        "label": "Direct160",
        "stride": 2,
        "aft": 160,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct160_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#08306b",
        "marker": "P",
    },
    {
        "name": "stride2_direct180",
        "label": "Direct180",
        "stride": 2,
        "aft": 180,
        "workdir": WORKDIRS / "pepapic_simvp_gsta_highmag_macro5_subsample2_direct180_trainfixed_disjoint_811_bs2_100ep",
        "h5": H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5",
        "color": "#4a1486",
        "marker": "X",
    },
]


def as_str_list(values):
    out = []
    for value in values:
        if isinstance(value, (bytes, bytearray)):
            out.append(value.decode())
        else:
            out.append(str(value))
    return out


def load_h5_info(h5_path):
    with h5py.File(h5_path, "r") as f:
        timesteps = np.asarray(f["timesteps"][()], dtype=np.int64)
        props = as_str_list(f["props"][()]) if "props" in f else CHANNEL_FALLBACK
    if len(timesteps) > 1:
        stride_steps = int(round(float(np.median(np.diff(timesteps)))))
    else:
        stride_steps = 1
    return timesteps, props, stride_steps


def build_test_starts(t_len, pre, aft):
    total = pre + aft
    test_start = int(np.floor(t_len * 0.9))
    last_start = t_len - total
    if last_start < test_start:
        return np.empty((0,), dtype=np.int64)
    return np.arange(test_start, last_start + 1, dtype=np.int64)


def block_starts(test_starts, aft):
    if len(test_starts) == 0:
        return test_starts
    first = int(test_starts[0])
    last = int(test_starts[-1])
    return np.arange(first, last + 1, int(aft), dtype=np.int64)


def load_case(case):
    saved = case["workdir"] / "saved"
    paths = {name: saved / f"{name}.npy" for name in ("inputs", "preds", "trues")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return None, f"missing saved arrays: {missing}"

    inputs = np.load(paths["inputs"], mmap_mode="r")
    preds = np.load(paths["preds"], mmap_mode="r")
    trues = np.load(paths["trues"], mmap_mode="r")
    if inputs.ndim != 5 or preds.ndim != 5 or trues.ndim != 5:
        return None, f"expected 5D arrays, got inputs={inputs.shape}, preds={preds.shape}, trues={trues.shape}"
    if int(inputs.shape[1]) != PRE:
        return None, f"unexpected input length: {inputs.shape}"
    if int(preds.shape[1]) != int(case["aft"]) or preds.shape != trues.shape:
        return None, f"unexpected output arrays: preds={preds.shape}, trues={trues.shape}"

    timesteps, props, stride_steps = load_h5_info(case["h5"])
    test_starts = build_test_starts(len(timesteps), PRE, int(case["aft"]))
    if len(test_starts) != int(preds.shape[0]):
        return None, (
            f"sample count mismatch: saved={preds.shape[0]}, "
            f"computed_test_starts={len(test_starts)}"
        )

    selected_starts = block_starts(test_starts, int(case["aft"]))
    selected_indices = selected_starts - test_starts[0]
    return {
        **case,
        "saved": saved,
        "inputs": inputs,
        "preds": preds,
        "trues": trues,
        "timesteps": timesteps,
        "props": props,
        "stride_steps": stride_steps,
        "dt_ns": stride_steps * BASE_DT_NS,
        "test_starts": test_starts,
        "block_starts": selected_starts,
        "block_indices": selected_indices,
    }, None


def case_rows(case):
    rows = []
    starts = case["block_starts"]
    indices = case["block_indices"]
    inputs = case["inputs"][indices]
    preds = case["preds"][indices]
    trues = case["trues"][indices]
    timesteps = case["timesteps"]

    last_input_timestep = timesteps[starts + PRE - 1]
    for block_id, start in enumerate(starts):
        output_first = int(start + PRE)
        output_last = int(start + PRE + case["aft"] - 1)
        for channel_index, channel in enumerate(case["props"]):
            persistence = inputs[block_id, -1, channel_index].astype(np.float64)
            for tp in range(case["aft"]):
                pred = preds[block_id, tp, channel_index].astype(np.float64)
                true = trues[block_id, tp, channel_index].astype(np.float64)
                copy = persistence
                model_mse = float(np.mean((pred - true) ** 2))
                copy_mse = float(np.mean((copy - true) ** 2))
                target_index = int(start + PRE + tp)
                target_timestep = int(timesteps[target_index])
                horizon_ns = float((target_timestep - last_input_timestep[block_id]) * BASE_DT_NS)
                target_time_us = float(target_timestep * BASE_DT_NS / 1000.0)
                rows.append({
                    "case": case["name"],
                    "label": case["label"],
                    "stride": case["stride"],
                    "aft": case["aft"],
                    "dt_ns": case["dt_ns"],
                    "block_id": int(block_id),
                    "input_start_index": int(start),
                    "input_end_index": int(start + PRE - 1),
                    "output_first_index": output_first,
                    "output_last_index": output_last,
                    "output_index": int(tp + 1),
                    "channel": channel,
                    "channel_index": int(channel_index),
                    "target_index": target_index,
                    "target_timestep": target_timestep,
                    "target_time_us": target_time_us,
                    "horizon_ns": horizon_ns,
                    "model_mse": model_mse,
                    "copy_mse": copy_mse,
                    "model_over_copy": model_mse / copy_mse if copy_mse > 0 else np.nan,
                })
    return rows


def write_csv(rows):
    path = OUTDIR / "stride2_direct_block_predictions.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {path}")


def plot_phi_target_time(rows, loaded_cases, yscale="log", suffix=""):
    plt.figure(figsize=(10, 5.8))
    for case in loaded_cases:
        r = [row for row in rows if row["case"] == case["name"] and row["channel"] == "phi"]
        if not r:
            continue
        x = np.asarray([row["target_time_us"] for row in r], dtype=np.float64)
        y = np.asarray([row["model_mse"] for row in r], dtype=np.float64)
        order = np.argsort(x)
        plt.plot(
            x[order],
            y[order],
            color=case["color"],
            linewidth=1.4,
            alpha=0.9,
            label=case["label"],
        )
    plt.xlabel("Target simulation time (us)")
    plt.ylabel("MSE (phi, normalized)")
    scale_label = "log scale" if yscale == "log" else "linear scale"
    plt.title(f"Stride2 true direct block prediction: phi MSE over test time ({scale_label})")
    plt.yscale(yscale)
    grid_which = "both" if yscale == "log" else "major"
    plt.grid(True, which=grid_which, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_png = OUTDIR / f"stride2_direct_block_phi_mse_target_time{suffix}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_phi_block_mean(rows, loaded_cases, yscale="log", suffix=""):
    plt.figure(figsize=(10, 5.8))
    for case in loaded_cases:
        r = [row for row in rows if row["case"] == case["name"] and row["channel"] == "phi"]
        if not r:
            continue
        block_ids = sorted({row["block_id"] for row in r})
        x = []
        y = []
        for block_id in block_ids:
            b = [row for row in r if row["block_id"] == block_id]
            x.append(np.mean([row["target_time_us"] for row in b]))
            y.append(np.mean([row["model_mse"] for row in b]))
        plt.plot(
            x,
            y,
            color=case["color"],
            marker=case["marker"],
            linewidth=1.7,
            markersize=4,
            label=case["label"],
        )
    plt.xlabel("Mean target simulation time in block (us)")
    plt.ylabel("Block-mean MSE (phi, normalized)")
    scale_label = "log scale" if yscale == "log" else "linear scale"
    plt.title(f"Stride2 true direct block prediction: block-mean phi MSE ({scale_label})")
    plt.yscale(yscale)
    grid_which = "both" if yscale == "log" else "major"
    plt.grid(True, which=grid_which, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_png = OUTDIR / f"stride2_direct_block_phi_mse_block_mean{suffix}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def plot_phi_horizon_mean(rows, loaded_cases, yscale="log", suffix=""):
    plt.figure(figsize=(10, 5.8))
    for case in loaded_cases:
        r = [row for row in rows if row["case"] == case["name"] and row["channel"] == "phi"]
        if not r:
            continue
        output_indices = sorted({row["output_index"] for row in r})
        x = []
        y = []
        q25 = []
        q75 = []
        for output_index in output_indices:
            o = [row for row in r if row["output_index"] == output_index]
            values = np.asarray([row["model_mse"] for row in o], dtype=np.float64)
            x.append(np.median([row["horizon_ns"] for row in o]))
            y.append(np.mean(values))
            q25.append(np.quantile(values, 0.25))
            q75.append(np.quantile(values, 0.75))
        x = np.asarray(x)
        y = np.asarray(y)
        q25 = np.asarray(q25)
        q75 = np.asarray(q75)
        plt.plot(
            x,
            y,
            color=case["color"],
            marker=case["marker"],
            linewidth=1.7,
            markersize=4,
            label=case["label"],
        )
        plt.fill_between(x, q25, q75, color=case["color"], alpha=0.14, linewidth=0)
    plt.xlabel("Prediction horizon from last input frame (ns)")
    plt.ylabel("Mean MSE (phi, normalized)")
    scale_label = "log scale" if yscale == "log" else "linear scale"
    plt.title(f"Stride2 true direct block prediction: phi MSE by horizon ({scale_label})")
    plt.yscale(yscale)
    grid_which = "both" if yscale == "log" else "major"
    plt.grid(True, which=grid_which, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_png = OUTDIR / f"stride2_direct_block_phi_mse_horizon_mean{suffix}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[PLOT] {out_png}")


def summarize(rows, loaded_cases, skipped):
    summary_cases = []
    for case in loaded_cases:
        phi_rows = [row for row in rows if row["case"] == case["name"] and row["channel"] == "phi"]
        if not phi_rows:
            continue
        mse = np.asarray([row["model_mse"] for row in phi_rows], dtype=np.float64)
        ratio = np.asarray([row["model_over_copy"] for row in phi_rows], dtype=np.float64)
        summary_cases.append({
            "name": case["name"],
            "label": case["label"],
            "aft": case["aft"],
            "dt_ns": case["dt_ns"],
            "n_blocks": int(len(case["block_starts"])),
            "n_phi_predictions": int(len(phi_rows)),
            "phi_mse_mean": float(np.mean(mse)),
            "phi_mse_median": float(np.median(mse)),
            "phi_mse_q25": float(np.quantile(mse, 0.25)),
            "phi_mse_q75": float(np.quantile(mse, 0.75)),
            "phi_model_over_copy_mean": float(np.nanmean(ratio)),
            "phi_model_over_copy_median": float(np.nanmedian(ratio)),
            "preds_shape": list(case["preds"].shape),
            "saved": str(case["saved"]),
        })

    summary = {
        "description": (
            "Stride2 true direct block prediction evaluation. For output length L, "
            "test starts advance by L frames. Target blocks are non-overlapping. "
            "The next block input uses true PIC frames immediately before the next "
            "target block, so those input frames overlap the previous target block "
            "but no predicted frames are fed back."
        ),
        "base_dt_ns": BASE_DT_NS,
        "pre_seq_length": PRE,
        "h5": str(H5_ROOT / "global_norm_trainfixed_minmax_margin20_t0_to_t4000_step2.h5"),
        "cases": summary_cases,
        "skipped_cases": skipped,
    }
    path = OUTDIR / "stride2_direct_block_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] {path}")


def write_readme():
    text = """# Experiment

Stride2 true direct long-horizon block evaluation.

# Meaning

For an output length `L`, this evaluates the model over the test segment using non-overlapping target blocks:

```text
input true frames [s : s+10) -> predict target frames [s+10 : s+10+L)
next start = s + L
```

The target blocks do not overlap. The next input window can overlap with the previous target block, but it uses true PIC frames, not predicted frames. Therefore this is teacher-forced direct prediction, not rollout.

# Key Files

- `stride2_direct_block_predictions.csv`: per-target-frame block metrics.
- `stride2_direct_block_summary.json`: metadata and aggregate `phi` summaries.
- `stride2_direct_block_phi_mse_target_time.png`: `phi` MSE over test simulation time.
- `stride2_direct_block_phi_mse_target_time_linear.png`: same plot with a linear y-axis.
- `stride2_direct_block_phi_mse_block_mean.png`: block-mean `phi` MSE.
- `stride2_direct_block_phi_mse_block_mean_linear.png`: same plot with a linear y-axis.
- `stride2_direct_block_phi_mse_horizon_mean.png`: mean `phi` MSE by horizon inside each direct prediction.
- `stride2_direct_block_phi_mse_horizon_mean_linear.png`: same plot with a linear y-axis.

# Caution

This folder is updated by `plot_stride2_direct_horizon_blocks.py`. Missing direct models are skipped, so check `stride2_direct_block_summary.json` to see which cases were included.

# Japanese Memo

このフォルダは、stride2 の true direct long-horizon モデルを、test 区間の最後までブロックごとに評価した結果です。

出力長を `L` とすると、評価は次のようになります。

```text
真値入力 [s : s+10) -> 一回の forward で真値ターゲット [s+10 : s+10+L) を予測
次の開始位置 = s + L
```

ターゲットブロック同士は重なりません。ただし、次の入力10枚は前のターゲット区間と一部重なることがあります。ここで使う入力は予測値ではなく真値 PIC フレームなので、この評価は rollout ではなく teacher-forced direct prediction です。

`*_linear.png` は通常目盛り、`_linear` がないものは対数目盛りです。
"""
    path = OUTDIR / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}")
    return

    text = """# Experiment

Stride2 true direct long-horizon block evaluation.

# Meaning

For an output length `L`, this evaluates the model over the test segment using non-overlapping target blocks:

```text
input true frames [s : s+10) -> predict target frames [s+10 : s+10+L)
next start = s + L
```

The target blocks do not overlap. The next input window does overlap with the previous target block, but it uses true PIC frames, not predicted frames. Therefore this is teacher-forced direct prediction, not rollout.

# Key Files

- `stride2_direct_block_predictions.csv`: per-target-frame block metrics.
- `stride2_direct_block_summary.json`: metadata and aggregate `phi` summaries.
- `stride2_direct_block_phi_mse_target_time.png`: `phi` MSE over test simulation time.
- `stride2_direct_block_phi_mse_block_mean.png`: block-mean `phi` MSE.
- `stride2_direct_block_phi_mse_horizon_mean.png`: mean `phi` MSE by horizon inside each direct prediction.

# Caution

This folder is valid only after `stride2_direct20`, `stride2_direct40`, and `stride2_direct80` have completed. Missing cases are skipped.

# 日本語訳

このフォルダは、stride2 の真の direct long-horizon 予測を、test 区間の最後までブロックごとに評価するものです。

出力長を `L` とすると、開始位置 `s` で真値10枚を入力し、次の `L` 枚を一度の forward で予測します。次の開始位置は `s + L` です。したがって予測ターゲット区間は重複しません。

ただし、次の入力10枚は前の予測ターゲット区間と重なります。これは予測値を戻しているのではなく、真値 PIC フレームを入力しているためです。そのため、この評価は rollout ではなく teacher-forced direct prediction です。
"""
    path = OUTDIR / "README.md"
    path.write_text(text, encoding="utf-8")
    print(f"[README] {path}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    loaded_cases = []
    skipped = []
    rows = []

    for case in CASES:
        loaded, reason = load_case(case)
        if loaded is None:
            skipped.append({"name": case["name"], "label": case["label"], "reason": reason})
            print(f"[SKIP] {case['name']}: {reason}")
            continue
        loaded_cases.append(loaded)
        rows.extend(case_rows(loaded))
        print(
            f"[LOAD] {case['name']}: preds={loaded['preds'].shape}, "
            f"blocks={len(loaded['block_starts'])}"
        )

    if not rows:
        raise RuntimeError("No cases were loaded.")

    write_csv(rows)
    plot_phi_target_time(rows, loaded_cases)
    plot_phi_target_time(rows, loaded_cases, yscale="linear", suffix="_linear")
    plot_phi_block_mean(rows, loaded_cases)
    plot_phi_block_mean(rows, loaded_cases, yscale="linear", suffix="_linear")
    plot_phi_horizon_mean(rows, loaded_cases)
    plot_phi_horizon_mean(rows, loaded_cases, yscale="linear", suffix="_linear")
    summarize(rows, loaded_cases, skipped)
    write_readme()


if __name__ == "__main__":
    main()
