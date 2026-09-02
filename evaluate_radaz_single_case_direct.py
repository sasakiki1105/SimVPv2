#!/usr/bin/env python3
"""Evaluate native-resolution RadAz direct prediction against copy baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CASE_ROOT = (
    ROOT.parent
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
    / "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
)
DATA_H5 = (
    CASE_ROOT
    / "SimVPv2_inputs"
    / "radaz_3ch_trainfixed_margin20_native257x256_pad260x256.h5"
)
EX_NAME = (
    "radaz_xe1p_bx20mt_ez10kvm_out15ns_native257x256_"
    "direct10_trainfixed_disjoint_811_bs1_100ep"
)
SAVED = ROOT / "workdirs" / EX_NAME / "saved"
OUTPUT = ROOT / "workdirs" / "compare_radaz_bx20mt_ez10kvm_native_direct10"
CHANNELS = ("electron_den", "ion_den", "phi")
VALID_H = 257
VALID_W = 256
B_T = 0.020
MTSI = np.arange(1, 7)
ECDI = np.arange(9, 22)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def transport_metrics(
    values: np.ndarray,
    x_mask: np.ndarray,
    dy_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total, MTSI-band, and ECDI-band E-cross-B particle-flux proxies."""
    ne = np.asarray(values[:, :, 0, :VALID_H, :VALID_W], dtype=np.float64)
    phi = np.asarray(values[:, :, 2, :VALID_H, :VALID_W], dtype=np.float64)
    ey = -(np.roll(phi, -1, axis=-1) - np.roll(phi, 1, axis=-1)) / (2.0 * dy_m)
    ne = ne[:, :, x_mask]
    ey = ey[:, :, x_mask]
    dne = ne - np.mean(ne, axis=-1, keepdims=True)
    dey = ey - np.mean(ey, axis=-1, keepdims=True)
    total = -np.mean(dne * dey, axis=(-2, -1)) / B_T

    ne_fft = np.fft.rfft(dne, axis=-1, norm="forward")
    ey_fft = np.fft.rfft(dey, axis=-1, norm="forward")
    weights = np.full(ne_fft.shape[-1], 2.0)
    weights[0] = 1.0
    weights[-1] = 1.0
    contribution = (
        -weights[None, None, None, :]
        * np.real(ne_fft * np.conj(ey_fft))
        / B_T
    )
    contribution = np.mean(contribution, axis=2)
    return (
        total,
        np.sum(contribution[:, :, MTSI], axis=-1),
        np.sum(contribution[:, :, ECDI], axis=-1),
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = np.load(SAVED / "inputs.npy", mmap_mode="r")
    preds = np.load(SAVED / "preds.npy", mmap_mode="r")
    trues = np.load(SAVED / "trues.npy", mmap_mode="r")
    expected = (182, 10, 3, 260, 256)
    for name, values in (("inputs", inputs), ("preds", preds), ("trues", trues)):
        if tuple(values.shape) != expected:
            raise ValueError(f"{name} shape={values.shape}, expected={expected}")

    with h5py.File(DATA_H5, "r") as handle:
        low = np.asarray(handle["norm_low"], dtype=np.float32)
        high = np.asarray(handle["norm_high"], dtype=np.float32)
        time_s = np.asarray(handle["time_s"], dtype=np.float64)
        x_m = np.asarray(handle["x_m"], dtype=np.float64)
        y_m = np.asarray(handle["y_m"], dtype=np.float64)
        valid_shape = tuple(np.asarray(handle["valid_spatial_shape"], dtype=int))
    if valid_shape != (VALID_H, VALID_W):
        raise ValueError(f"Unexpected valid shape: {valid_shape}")
    scale = high - low

    def denormalize(values: np.ndarray) -> np.ndarray:
        return (
            np.asarray(values, dtype=np.float32)
            * scale[None, None, :, None, None]
            + low[None, None, :, None, None]
        )

    input_phys = denormalize(inputs)
    pred_phys = denormalize(preds)
    true_phys = denormalize(trues)
    copy_phys = np.repeat(input_phys[:, -1:, :, :, :], 10, axis=1)

    pred_valid = pred_phys[:, :, :, :VALID_H, :VALID_W]
    true_valid = true_phys[:, :, :, :VALID_H, :VALID_W]
    copy_valid = copy_phys[:, :, :, :VALID_H, :VALID_W]
    error_model = np.empty(pred_valid.shape[:3], dtype=np.float64)
    error_copy = np.empty(copy_valid.shape[:3], dtype=np.float64)
    for channel in range(len(CHANNELS)):
        true_channel = np.asarray(true_valid[:, :, channel], dtype=np.float64)
        model_difference = (
            np.asarray(pred_valid[:, :, channel], dtype=np.float64) - true_channel
        )
        copy_difference = (
            np.asarray(copy_valid[:, :, channel], dtype=np.float64) - true_channel
        )
        error_model[:, :, channel] = np.mean(
            model_difference * model_difference, axis=(-2, -1)
        )
        error_copy[:, :, channel] = np.mean(
            copy_difference * copy_difference, axis=(-2, -1)
        )

    phi_pred = np.asarray(pred_valid[:, :, 2], dtype=np.float64)
    phi_true = np.asarray(true_valid[:, :, 2], dtype=np.float64)
    phi_copy = np.asarray(copy_valid[:, :, 2], dtype=np.float64)
    dy_m = float(np.median(np.diff(y_m)))

    def ey(phi: np.ndarray) -> np.ndarray:
        return -(np.roll(phi, -1, axis=-1) - np.roll(phi, 1, axis=-1)) / (
            2.0 * dy_m
        )

    ey_true = ey(phi_true)
    ey_model = ey(phi_pred)
    ey_copy = ey(phi_copy)
    ey_mse_model = np.mean((ey_model - ey_true) ** 2, axis=(-2, -1))
    ey_mse_copy = np.mean((ey_copy - ey_true) ** 2, axis=(-2, -1))

    x_mask = (x_m >= 0.09e-2 - 1.0e-15) & (x_m <= 1.19e-2 + 1.0e-15)
    true_transport = transport_metrics(true_phys, x_mask, dy_m)
    model_transport = transport_metrics(pred_phys, x_mask, dy_m)
    copy_transport = transport_metrics(copy_phys, x_mask, dy_m)
    transport_names = ("total", "mtsi", "ecdi")

    horizon_ns = np.arange(1, 11, dtype=float) * 15.0
    rows = []
    for horizon in range(10):
        row = {"horizon_frame": horizon + 1, "horizon_ns": horizon_ns[horizon]}
        for channel, name in enumerate(CHANNELS):
            model_mse = float(np.mean(error_model[:, horizon, channel]))
            copy_mse = float(np.mean(error_copy[:, horizon, channel]))
            row[f"{name}_model_mse"] = model_mse
            row[f"{name}_copy_mse"] = copy_mse
            row[f"{name}_model_over_copy"] = model_mse / copy_mse
        model_ey = float(np.mean(ey_mse_model[:, horizon]))
        copy_ey = float(np.mean(ey_mse_copy[:, horizon]))
        row["ey_model_mse"] = model_ey
        row["ey_copy_mse"] = copy_ey
        row["ey_model_over_copy"] = model_ey / copy_ey
        for index, name in enumerate(transport_names):
            model_mae = float(
                np.mean(
                    np.abs(
                        model_transport[index][:, horizon]
                        - true_transport[index][:, horizon]
                    )
                )
            )
            copy_mae = float(
                np.mean(
                    np.abs(
                        copy_transport[index][:, horizon]
                        - true_transport[index][:, horizon]
                    )
                )
            )
            row[f"transport_{name}_model_mae"] = model_mae
            row[f"transport_{name}_copy_mae"] = copy_mae
            row[f"transport_{name}_model_over_copy"] = model_mae / copy_mae
        rows.append(row)
    write_rows(OUTPUT / "metrics_by_horizon.csv", rows)

    starts = np.arange(1800, 1982, dtype=int)
    target_sum_model = np.zeros(2001)
    target_sum_copy = np.zeros(2001)
    target_count = np.zeros(2001, dtype=int)
    for sample, start in enumerate(starts):
        for horizon in range(10):
            target = start + 10 + horizon
            target_sum_model[target] += error_model[sample, horizon, 2]
            target_sum_copy[target] += error_copy[sample, horizon, 2]
            target_count[target] += 1
    target_mask = target_count > 0
    target_frames = np.flatnonzero(target_mask)
    target_model = target_sum_model[target_mask] / target_count[target_mask]
    target_copy = target_sum_copy[target_mask] / target_count[target_mask]
    target_rows = [
        {
            "target_frame": int(frame),
            "target_time_us": float(time_s[frame] * 1.0e6),
            "prediction_count": int(target_count[frame]),
            "phi_model_mse": float(target_model[index]),
            "phi_copy_mse": float(target_copy[index]),
            "phi_model_over_copy": float(target_model[index] / target_copy[index]),
        }
        for index, frame in enumerate(target_frames)
    ]
    write_rows(OUTPUT / "phi_metrics_by_target_time.csv", target_rows)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    field_curves = (
        (
            "Electron density",
            np.mean(error_model[:, :, 0], axis=0),
            np.mean(error_copy[:, :, 0], axis=0),
            r"MSE [$\mathrm{m}^{-6}$]",
        ),
        (
            "Ion density",
            np.mean(error_model[:, :, 1], axis=0),
            np.mean(error_copy[:, :, 1], axis=0),
            r"MSE [$\mathrm{m}^{-6}$]",
        ),
        (
            "Potential",
            np.mean(error_model[:, :, 2], axis=0),
            np.mean(error_copy[:, :, 2], axis=0),
            r"MSE [$\mathrm{V}^{2}$]",
        ),
        (
            "Azimuthal electric field",
            np.mean(ey_mse_model, axis=0),
            np.mean(ey_mse_copy, axis=0),
            r"MSE [$\mathrm{(V/m)}^{2}$]",
        ),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for axis, (title, model_values, copy_values, ylabel) in zip(
        axes.flat, field_curves
    ):
        axis.plot(horizon_ns, model_values, marker="o", label="model")
        axis.plot(
            horizon_ns,
            copy_values,
            marker="x",
            linestyle="--",
            color="#737373",
            label="copy baseline",
        )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend(loc="lower right")
    for axis in axes[-1]:
        axis.set_xlabel("Direct prediction horizon [ns]")
    fig.suptitle("Native-resolution direct10: field prediction error")
    fig.tight_layout()
    fig.savefig(OUTPUT / "model_vs_copy_by_horizon.png", dpi=180)
    fig.savefig(OUTPUT / "field_model_vs_copy_by_horizon.png", dpi=180)
    plt.close(fig)

    transport_curves = []
    for index, name in enumerate(("Total", "MTSI band", "ECDI band")):
        transport_curves.append(
            (
                name,
                np.mean(
                    np.abs(model_transport[index] - true_transport[index]), axis=0
                ),
                np.mean(
                    np.abs(copy_transport[index] - true_transport[index]), axis=0
                ),
            )
        )

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for axis, (title, model_values, copy_values) in zip(axes, transport_curves):
        axis.plot(horizon_ns, model_values, marker="o", label="model")
        axis.plot(
            horizon_ns,
            copy_values,
            marker="x",
            linestyle="--",
            color="#737373",
            label="copy baseline",
        )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel(r"MAE [$\mathrm{m}^{-2}\,\mathrm{s}^{-1}$]")
        axis.legend(loc="lower right")
    axes[-1].set_xlabel("Direct prediction horizon [ns]")
    fig.suptitle("Density-Ey transport proxy: model vs copy baseline")
    fig.tight_layout()
    fig.savefig(OUTPUT / "transport_model_vs_copy_by_horizon.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for title, model_values, copy_values, _ in field_curves:
        axes[0].plot(
            horizon_ns,
            model_values / copy_values,
            marker="o",
            label=title,
        )
    axes[0].axhline(1.0, color="black", linestyle=":", label="equal to copy")
    axes[0].set_ylabel("Model error / copy error")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_title("Field errors (below 1 means model is better)")
    axes[0].legend(loc="lower right", ncol=2)

    for title, model_values, copy_values in transport_curves:
        axes[1].plot(
            horizon_ns,
            model_values / copy_values,
            marker="o",
            label=title,
        )
    axes[1].axhline(1.0, color="black", linestyle=":", label="equal to copy")
    axes[1].set_xlabel("Direct prediction horizon [ns]")
    axes[1].set_ylabel("Model error / copy error")
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_title("Transport-proxy errors (below 1 means model is better)")
    axes[1].legend(loc="lower right", ncol=2)
    fig.suptitle("Where the model beats or loses to copy baseline")
    fig.tight_layout()
    fig.savefig(OUTPUT / "model_over_copy_by_horizon.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        time_s[target_frames] * 1.0e6,
        target_model,
        label="model",
        color="#2b8cbe",
    )
    axis.plot(
        time_s[target_frames] * 1.0e6,
        target_copy,
        label="copy baseline",
        color="#737373",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Target physical time [us]")
    axis.set_ylabel("phi MSE [V^2]")
    axis.set_title("Unseen test interval: phi error by target time")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT / "phi_mse_by_target_time.png", dpi=180)
    plt.close(fig)

    overall = {}
    for channel, name in enumerate(CHANNELS):
        model_value = float(np.mean(error_model[:, :, channel]))
        copy_value = float(np.mean(error_copy[:, :, channel]))
        overall[name] = {
            "model_mse": model_value,
            "copy_mse": copy_value,
            "model_over_copy": model_value / copy_value,
        }
    overall_ey_model = float(np.mean(ey_mse_model))
    overall_ey_copy = float(np.mean(ey_mse_copy))
    overall["ey"] = {
        "model_mse": overall_ey_model,
        "copy_mse": overall_ey_copy,
        "model_over_copy": overall_ey_model / overall_ey_copy,
    }
    transport_summary = {}
    for index, name in enumerate(transport_names):
        model_mae = float(np.mean(np.abs(model_transport[index] - true_transport[index])))
        copy_mae = float(np.mean(np.abs(copy_transport[index] - true_transport[index])))
        transport_summary[name] = {
            "model_mae": model_mae,
            "copy_mae": copy_mae,
            "model_over_copy": model_mae / copy_mae,
            "model_true_correlation": corrcoef(
                model_transport[index].ravel(), true_transport[index].ravel()
            ),
            "copy_true_correlation": corrcoef(
                copy_transport[index].ravel(), true_transport[index].ravel()
            ),
        }

    summary = {
        "status": "PASS",
        "experiment": EX_NAME,
        "task": {
            "input_frames": 10,
            "output_frames": 10,
            "frame_interval_ns": 15.0,
            "direct_not_rollout": True,
            "train_frame_range": [0, 1599],
            "validation_frame_range": [1600, 1799],
            "test_frame_range": [1800, 2000],
            "test_time_us": [27.0, 30.0],
            "test_samples": 182,
        },
        "overall": overall,
        "transport_proxy": transport_summary,
        "padding_excluded_from_metrics": True,
        "valid_spatial_shape": [VALID_H, VALID_W],
    }
    (OUTPUT / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readme = f"""# RadAz native-resolution direct10 evaluation

## 日本語
`Bx=20 mT, Ez=10 kV/m` の単一PICケースを、完全にフレーム非重複な
train/validation/test = 8:1:1 に分割した評価です。

- train: 0-23.985 us
- validation: 24.000-26.985 us
- test: 27.000-30.000 us
- 入力: 真値10枚（150 ns）
- 出力: 次の10枚（150 ns）を一度にdirect prediction
- チャネル: electron density / ion density / phi
- 空間解像度: 有効257x256。モデル都合のradial padding 3行は評価から除外
- test sample数: 182

これはtest期間内の真値入力窓を毎回与えるteacher-forced direct評価です。
24 usから30 usまで自己出力だけで進むrolloutではありません。

## 全test窓・全horizon平均
| 指標 | model | copy | model/copy |
|---|---:|---:|---:|
| electron density MSE | {overall['electron_den']['model_mse']:.4e} | {overall['electron_den']['copy_mse']:.4e} | {overall['electron_den']['model_over_copy']:.4f} |
| ion density MSE | {overall['ion_den']['model_mse']:.4e} | {overall['ion_den']['copy_mse']:.4e} | {overall['ion_den']['model_over_copy']:.4f} |
| phi MSE | {overall['phi']['model_mse']:.4e} | {overall['phi']['copy_mse']:.4e} | {overall['phi']['model_over_copy']:.4f} |
| Ey MSE | {overall['ey']['model_mse']:.4e} | {overall['ey']['copy_mse']:.4e} | {overall['ey']['model_over_copy']:.4f} |
| total transport-proxy MAE | {transport_summary['total']['model_mae']:.4e} | {transport_summary['total']['copy_mae']:.4e} | {transport_summary['total']['model_over_copy']:.4f} |
| MTSI-band transport MAE | {transport_summary['mtsi']['model_mae']:.4e} | {transport_summary['mtsi']['copy_mae']:.4e} | {transport_summary['mtsi']['model_over_copy']:.4f} |
| ECDI-band transport MAE | {transport_summary['ecdi']['model_mae']:.4e} | {transport_summary['ecdi']['copy_mae']:.4e} | {transport_summary['ecdi']['model_over_copy']:.4f} |

`model/copy < 1`ならmodelがcopy baselineより良く、`> 1`ならcopy baselineの方が良いことを表します。

## 図
- `field_model_vs_copy_by_horizon.png`: 4物理量のhorizon別MSE
- `model_over_copy_by_horizon.png`: model/copy比。1より下ならmodel勝ち
- `transport_model_vs_copy_by_horizon.png`: 輸送proxyのhorizon別MAE
- `phi_mse_by_target_time.png`: 未学習test期間内のtarget時刻ごとのphi MSE
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[PASS] output={OUTPUT}")


if __name__ == "__main__":
    main()
