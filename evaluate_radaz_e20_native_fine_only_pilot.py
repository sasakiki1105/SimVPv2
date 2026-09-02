#!/usr/bin/env python3
"""Run the existing-data fine-only/native-G2 pilot for E20/B20.

The frozen SimVP checkpoint was trained only on fine PIC trajectories
(E10+E20); no artificial or native coarse data entered its training.  The
script runs three linked experiments:

A. free fine-dynamics rollout from fine and native-G2 histories;
B. fine/artificial-G2/native-G2 encoder-latent overlap diagnostics;
C. non-parametric local lifting onto the fine training manifold, with
   hyperparameters selected only on a synthetic native-node validation set.

Native G2 and fine PIC are independent particle realizations.  Accordingly,
the native test is evaluated using statistical/physical gates rather than
frame-wise fine MSE.  Frame-wise MSE is used only for the paired artificial
validation control and the in-domain fine forecasting diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

import evaluate_radaz_native_g2_reconstruction as native_eval
from openstl.models.simvp_model import SimVP_Model
from train_radaz_g2_residual_superresolution import make_g2_interpolated


ROOT = Path(__file__).resolve().parent
WORKDIR = (
    ROOT
    / "workdirs"
    / "2D_RadAz"
    / "radaz_bx20mt_lowE_E10_E20_mixed_direct10_sourcepool_noclip_disjoint811_bs1_100ep"
)
MANIFEST = (
    ROOT
    / "workdirs"
    / "2D_RadAz"
    / "radaz_regime_generalization_manifests"
    / "radaz_low_source_manifest.json"
)
CONFIG = ROOT / "configs" / "custom" / "pepapic" / "SimVP_gSTA_radaz_direct.py"
FINE_CASE = "2D_RadAz_Xe1p_Bx20mT_Ez20kVm_dt15ps_out15ns"
FINE_H5 = (
    ROOT.parent
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / FINE_CASE
    / FINE_CASE
    / "analysis_fields_uncompressed.h5"
)
NATIVE_CASE = "2D_RadAz_Xe1p_Bx20mT_Ez20kVm_G2coarse_0to30us_dt15ps_out15ns"
NATIVE_H5 = (
    ROOT.parent
    / "research_results"
    / "2D_RadAz"
    / "PEPAPIC"
    / "2D_Landmark"
    / NATIVE_CASE
    / NATIVE_CASE
    / "analysis_fields_uncompressed_20to30us.h5"
)
OUTPUT = ROOT / "workdirs" / "radaz_e20_native_g2_fine_only_pilot"
CHANNELS = ("electron_den", "ion_den", "phi")
DISPLAY = (r"$n_e$", r"$n_i$", r"$\phi$")
PRE = 10
AFT = 10
EPS = np.finfo(np.float64).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--fine-h5", type=Path, default=FINE_H5)
    parser.add_argument("--native-h5", type=Path, default=NATIVE_H5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--phase", choices=("all", "a", "bc"), default="all")
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def checkpoint_path(workdir: Path) -> Path:
    for candidate in (
        workdir / "checkpoints" / "best.ckpt",
        workdir / "best.ckpt",
        workdir / "checkpoints" / "last.ckpt",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No checkpoint under {workdir}")


def protocol(args: argparse.Namespace) -> dict:
    return {
        "declared_before_pilot_results": True,
        "case": {"Bx_mT": 20.0, "Ez_kVm": 20.0},
        "checkpoint": str(checkpoint_path(args.workdir).resolve()),
        "checkpoint_training": {
            "inputs": "fine PIC only; pooled E10 and E20",
            "artificial_coarse_seen": False,
            "native_coarse_seen": False,
            "strict_E20_only": False,
            "role": "existing no-artificial-coarse screening control",
        },
        "time_blocks_us": {
            "fine_manifold_fit": [12.0, 24.0],
            "synthetic_lifting_validation": [24.0, 27.0],
            "native_test": [27.0, 30.0],
            "free_rollout_initial_history_start": 24.0,
        },
        "experiment_A": {
            "task": "closed-loop fine dynamics from fine/native histories",
            "primary_window_us": [27.0, 30.0],
            "controls": [
                "fine teacher-forced direct10 upper bound",
                "fine-init rollout",
                "native interpolation",
            ],
            "no_output_clipping": True,
        },
        "experiment_B": {
            "sources": ["fine", "artificial G2 block-average", "native G2"],
            "latent": f"last-frame encoder state, adaptive pool {args.pool_size}x{args.pool_size}",
            "PCA_fit": "fine 12-24 us only",
            "diagnostics": [
                "fine-train kNN distance",
                "PCA residual",
                "group-held-out logistic AUC",
                "RBF MMD",
            ],
        },
        "experiment_C": {
            "candidate_manifold": "fine 12-24 us frames",
            "validation": "paired synthetic native-node observations from fine 24-27 us",
            "test": "native G2 27-30 us; independent fine reference used only for statistics",
            "native_observation_operator": "coordinate restriction x[::2], y[::2] on unique nodes",
            "hyperparameters": {
                "alpha_latent": [0.0, 0.25, 0.5, 0.75, 1.0],
                "neighbors": [1, 2, 4, 8],
                "temperature": [0.25, 0.5, 1.0, 2.0],
            },
            "selection": "minimum paired normalized validation MSE; native test untouched",
        },
        "primary_gates": {
            "field": "phi distribution and radial-profile distances no worse than native interpolation",
            "mode": "ne and phi n=1-32 log-power RMSE no worse than native interpolation",
            "transport": "ECDI and full-band transport distribution distances no worse than native interpolation",
            "charge": "ni-ne distribution and n=1-32 mode distances no worse than native interpolation",
            "physics": "median Poisson residual <= 10 times the fine reference floor",
            "observation": "reported separately; strong for C, distributional/short-time only for A",
        },
        "non_success_criteria": [
            "latent overlap alone",
            "dominant n=2 alone",
            "field MSE improvement that worsens charge or physics",
        ],
        "paths": {
            "fine_h5": str(args.fine_h5.resolve()),
            "native_h5": str(args.native_h5.resolve()),
            "manifest": str(args.manifest.resolve()),
            "output": str(args.output_dir.resolve()),
        },
    }


def load_normalization(manifest_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float32)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float32)
    if low.shape != (3,) or np.any(high <= low):
        raise ValueError("Invalid source-pool normalization")
    if manifest["normalization"].get("clip", True):
        raise ValueError("The pilot requires the declared unclipped normalization")
    return low, high, manifest


def select_indices(time_s: np.ndarray, begin_us: float, end_us: float) -> np.ndarray:
    time_us = np.asarray(time_s, dtype=np.float64) * 1.0e6
    selected = np.flatnonzero(
        (time_us >= begin_us - 1.0e-8) & (time_us <= end_us + 1.0e-8)
    )
    if not len(selected) or not np.all(np.diff(selected) == 1):
        raise ValueError(f"Empty/non-contiguous interval {begin_us}-{end_us} us")
    return selected


def load_fine(
    path: Path, begin_us: float, end_us: float, full_radial: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as source:
        time_s = np.asarray(source["axes/time_s"], dtype=np.float64)
        indices = select_indices(time_s, begin_us, end_us)
        radial = 257 if full_radial else 256
        fields = np.empty((len(indices), 3, radial, 256), dtype=np.float32)
        for channel, name in enumerate(CHANNELS):
            fields[:, channel] = np.asarray(
                source[f"fields/{name}"][indices[0] : indices[-1] + 1, :radial, :256],
                dtype=np.float32,
            )
        x_m = np.asarray(source["axes/x_m"], dtype=np.float64)
        y_m = np.asarray(source["axes/y_m"], dtype=np.float64)
    return fields, time_s[indices], x_m, y_m


def load_native(path: Path, begin_us: float, end_us: float):
    with h5py.File(path, "r") as source:
        time_s = np.asarray(source["axes/time_s"], dtype=np.float64)
        indices = select_indices(time_s, begin_us, end_us)
        fields = np.empty((len(indices), 3, 129, 128), dtype=np.float32)
        for channel, name in enumerate(CHANNELS):
            fields[:, channel] = np.asarray(
                source[f"fields/{name}"][indices[0] : indices[-1] + 1, :, :128],
                dtype=np.float32,
            )
        x_m = np.asarray(source["axes/x_m"], dtype=np.float64)
        y_m = np.asarray(source["axes/y_m"], dtype=np.float64)
    return fields, time_s[indices], x_m, y_m


def normalize_model(fields: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Normalize physical 256/257-radial fields and pad to the 260 model height."""
    span = high - low
    normalized = (
        np.asarray(fields, dtype=np.float32) - low[None, :, None, None]
    ) / span[None, :, None, None]
    core = normalized[:, :, :257, :256]
    output = np.pad(core, ((0, 0), (0, 0), (0, 260 - core.shape[2]), (0, 0)), mode="edge")
    return np.asarray(output, dtype=np.float32)


def interpolate_native_full_to_model(
    coarse: np.ndarray,
    coarse_x: np.ndarray,
    coarse_y_unique: np.ndarray,
    fine_x: np.ndarray,
    fine_y: np.ndarray,
) -> np.ndarray:
    """Coordinate interpolation including the physical x=12.8 mm boundary node."""
    if coarse.ndim != 4 or coarse.shape[1:] != (3, 129, 128):
        raise ValueError(f"Expected native shape (T,3,129,128), got {coarse.shape}")
    r0, r1, ra = native_eval.interpolation_stencil(
        coarse_x, fine_x[:257], periodic=False
    )
    a0, a1, aa = native_eval.interpolation_stencil(
        coarse_y_unique, fine_y[:256], periodic=True
    )
    radial = (
        coarse[:, :, r0, :] * (1.0 - ra)[None, None, :, None]
        + coarse[:, :, r1, :] * ra[None, None, :, None]
    )
    core = (
        radial[:, :, :, a0] * (1.0 - aa)[None, None, None, :]
        + radial[:, :, :, a1] * aa[None, None, None, :]
    )
    return np.pad(core, ((0, 0), (0, 0), (0, 3), (0, 0)), mode="edge").astype(
        np.float32
    )


def denormalize_core(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (
        np.asarray(values[:, :, :256, :256], dtype=np.float32)
        * (high - low)[None, :, None, None]
        + low[None, :, None, None]
    )


def build_model(args: argparse.Namespace, device: torch.device) -> SimVP_Model:
    cfg = runpy.run_path(str(args.config))
    model = SimVP_Model(
        in_shape=(PRE, 3, 260, 256),
        hid_S=int(cfg.get("hid_S", 64)),
        hid_T=int(cfg.get("hid_T", 512)),
        N_S=int(cfg.get("N_S", 4)),
        N_T=int(cfg.get("N_T", 8)),
        model_type=cfg.get("model_type", "gSTA"),
        drop_path=float(cfg.get("drop_path", 0.0)),
        spatio_kernel_enc=int(cfg.get("spatio_kernel_enc", 3)),
        spatio_kernel_dec=int(cfg.get("spatio_kernel_dec", 3)),
        aft_seq_length=AFT,
        simvp_direct_aft_seq=bool(cfg.get("simvp_direct_aft_seq", True)),
    )
    loaded = torch.load(str(checkpoint_path(args.workdir)), map_location="cpu")
    state = loaded["state_dict"] if "state_dict" in loaded else loaded
    state = {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()


@torch.inference_mode()
def rollout(model: SimVP_Model, history: np.ndarray, steps: int, device: torch.device):
    current = np.asarray(history, dtype=np.float32)
    if current.shape != (PRE, 3, 260, 256):
        raise ValueError(f"Bad rollout history {current.shape}")
    pieces = []
    finite = True
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        while sum(len(piece) for piece in pieces) < steps:
            tensor = torch.from_numpy(current[None]).to(device)
            prediction = model(tensor)[0].float().cpu().numpy()
            if not np.all(np.isfinite(prediction)):
                finite = False
                break
            pieces.append(prediction)
            current = prediction[-PRE:]
            print(
                f"[A rollout] {min(sum(len(piece) for piece in pieces), steps)}/{steps}",
                flush=True,
            )
    if not pieces:
        return np.empty((0, 3, 260, 256), dtype=np.float32), finite
    return np.concatenate(pieces, axis=0)[:steps], finite


@torch.inference_mode()
def teacher_forced_direct10(
    model: SimVP_Model,
    sequence: np.ndarray,
    first_target: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict disjoint 10-frame blocks from true fine histories."""
    target_starts = np.arange(first_target, len(sequence) - AFT + 1, AFT, dtype=np.int64)
    outputs = []
    targets = []
    for target_start in target_starts:
        input_start = int(target_start) - PRE
        if input_start < 0:
            continue
        tensor = torch.from_numpy(sequence[input_start:int(target_start)][None]).to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            prediction = model(tensor)[0].float().cpu().numpy()
        outputs.append(prediction)
        targets.append(np.arange(target_start, target_start + AFT, dtype=np.int64))
        print(f"[A teacher-forced] {len(outputs)}/{len(target_starts)}", flush=True)
    if not outputs:
        raise RuntimeError("No teacher-forced test windows")
    return np.concatenate(outputs, axis=0), np.concatenate(targets)


def charge_modal_power(values: np.ndarray) -> np.ndarray:
    rho = np.asarray(values[:, 1] - values[:, 0], dtype=np.float32)
    total = np.zeros(65, dtype=np.float64)
    samples = 0
    for start in range(0, len(rho), 16):
        block = rho[start : start + 16]
        coeff = np.fft.rfft(block, axis=-1, norm="forward")[..., :65]
        total += np.sum(np.abs(coeff) ** 2, axis=(0, 1))
        samples += block.shape[0] * block.shape[1]
    return total / max(samples, 1)


def single_modal_summary(reference: np.ndarray, candidate: np.ndarray) -> dict:
    floor = max(float(np.max(reference[1:33])) * 1.0e-14, EPS)
    error = np.log10(np.maximum(candidate[1:33], floor)) - np.log10(
        np.maximum(reference[1:33], floor)
    )
    return {
        "log10_power_rmse_n1_32": float(np.sqrt(np.mean(error**2))),
        "n1_6_power_ratio": float(np.sum(candidate[1:7]) / max(np.sum(reference[1:7]), EPS)),
        "n9_21_power_ratio": float(np.sum(candidate[9:22]) / max(np.sum(reference[9:22]), EPS)),
    }


def temporal_feature(values: np.ndarray) -> np.ndarray:
    phi = np.asarray(values[:, 2], dtype=np.float64)
    coefficient = np.fft.rfft(phi, axis=-1, norm="forward")[:, :, 2]
    return np.mean(coefficient, axis=1)


def temporal_summary(reference: np.ndarray, candidate: np.ndarray, dt_s: float) -> dict:
    ref = temporal_feature(reference)
    cand = temporal_feature(candidate)
    length = min(len(ref), len(cand))
    ref, cand = ref[:length], cand[:length]
    nperseg = min(128, max(16, length // 2))
    frequency, ref_real = welch(ref.real, fs=1.0 / dt_s, nperseg=nperseg)
    _, ref_imag = welch(ref.imag, fs=1.0 / dt_s, nperseg=nperseg)
    _, cand_real = welch(cand.real, fs=1.0 / dt_s, nperseg=nperseg)
    _, cand_imag = welch(cand.imag, fs=1.0 / dt_s, nperseg=nperseg)
    ref_psd = ref_real + ref_imag
    cand_psd = cand_real + cand_imag
    floor = max(float(np.max(ref_psd)) * 1.0e-10, EPS)
    log_rmse = np.sqrt(
        np.mean(
            (
                np.log10(np.maximum(cand_psd[1:], floor))
                - np.log10(np.maximum(ref_psd[1:], floor))
            )
            ** 2
        )
    )
    ref_peak = int(np.argmax(ref_psd[1:]) + 1)
    cand_peak = int(np.argmax(cand_psd[1:]) + 1)
    return {
        "phi_n2_temporal_log10_psd_rmse": float(log_rmse),
        "reference_peak_frequency_MHz": float(frequency[ref_peak] * 1.0e-6),
        "candidate_peak_frequency_MHz": float(frequency[cand_peak] * 1.0e-6),
        "phi_n2_rms_ratio": float(
            np.sqrt(np.mean(np.abs(cand) ** 2))
            / max(float(np.sqrt(np.mean(np.abs(ref) ** 2))), EPS)
        ),
    }


def summarize_candidate(reference: np.ndarray, candidate: np.ndarray, dx: float, dy: float, dt_s: float) -> dict:
    reference = np.asarray(reference[:, :, :256, :256], dtype=np.float32)
    candidate = np.asarray(candidate[:, :, :256, :256], dtype=np.float32)
    result = {"field": {}, "profile": {}}
    for channel, name in enumerate(CHANNELS):
        result["field"][name] = native_eval.distribution_metrics(
            reference[:, channel], candidate[:, channel]
        )
        result["profile"][name] = native_eval.profile_metrics(
            np.mean(reference[:, channel], axis=(0, 2), dtype=np.float64),
            np.mean(candidate[:, channel], axis=(0, 2), dtype=np.float64),
        )
    reference_power = native_eval.modal_power(reference)
    candidate_power = native_eval.modal_power(candidate)
    result["mode"] = native_eval.modal_summary(reference_power, candidate_power)
    reference_transport = native_eval.modal_transport(reference, dy)
    candidate_transport = native_eval.modal_transport(candidate, dy)
    result["transport"] = {
        band: native_eval.distribution_metrics(reference_transport[band], candidate_transport[band])
        for band in reference_transport
    }
    ref_charge = reference[:, 1] - reference[:, 0]
    cand_charge = candidate[:, 1] - candidate[:, 0]
    result["charge"] = {
        "distribution": native_eval.distribution_metrics(ref_charge, cand_charge),
        "profile": native_eval.profile_metrics(
            np.mean(ref_charge, axis=(0, 2), dtype=np.float64),
            np.mean(cand_charge, axis=(0, 2), dtype=np.float64),
        ),
        "mode": single_modal_summary(
            charge_modal_power(reference), charge_modal_power(candidate)
        ),
    }
    result["poisson"] = native_eval.poisson_summary(candidate, dx, dy)
    result["temporal"] = temporal_summary(reference, candidate, dt_s)
    result["modal_power"] = candidate_power
    return result


def gate_summary(fine_summary: dict, baseline: dict, candidate: dict) -> dict:
    def q(summary, channel):
        return summary["field"][channel]["quantile_rmse_over_reference_std"]

    def p(summary, channel):
        return summary["profile"][channel]["relative_l2"]

    result = {
        "field": q(candidate, "phi") <= q(baseline, "phi") and p(candidate, "phi") <= p(baseline, "phi"),
        "mode": all(
            candidate["mode"][channel]["log10_power_rmse_n1_32"]
            <= baseline["mode"][channel]["log10_power_rmse_n1_32"]
            for channel in ("electron_den", "phi")
        ),
        "transport": all(
            candidate["transport"][band]["quantile_rmse_over_reference_std"]
            <= baseline["transport"][band]["quantile_rmse_over_reference_std"]
            for band in ("n9_21", "n1_32")
        ),
        "charge": (
            candidate["charge"]["distribution"]["quantile_rmse_over_reference_std"]
            <= baseline["charge"]["distribution"]["quantile_rmse_over_reference_std"]
            and candidate["charge"]["mode"]["log10_power_rmse_n1_32"]
            <= baseline["charge"]["mode"]["log10_power_rmse_n1_32"]
        ),
        "physics": candidate["poisson"]["relative_poisson_residual_median"]
        <= 10.0 * fine_summary["poisson"]["relative_poisson_residual_median"],
        "attractor": candidate["temporal"]["phi_n2_temporal_log10_psd_rmse"]
        <= baseline["temporal"]["phi_n2_temporal_log10_psd_rmse"],
    }
    result["overall"] = bool(all(result.values()))
    return result


def coarse_consistency(candidate: np.ndarray, native: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    """Compare shared inner native nodes; the model omits the x=12.8 mm boundary."""
    predicted = np.asarray(candidate[:, :, :256:2, :256:2], dtype=np.float64)
    observed = np.asarray(native[:, :, :128, :128], dtype=np.float64)
    span = np.asarray(high - low, dtype=np.float64)
    error = (predicted - observed) / span[None, :, None, None]
    channel = np.sqrt(np.mean(error**2, axis=(0, 2, 3)))
    return {
        "normalized_rmse_all": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse_by_channel": dict(zip(CHANNELS, channel.tolist())),
    }


def plot_modes(path: Path, summaries: dict[str, dict]) -> None:
    modes = np.arange(1, 33)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for channel, axis in enumerate(axes):
        for (name, summary), style in zip(summaries.items(), styles):
            power = summary["modal_power"][channel]
            axis.semilogy(modes, np.maximum(power[1:33], EPS), linestyle=style, label=name)
        axis.set_title(DISPLAY[channel])
        axis.set_xlabel("azimuthal mode n")
        axis.axvspan(9, 21, color="tab:red", alpha=0.08)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean modal power")
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_experiment_a(
    args: argparse.Namespace,
    model: SimVP_Model,
    device: torch.device,
    low: np.ndarray,
    high: np.ndarray,
) -> dict:
    fine, fine_time, fine_x, fine_y = load_fine(args.fine_h5, 24.0, 30.0, full_radial=True)
    native, native_time, native_x, native_y = load_native(args.native_h5, 24.0, 30.0)
    if not np.allclose(fine_time, native_time, atol=1.0e-14):
        raise ValueError("Fine/native time axes differ in Experiment A")
    if not np.allclose(fine_x[::2], native_x, atol=1.0e-14):
        raise ValueError("Fine/native radial node coordinates are not nested")
    if not np.allclose(fine_y[:256:2], native_y[:128], atol=1.0e-14):
        raise ValueError("Fine/native azimuthal node coordinates are not nested")
    native_model_physical = interpolate_native_full_to_model(
        native, native_x, native_y[:128], fine_x, fine_y
    )
    fine_normalized = normalize_model(fine, low, high)
    native_normalized = normalize_model(native_model_physical, low, high)
    first_test_target = int(np.flatnonzero(fine_time * 1.0e6 >= 27.0 - 1.0e-8)[0])
    teacher_norm, teacher_indices = teacher_forced_direct10(
        model, fine_normalized, first_test_target, device
    )
    teacher_truth_norm = fine_normalized[teacher_indices]
    teacher_mse = float(
        np.mean(
            (
                teacher_norm[:, :, :257, :]
                - teacher_truth_norm[:, :, :257, :]
            ).astype(np.float64)
            ** 2
        )
    )
    teacher_prediction = denormalize_core(teacher_norm, low, high)
    teacher_truth = fine[teacher_indices, :, :256, :256]
    steps = len(fine) - PRE
    fine_rollout_norm, fine_finite = rollout(model, fine_normalized[:PRE], steps, device)
    native_rollout_norm, native_finite = rollout(model, native_normalized[:PRE], steps, device)
    prediction_time = fine_time[PRE : PRE + min(len(fine_rollout_norm), len(native_rollout_norm))]
    test = np.flatnonzero(prediction_time * 1.0e6 >= 27.0 - 1.0e-8)
    if not len(test):
        raise RuntimeError("No 27-30 us rollout frames")
    count = int(test[-1]) + 1
    prediction_time = prediction_time[:count]
    fine_reference = fine[PRE : PRE + count, :, :256, :256][test]
    native_reference = native_model_physical[PRE : PRE + count, :, :256, :256][test]
    fine_rollout = denormalize_core(fine_rollout_norm[:count], low, high)[test]
    native_rollout = denormalize_core(native_rollout_norm[:count], low, high)[test]
    native_test_raw = native[PRE : PRE + count][test]
    dx = float(np.median(np.diff(fine_x)))
    dy = float(np.median(np.diff(fine_y[:256])))
    dt_s = float(np.median(np.diff(prediction_time)))
    fine_self = summarize_candidate(fine_reference, fine_reference, dx, dy, dt_s)
    summaries = {
        "fine_reference": fine_self,
        "native_interpolation": summarize_candidate(fine_reference, native_reference, dx, dy, dt_s),
        "fine_teacher_forced_direct10": summarize_candidate(
            teacher_truth, teacher_prediction, dx, dy, dt_s
        ),
        "fine_init_rollout": summarize_candidate(fine_reference, fine_rollout, dx, dy, dt_s),
        "native_init_rollout": summarize_candidate(fine_reference, native_rollout, dx, dy, dt_s),
    }
    gates = {
        name: gate_summary(fine_self, summaries["native_interpolation"], summary)
        for name, summary in summaries.items()
        if name not in ("fine_reference", "native_interpolation")
    }
    consistency = {
        "native_interpolation": coarse_consistency(native_reference, native_test_raw, low, high),
        "fine_init_rollout": coarse_consistency(fine_rollout, native_test_raw, low, high),
        "native_init_rollout": coarse_consistency(native_rollout, native_test_raw, low, high),
    }
    plot_modes(args.output_dir / "experiment_A_mode_spectra.png", summaries)
    result = {
        "task": "fine-only closed-loop dynamics; checkpoint trained on fine E10+E20",
        "time_us": [float(prediction_time[test[0]] * 1.0e6), float(prediction_time[test[-1]] * 1.0e6)],
        "frames": len(test),
        "teacher_forced_frames": len(teacher_indices),
        "teacher_forced_normalized_mse": teacher_mse,
        "finite": {"fine_init": fine_finite, "native_init": native_finite},
        "summaries": summaries,
        "gates": gates,
        "coarse_consistency": consistency,
    }
    write_json(args.output_dir / "experiment_A_summary.json", result)
    return result


@torch.inference_mode()
def encode_frames(
    model: SimVP_Model,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int,
    pool_size: int,
    label: str,
) -> np.ndarray:
    pieces = []
    for start in range(0, len(frames), batch_size):
        stop = min(start + batch_size, len(frames))
        tensor = torch.from_numpy(np.asarray(frames[start:stop], dtype=np.float32)).to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            encoded, _ = model.enc(tensor)
            pooled = F.adaptive_avg_pool2d(encoded, (pool_size, pool_size))
        pieces.append(pooled.float().cpu().numpy().reshape(stop - start, -1))
        if stop == len(frames) or stop % 100 < batch_size:
            print(f"[B encode {label}] {stop}/{len(frames)}", flush=True)
    return np.concatenate(pieces).astype(np.float32)


def whitened_scores(pca: PCA, values: np.ndarray) -> np.ndarray:
    return pca.transform(values) / np.sqrt(
        np.maximum(pca.explained_variance_[None], 1.0e-12)
    )


def rbf_mmd(left: np.ndarray, right: np.ndarray) -> float:
    count = min(len(left), len(right), 300)
    a = np.asarray(left[:count], dtype=np.float64)
    b = np.asarray(right[:count], dtype=np.float64)
    joined = np.concatenate((a, b), axis=0)
    distances = np.sum((joined[:, None] - joined[None, :]) ** 2, axis=-1)
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    kernel = np.exp(-distances / max(2.0 * bandwidth, EPS))
    n = len(a)
    return float(kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2.0 * kernel[:n, n:].mean())


def classifier_auc(fine: np.ndarray, other: np.ndarray) -> float:
    count = min(len(fine), len(other))
    x = np.concatenate((fine[:count], other[:count]), axis=0)
    y = np.concatenate((np.zeros(count), np.ones(count)))
    blocks = np.arange(count) * 5 // count
    groups = np.concatenate((blocks, blocks))
    predictions = np.full(2 * count, np.nan)
    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
        classifier = LogisticRegression(max_iter=1000, C=1.0)
        classifier.fit(x[train], y[train])
        predictions[test] = classifier.predict_proba(x[test])[:, 1]
    return float(roc_auc_score(y, predictions))


def latent_source_summary(
    train_flat: np.ndarray,
    train_scores: np.ndarray,
    pca: PCA,
    fine_scores: np.ndarray,
    source_flat: np.ndarray,
    source_scores: np.ndarray,
) -> dict:
    neighbor = NearestNeighbors(n_neighbors=5).fit(train_scores)
    fine_distance = neighbor.kneighbors(fine_scores, return_distance=True)[0][:, -1]
    source_distance = neighbor.kneighbors(source_scores, return_distance=True)[0][:, -1]
    reconstruction = pca.inverse_transform(pca.transform(source_flat))
    fine_reconstruction = pca.inverse_transform(pca.transform(train_flat))
    residual = np.sqrt(np.mean((source_flat - reconstruction) ** 2, axis=1))
    fine_residual = np.sqrt(np.mean((train_flat - fine_reconstruction) ** 2, axis=1))
    return {
        "median_knn_distance": float(np.median(source_distance)),
        "median_knn_distance_over_fine_holdout": float(
            np.median(source_distance) / max(float(np.median(fine_distance)), EPS)
        ),
        "p95_knn_distance_over_fine_holdout_p95": float(
            np.quantile(source_distance, 0.95) / max(float(np.quantile(fine_distance, 0.95)), EPS)
        ),
        "median_pca_residual_over_fine_train": float(
            np.median(residual) / max(float(np.median(fine_residual)), EPS)
        ),
        "group_heldout_logistic_auc_vs_fine": classifier_auc(fine_scores, source_scores),
        "rbf_mmd2_vs_fine": rbf_mmd(fine_scores, source_scores),
    }


def observation_features(coarse: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Low-k radial/Fourier features on the 129x128 native-node field."""
    normalized = (
        np.asarray(coarse, dtype=np.float64) - low[None, :, None, None]
    ) / (high - low)[None, :, None, None]
    features = []
    radial_groups = np.array_split(np.arange(129), 8)
    for group in radial_groups:
        average = np.mean(normalized[:, :, group, :], axis=2)
        coeff = np.fft.rfft(average, axis=-1, norm="forward")[:, :, :17]
        features.extend((coeff.real.reshape(len(coarse), -1), coeff.imag.reshape(len(coarse), -1)))
    radial = np.mean(normalized, axis=-1)
    for group in np.array_split(np.arange(129), 32):
        features.append(np.mean(radial[:, :, group], axis=2))
    return np.concatenate(features, axis=1).astype(np.float32)


def distance_matrix(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    q2 = np.sum(np.asarray(query, dtype=np.float64) ** 2, axis=1, keepdims=True)
    c2 = np.sum(np.asarray(candidate, dtype=np.float64) ** 2, axis=1)[None]
    distances = q2 + c2 - 2.0 * np.asarray(query, dtype=np.float64) @ np.asarray(candidate, dtype=np.float64).T
    return np.maximum(distances, 0.0) / candidate.shape[1]


def neighbor_weights(distances: np.ndarray, k: int, temperature: float):
    indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    local = np.take_along_axis(distances, indices, axis=1)
    order = np.argsort(local, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    local = np.take_along_axis(local, order, axis=1)
    if k == 1:
        return indices, np.ones((len(distances), 1), dtype=np.float64)
    scale = np.maximum(np.median(local[:, -1] - local[:, 0]), 1.0e-12)
    logits = -(local - local[:, :1]) / (temperature * scale)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return indices, weights


def weighted_fields(candidates: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.empty((len(indices),) + candidates.shape[1:], dtype=np.float32)
    for row in range(len(indices)):
        output[row] = np.tensordot(
            weights[row].astype(np.float32), candidates[indices[row]], axes=(0, 0)
        )
    return output


def paired_normalized_mse(truth: np.ndarray, estimate: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    error = (
        np.asarray(estimate[:, :, ::4, ::4], dtype=np.float64)
        - np.asarray(truth[:, :, ::4, ::4], dtype=np.float64)
    ) / (high - low)[None, :, None, None]
    return float(np.mean(error**2))


def fit_feature_pca(train: np.ndarray, components: int) -> tuple[PCA, np.ndarray]:
    count = min(components, len(train) - 1, train.shape[1])
    model = PCA(n_components=count, svd_solver="randomized", random_state=0)
    scores = model.fit_transform(train) / np.sqrt(
        np.maximum(model.explained_variance_[None], 1.0e-12)
    )
    return model, scores


def run_experiments_bc(
    args: argparse.Namespace,
    model: SimVP_Model,
    device: torch.device,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[dict, dict]:
    fine_train, train_time, fine_x, fine_y = load_fine(args.fine_h5, 12.0, 23.985, True)
    fine_val, val_time, _, _ = load_fine(args.fine_h5, 24.0, 26.985, True)
    fine_test, test_time, _, _ = load_fine(args.fine_h5, 27.0, 30.0, True)
    native_test, native_time, native_x, native_y = load_native(args.native_h5, 27.0, 30.0)
    if not np.allclose(test_time, native_time, atol=1.0e-14):
        raise ValueError("Fine/native test time axes differ")
    if not np.allclose(fine_x[::2], native_x, atol=1.0e-14) or not np.allclose(
        fine_y[:256:2], native_y[:128], atol=1.0e-14
    ):
        raise ValueError("Fine/native nested-grid coordinate check failed")

    train_model = normalize_model(fine_train, low, high)
    val_model = normalize_model(fine_val, low, high)
    test_model = normalize_model(fine_test, low, high)
    artificial_test_model = make_g2_interpolated(test_model)
    synthetic_val_node = fine_val[:, :, ::2, ::2]
    synthetic_val_interp = interpolate_native_full_to_model(
        synthetic_val_node, fine_x[::2], fine_y[:256:2], fine_x, fine_y
    )
    synthetic_val_model = normalize_model(synthetic_val_interp, low, high)
    native_test_interp = interpolate_native_full_to_model(
        native_test, native_x, native_y[:128], fine_x, fine_y
    )
    native_test_model = normalize_model(native_test_interp, low, high)

    latent_flat = {
        "fine_train": encode_frames(model, train_model, device, args.encoder_batch_size, args.pool_size, "fine_train"),
        "fine_test": encode_frames(model, test_model, device, args.encoder_batch_size, args.pool_size, "fine_test"),
        "artificial_g2_test": encode_frames(model, artificial_test_model, device, args.encoder_batch_size, args.pool_size, "artificial_test"),
        "native_g2_test": encode_frames(model, native_test_model, device, args.encoder_batch_size, args.pool_size, "native_test"),
        "synthetic_node_val": encode_frames(model, synthetic_val_model, device, args.encoder_batch_size, args.pool_size, "synthetic_val"),
    }
    components = min(args.pca_components, len(latent_flat["fine_train"]) - 1)
    latent_pca = PCA(n_components=components, svd_solver="randomized", random_state=0)
    latent_pca.fit(latent_flat["fine_train"])
    latent_scores = {
        name: whitened_scores(latent_pca, values) for name, values in latent_flat.items()
    }
    b_sources = {}
    for name in ("artificial_g2_test", "native_g2_test"):
        b_sources[name] = latent_source_summary(
            latent_flat["fine_train"],
            latent_scores["fine_train"],
            latent_pca,
            latent_scores["fine_test"],
            latent_flat[name],
            latent_scores[name],
        )
    experiment_b = {
        "checkpoint_training": "fine E10+E20 only; no coarse input",
        "PCA_components": components,
        "PCA_explained_variance_ratio_sum": float(np.sum(latent_pca.explained_variance_ratio_)),
        "fit_frames": len(fine_train),
        "test_frames": len(fine_test),
        "sources": b_sources,
    }
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    selections = {
        "fine train": (latent_scores["fine_train"], 0.25, "."),
        "fine test": (latent_scores["fine_test"], 0.7, "o"),
        "artificial G2 test": (latent_scores["artificial_g2_test"], 0.7, "x"),
        "native G2 test": (latent_scores["native_g2_test"], 0.7, "+"),
    }
    for label, (values, alpha, marker) in selections.items():
        axis.scatter(values[:, 0], values[:, 1], s=10, alpha=alpha, marker=marker, label=label)
    axis.set(xlabel="whitened latent PC1", ylabel="whitened latent PC2", title="Fine-only encoder latent overlap")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.savefig(args.output_dir / "experiment_B_latent_PCA.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(
        args.output_dir / "experiment_B_latent_scores.npz",
        **{name: values.astype(np.float32) for name, values in latent_scores.items()},
        explained_variance_ratio=latent_pca.explained_variance_ratio_.astype(np.float32),
    )
    write_json(args.output_dir / "experiment_B_summary.json", experiment_b)

    candidate_coarse = fine_train[:, :, ::2, ::2]
    observation_train = observation_features(candidate_coarse, low, high)
    observation_val = observation_features(synthetic_val_node, low, high)
    observation_native = observation_features(native_test, low, high)
    observation_pca, observation_train_scores = fit_feature_pca(
        observation_train, args.pca_components
    )
    observation_val_scores = observation_pca.transform(observation_val) / np.sqrt(
        np.maximum(observation_pca.explained_variance_[None], 1.0e-12)
    )
    observation_native_scores = observation_pca.transform(observation_native) / np.sqrt(
        np.maximum(observation_pca.explained_variance_[None], 1.0e-12)
    )
    latent_val_distance = distance_matrix(latent_scores["synthetic_node_val"], latent_scores["fine_train"])
    latent_native_distance = distance_matrix(latent_scores["native_g2_test"], latent_scores["fine_train"])
    observation_val_distance = distance_matrix(observation_val_scores, observation_train_scores)
    observation_native_distance = distance_matrix(observation_native_scores, observation_train_scores)

    candidates_for_tuning = fine_train[:, :, :257, :256]
    hyper_rows = []
    selected_by_alpha = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        combined = alpha * latent_val_distance + (1.0 - alpha) * observation_val_distance
        best = None
        for neighbors in (1, 2, 4, 8):
            for temperature in (0.25, 0.5, 1.0, 2.0):
                indices, weights = neighbor_weights(combined, neighbors, temperature)
                estimate = weighted_fields(candidates_for_tuning[:, :, ::4, ::4], indices, weights)
                truth = fine_val[:, :, ::4, ::4]
                error = (estimate.astype(np.float64) - truth.astype(np.float64)) / (
                    high - low
                )[None, :, None, None]
                score = float(np.mean(error**2))
                row = {
                    "alpha_latent": alpha,
                    "neighbors": neighbors,
                    "temperature": temperature,
                    "paired_validation_normalized_mse": score,
                }
                hyper_rows.append(row)
                if best is None or score < best["paired_validation_normalized_mse"]:
                    best = row
        selected_by_alpha[alpha] = best
    selected = min(hyper_rows, key=lambda row: row["paired_validation_normalized_mse"])

    methods = {
        "observation_only": selected_by_alpha[0.0],
        "combined": selected,
        "latent_only": selected_by_alpha[1.0],
    }
    lifted = {}
    neighbor_artifacts = {}
    continuity = {}
    for name, choice in methods.items():
        alpha = float(choice["alpha_latent"])
        distance = alpha * latent_native_distance + (1.0 - alpha) * observation_native_distance
        indices, weights = neighbor_weights(
            distance, int(choice["neighbors"]), float(choice["temperature"])
        )
        lifted[name] = weighted_fields(candidates_for_tuning, indices, weights)
        neighbor_artifacts[f"{name}_indices"] = indices.astype(np.int32)
        neighbor_artifacts[f"{name}_weights"] = weights.astype(np.float32)
        top_time_us = train_time[indices[:, 0]] * 1.0e6
        entropy = -np.sum(weights * np.log(np.maximum(weights, EPS)), axis=1)
        continuity[name] = {
            "top_neighbor_repeat_fraction": float(np.mean(indices[1:, 0] == indices[:-1, 0])),
            "median_abs_top_neighbor_time_jump_us": float(
                np.median(np.abs(np.diff(top_time_us)))
            ),
            "mean_weight_entropy": float(np.mean(entropy)),
        }
    np.savez_compressed(
        args.output_dir / "experiment_C_neighbor_weights.npz",
        fine_train_time_s=train_time,
        native_test_time_s=native_time,
        **neighbor_artifacts,
    )

    dx = float(np.median(np.diff(fine_x)))
    dy = float(np.median(np.diff(fine_y[:256])))
    dt_s = float(np.median(np.diff(test_time)))
    reference = fine_test[:, :, :256, :256]
    native_interp_core = native_test_interp[:, :, :256, :256]
    fine_self = summarize_candidate(reference, reference, dx, dy, dt_s)
    summaries = {
        "fine_reference": fine_self,
        "native_interpolation": summarize_candidate(reference, native_interp_core, dx, dy, dt_s),
    }
    consistency = {
        "native_interpolation": coarse_consistency(native_interp_core, native_test, low, high)
    }
    gates = {}
    for name, values in lifted.items():
        summaries[name] = summarize_candidate(reference, values[:, :, :256, :256], dx, dy, dt_s)
        consistency[name] = coarse_consistency(values[:, :, :256, :256], native_test, low, high)
        gates[name] = gate_summary(fine_self, summaries["native_interpolation"], summaries[name])
        gates[name]["observation"] = (
            consistency[name]["normalized_rmse_all"]
            <= consistency["native_interpolation"]["normalized_rmse_all"] + 1.0e-7
        )
        gates[name]["overall_with_observation"] = bool(
            gates[name]["overall"] and gates[name]["observation"]
        )
    plot_modes(args.output_dir / "experiment_C_mode_spectra.png", summaries)
    experiment_c = {
        "manifold_candidates": len(fine_train),
        "synthetic_validation_frames": len(fine_val),
        "native_test_frames": len(native_test),
        "observation_PCA_explained_variance_ratio_sum": float(
            np.sum(observation_pca.explained_variance_ratio_)
        ),
        "hyperparameter_grid": hyper_rows,
        "selected": selected,
        "selected_controls": {str(key): value for key, value in selected_by_alpha.items()},
        "temporal_continuity_diagnostics": continuity,
        "summaries": summaries,
        "coarse_consistency": consistency,
        "gates": gates,
    }
    write_json(args.output_dir / "experiment_C_summary.json", experiment_c)
    return experiment_b, experiment_c


def compact_summary(a: dict | None, b: dict | None, c: dict | None) -> dict:
    result = {}
    if a is not None:
        result["A"] = {
            "frames": a["frames"],
            "finite": a["finite"],
            "gates": a["gates"],
            "coarse_consistency": a["coarse_consistency"],
        }
    if b is not None:
        result["B"] = b
    if c is not None:
        result["C"] = {
            "selected": c["selected"],
            "gates": c["gates"],
            "coarse_consistency": c["coarse_consistency"],
        }
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "protocol.json", protocol(args))
    if args.protocol_only:
        print(f"[protocol] {args.output_dir / 'protocol.json'}", flush=True)
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    low, high, manifest = load_normalization(args.manifest)
    started = time.time()
    model = build_model(args, device)
    a = b = c = None
    if args.phase in ("all", "a"):
        a = run_experiment_a(args, model, device, low, high)
    if args.phase in ("all", "bc"):
        b, c = run_experiments_bc(args, model, device, low, high)
    final = {
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "normalization": manifest["normalization"],
        "results": compact_summary(a, b, c),
    }
    write_json(args.output_dir / "final_summary.json", final)
    print(json.dumps(json_safe(final["results"]), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
