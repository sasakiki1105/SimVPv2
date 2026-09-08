"""Evaluate the RadAz conditioned 2x2 factorial (U-D, C-D, U-P, C-P).

Pre-registered protocol (Section 15.5 of ICL_reserch_memo.md):

* Window: the frame-disjoint test split, frames 1800-1999 (27.0-30.0 us).
  For the six source cases these frames were never used for training
  (0-1599) or checkpoint selection (1600-1799); the four axis holdouts were
  never used at all.  Every cell therefore sees the same unseen frames.
* Primary: direct10 -- ten non-overlapping windows per case, input frames
  s..s+9, target s+10..s+19, s = 1800, 1820, ..., 1980.
* Secondary: closed-loop 40 -- four chained direct10 blocks from the single
  history 1800-1809, predicting 1810-1849.
* Baseline: copy/persistence (last input frame held for all ten outputs).
* All four cells receive byte-identical 5-channel samples; the unconditioned
  cells strip the two condition channels inside the model.

The q-coordinate mode and modal-transport observables are computed with the
training loss module itself (PEPAPICSpectralLoss._band_coefficients and
._interpolate_to_q) so the evaluation numbers are the same functional the
P cells were trained on.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import time
from pathlib import Path

# Same MKL/OpenMP workaround the training queue applies in this environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import h5py
import numpy as np
import torch

from openstl.models.simvp_model import SimVP_Model
from openstl.models.simvp_factory import build_simvp_model
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

PRE = 10
AFT = 10
VALID_H = 257
VALID_W = 256
MODEL_H = 260
MODEL_W = 256
TEST_START = 1800
TEST_STOP = 2000  # exclusive frame index for direct10 targets
CLOSED_LOOP_BLOCKS = 4

ROOT = Path(r"C:\Users\astro\research\SimVPv2")
MANIFEST = ROOT / "workdirs/2D_RadAz/radaz_conditioned_factorial_manifests/radaz_axis_factorial_manifest.json"

CELLS = {
    "U-D": {
        "workdir": ROOT / "workdirs/2D_RadAz/radaz_axis_factorial_UD_direct10_bs1_60ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_radaz_factorial_UD_60ep.py",
        "conditioned": False,
        "physics_loss": False,
    },
    "C-D": {
        "workdir": ROOT / "workdirs/2D_RadAz/radaz_axis_factorial_CD_film_direct10_bs1_60ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_radaz_factorial_CD_60ep.py",
        "conditioned": True,
        "physics_loss": False,
    },
    "U-P": {
        "workdir": ROOT / "workdirs/2D_RadAz/radaz_axis_factorial_UP_qtransport_direct10_bs1_60ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_radaz_factorial_UP_60ep.py",
        "conditioned": False,
        "physics_loss": True,
    },
    "C-P": {
        "workdir": ROOT / "workdirs/2D_RadAz/radaz_axis_factorial_CP_film_qtransport_direct10_bs1_60ep",
        "config": ROOT / "configs/custom/pepapic/SimVP_gSTA_radaz_factorial_CP_60ep.py",
        "conditioned": True,
        "physics_loss": True,
    },
}

ELECTRON_MASS = 9.1093837015e-31
ELECTRON_CHARGE = 1.602176634e-19


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def condition_vector(case, manifest):
    """Reproduce dataloader_pepapic_h5._condition_values for (log_vE, log_n0)."""
    e_v_m = float(case["Ez_kVm"]) * 1.0e3
    b_t = float(case["B_mT"]) * 1.0e-3
    ly = float(case.get("Ly_m", 1.28e-2))
    log_ve = float(np.log(e_v_m / b_t))
    n0 = ELECTRON_CHARGE * b_t ** 2 * ly / (2.0 * np.pi * ELECTRON_MASS * e_v_m)
    log_n0 = float(np.log(n0))
    norm = manifest["condition_normalization"]
    if list(norm["names"]) != ["log_vE", "log_n0"]:
        raise ValueError("Unexpected condition names")
    mean = np.asarray(norm["mean"], dtype=np.float64)
    std = np.asarray(norm["std"], dtype=np.float64)
    raw = np.asarray([log_ve, log_n0], dtype=np.float64)
    return ((raw - mean) / std).astype(np.float32), float(np.exp(log_ve)), float(n0)


def load_case_frames(case, low, high, start, stop):
    """Return (normalized [T,3,260,256], physical [T,3,257,256], x_m, y_m)."""
    channels = list(case["channels"])
    with h5py.File(case["path"], "r") as handle:
        physical = np.empty((stop - start, len(channels), VALID_H, VALID_W),
                            dtype=np.float64)
        for index, name in enumerate(channels):
            physical[:, index] = np.asarray(
                handle["fields/" + name][start:stop, :VALID_H, :VALID_W],
                dtype=np.float64,
            )
        x_m = np.asarray(handle["axes/x_m"], dtype=np.float64)
        y_m = np.asarray(handle["axes/y_m"], dtype=np.float64)
    span = (high - low)[None, :, None, None]
    normalized = (physical - low[None, :, None, None]) / span
    model = np.empty((stop - start, len(channels), MODEL_H, MODEL_W), dtype=np.float32)
    model[:, :, :VALID_H, :VALID_W] = normalized.astype(np.float32)
    model[:, :, VALID_H:, :VALID_W] = model[:, :, VALID_H - 1: VALID_H, :VALID_W]
    if not np.all(np.isfinite(model)):
        raise ValueError("Non-finite frames in " + str(case["case_key"]))
    return model, physical.astype(np.float32), x_m, y_m


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def build_model(config, checkpoint, device):
    cfg = runpy.run_path(str(config))
    condition_dim = int(cfg.get("condition_dim", 0))
    # Forward the COMPLETE configuration, including anisotropic downsampling,
    # normalization and all future structural options, just as training does.
    cfg = {k: v for k, v in cfg.items() if not k.startswith("__")}
    cfg["in_shape"] = (PRE, 3 + condition_dim, MODEL_H, MODEL_W)
    cfg.setdefault("aft_seq_length", AFT)
    cfg.setdefault("simvp_direct_aft_seq", True)
    cfg.setdefault("out_channels", 3)
    model = build_simvp_model(cfg)
    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = loaded["state_dict"] if "state_dict" in loaded else loaded
    state = dict((k[6:] if k.startswith("model.") else k, v) for k, v in state.items())
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError("Checkpoint mismatch: missing=%s unexpected=%s"
                           % (missing, unexpected))
    return model.to(device).eval(), int(loaded.get("epoch", -1))


@torch.inference_mode()
def predict_direct10(model, inputs, device):
    """inputs [B,10,5,260,256] -> [B,10,3,260,256], one window at a time."""
    outputs = []
    for index in range(len(inputs)):
        tensor = torch.from_numpy(inputs[index: index + 1]).to(device)
        prediction = model(tensor)
        if not torch.all(torch.isfinite(prediction)):
            raise RuntimeError("Non-finite direct10 prediction")
        outputs.append(prediction[0].float().cpu().numpy())
    return np.stack(outputs, axis=0)


@torch.inference_mode()
def closed_loop(model, history, condition, blocks, device):
    """history [10,5,260,256] -> [blocks*10,3,260,256]; feeds predictions back."""
    current = np.array(history, dtype=np.float32, copy=True)
    pieces = []
    finite = True
    for _ in range(blocks):
        tensor = torch.from_numpy(current[None]).to(device)
        prediction = model(tensor)[0].float().cpu().numpy()
        if not np.all(np.isfinite(prediction)):
            finite = False
            break
        pieces.append(prediction)
        nxt = np.empty_like(current)
        nxt[:, :3] = prediction[-PRE:]
        nxt[:, 3:] = condition[None, :, None, None]
        current = nxt
    if not pieces:
        return np.empty((0, 3, MODEL_H, MODEL_W), dtype=np.float32), finite
    return np.concatenate(pieces, axis=0), finite


# --------------------------------------------------------------------------
# observables
# --------------------------------------------------------------------------
def denormalize(normalized, low, high):
    span = (high - low).astype(np.float64)
    core = normalized[..., :VALID_H, :VALID_W].astype(np.float64)
    return core * span[None, None, :, None, None] \
        + low.astype(np.float64)[None, None, :, None, None]


def azimuthal_ey(phi, dy):
    """-d(phi)/dy with the periodic central difference used by the loss."""
    return -(np.roll(phi, -1, axis=-1) - np.roll(phi, 1, axis=-1)) / (2.0 * dy)


def field_metrics(pred, truth, dy):
    """pred/truth: [B,T,3,257,256] in physical units."""
    out = {}
    names = ("electron_den", "ion_den", "phi")
    for index, name in enumerate(names):
        out["mse_" + name] = float(np.mean((pred[:, :, index] - truth[:, :, index]) ** 2))
    ey_pred = azimuthal_ey(pred[:, :, 2], dy)
    ey_true = azimuthal_ey(truth[:, :, 2], dy)
    out["mse_Ey"] = float(np.mean((ey_pred - ey_true) ** 2))
    return out


def q_observables(loss_module, pred_norm, truth_norm, mode_n0, drift, device):
    """Reproduce the training q-mode/transport functionals plus diagnostics."""
    pred = torch.from_numpy(pred_norm).to(device)
    truth = torch.from_numpy(truth_norm).to(device)
    batch = pred.shape[0]
    n0 = torch.full((batch,), float(mode_n0), dtype=torch.float32, device=device)
    ve = torch.full((batch,), float(drift), dtype=torch.float32, device=device)
    pred_coeff = loss_module._band_coefficients(pred, physical_units=False)
    true_coeff = loss_module._band_coefficients(truth, physical_units=False)
    pred_q, mask = loss_module._interpolate_to_q(pred_coeff, n0)
    true_q, _ = loss_module._interpolate_to_q(true_coeff, n0)
    complex_loss = float(loss_module._complex_mode_loss(pred_q, true_q, mask))
    transport_loss = float(loss_module._transport_loss(pred_q, true_q, mask, ve, n0))

    def unpack(values):
        return values[..., 0].to(torch.float64), values[..., 1].to(torch.float64)

    # field axis order inside _band_coefficients: 0=phi, 1=electron_den, 2=Ey
    pr, pi = unpack(pred_q[:, :, 1])
    tr, ti = unpack(true_q[:, :, 1])
    epr, epi = unpack(pred_q[:, :, 2])
    etr, eti = unpack(true_q[:, :, 2])
    valid = mask[:, :, 0, :, :, 0].to(torch.float64)

    pred_amp = torch.sqrt(pr * pr + pi * pi)
    true_amp = torch.sqrt(tr * tr + ti * ti)
    cross_pred_r = pr * epr + pi * epi
    cross_pred_i = pi * epr - pr * epi
    cross_true_r = tr * etr + ti * eti
    cross_true_i = ti * etr - tr * eti
    magnetic_t = mode_n0 * (2.0 * np.pi * ELECTRON_MASS) * drift \
        / (ELECTRON_CHARGE * 1.28e-2)
    relative_b = magnetic_t / 0.020
    gamma_pred = -cross_pred_r / relative_b
    gamma_true = -cross_true_r / relative_b

    weight = valid.sum()

    def masked_mean(values):
        return float((values * valid).sum() / torch.clamp(weight, min=1.0))

    phase_pred = torch.atan2(cross_pred_i, cross_pred_r)
    phase_true = torch.atan2(cross_true_i, cross_true_r)
    delta = torch.atan2(torch.sin(phase_pred - phase_true),
                        torch.cos(phase_pred - phase_true))
    power = torch.sqrt(cross_true_r ** 2 + cross_true_i ** 2) * valid
    cross_phase_error = float(
        (torch.abs(delta) * power).sum() / torch.clamp(power.sum(), min=1e-300)
    )

    q_grid = loss_module.q_grid.to(torch.float64)
    ecdi = ((q_grid >= 0.85) & (q_grid <= 1.15)).to(torch.float64)
    ecdi_w = valid * ecdi[None, None, None, :]

    def ecdi_mean(values):
        return float((values * ecdi_w).sum() / torch.clamp(ecdi_w.sum(), min=1.0))

    return {
        "complex_mode_loss": complex_loss,
        "transport_loss": transport_loss,
        "amplitude_log_bias": masked_mean(
            torch.log((pred_amp + 1e-30) / (true_amp + 1e-30))
        ),
        "cross_phase_error_rad": cross_phase_error,
        "gamma_mean_pred": masked_mean(gamma_pred),
        "gamma_mean_true": masked_mean(gamma_true),
        "gamma_ecdi_pred": ecdi_mean(gamma_pred),
        "gamma_ecdi_true": ecdi_mean(gamma_true),
        "gamma_mse": masked_mean((gamma_pred - gamma_true) ** 2),
        "amplitude_mse": masked_mean((pred_amp - true_amp) ** 2),
    }


def mode1_power(physical):
    """Raw n=1 azimuthal power of the electron density (B30 observable)."""
    field = physical[:, :, 0].astype(np.float64)
    fluctuation = field - field.mean(axis=-1, keepdims=True)
    coefficients = np.fft.rfft(fluctuation, axis=-1, norm="forward")
    return float(np.mean(np.abs(coefficients[..., 1]) ** 2))


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "workdirs/2D_RadAz/radaz_conditioned_factorial_evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--cases", default="all")
    parser.add_argument("--cells", default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low = np.asarray(manifest["normalization"]["low"], dtype=np.float64)
    high = np.asarray(manifest["normalization"]["high"], dtype=np.float64)
    if manifest["normalization"].get("clip", True):
        raise ValueError("Expected the declared unclipped normalization")
    device = torch.device(args.device)

    cases = manifest["cases"]
    if args.cases != "all":
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["case_key"] in wanted]
    cell_names = list(CELLS) if args.cells == "all" else args.cells.split(",")

    loss_module = PEPAPICSpectralLoss(
        data_root=str(MANIFEST),
        max_mode=64,
        radial_bands=4,
        radial_min_m=0.09e-2,
        radial_max_m=1.19e-2,
        coordinate_system="q_normalized",
        q_min=0.30,
        q_max=1.50,
        q_bins=49,
    ).to(device)

    starts = list(range(TEST_START, TEST_STOP, PRE + AFT))
    starts = [s for s in starts if s + PRE + AFT <= TEST_STOP]

    models = {}
    for name in cell_names:
        spec = CELLS[name]
        ckpt = spec["workdir"] / "checkpoints" / (args.checkpoint + ".ckpt")
        model, epoch = build_model(spec["config"], ckpt, device)
        models[name] = (model, epoch)
        print("[model] %s epoch=%d %s" % (name, epoch, ckpt.name), flush=True)

    results = {}
    for case in cases:
        key = case["case_key"]
        condition, drift, mode_n0 = condition_vector(case, manifest)
        need_stop = max(TEST_START + len(starts) * (PRE + AFT),
                        TEST_START + PRE + CLOSED_LOOP_BLOCKS * AFT)
        model_frames, physical_frames, x_m, y_m = load_case_frames(
            case, low, high, TEST_START, need_stop)
        dy = float(np.median(np.diff(y_m[:VALID_W + 1])))
        offset = TEST_START

        inputs = np.empty((len(starts), PRE, 5, MODEL_H, MODEL_W), dtype=np.float32)
        truth_norm = np.empty((len(starts), AFT, 3, MODEL_H, MODEL_W), dtype=np.float32)
        for index, s in enumerate(starts):
            i0 = s - offset
            inputs[index, :, :3] = model_frames[i0: i0 + PRE]
            inputs[index, :, 3:] = condition[None, :, None, None]
            truth_norm[index] = model_frames[i0 + PRE: i0 + PRE + AFT]
        truth_phys = denormalize(truth_norm, low, high)

        copy_norm = np.repeat(inputs[:, -1:, :3], AFT, axis=1)
        copy_phys = denormalize(copy_norm, low, high)

        case_result = {
            "case_key": key,
            "role": case["role"],
            "B_mT": case["B_mT"],
            "Ez_kVm": case["Ez_kVm"],
            "mode_n0": mode_n0,
            "drift_velocity_m_s": drift,
            "condition_standardized": condition.tolist(),
            "direct10_windows": len(starts),
            "frames": [TEST_START, need_stop - 1],
            "baselines": {},
            "cells": {},
        }
        case_result["baselines"]["copy"] = dict(
            field_metrics(copy_phys, truth_phys, dy),
            **q_observables(loss_module, copy_norm, truth_norm, mode_n0, drift, device)
        )
        case_result["baselines"]["copy"]["mode1_ne_power"] = mode1_power(copy_phys)
        case_result["baselines"]["truth"] = {"mode1_ne_power": mode1_power(truth_phys)}

        for name in cell_names:
            model, epoch = models[name]
            started = time.time()
            pred_norm = predict_direct10(model, inputs, device)
            pred_phys = denormalize(pred_norm, low, high)
            entry = dict(
                field_metrics(pred_phys, truth_phys, dy),
                **q_observables(loss_module, pred_norm, truth_norm,
                                mode_n0, drift, device)
            )
            entry["checkpoint_epoch"] = epoch
            entry["mode1_ne_power"] = mode1_power(pred_phys)

            history = inputs[0]
            rollout, finite = closed_loop(model, history, condition,
                                          CLOSED_LOOP_BLOCKS, device)
            cl = {"finite": bool(finite), "frames": int(len(rollout))}
            if len(rollout):
                count = len(rollout)
                i0 = TEST_START + PRE - offset
                cl_truth = denormalize(model_frames[i0: i0 + count][None], low, high)
                cl_pred = denormalize(rollout[None], low, high)
                cl_copy = denormalize(
                    np.repeat(history[None, -1:, :3], count, axis=1), low, high)
                cl["model"] = field_metrics(cl_pred, cl_truth, dy)
                cl["copy"] = field_metrics(cl_copy, cl_truth, dy)
                blocks = []
                for b in range(count // AFT):
                    sl = slice(b * AFT, (b + 1) * AFT)
                    blocks.append({
                        "block": b,
                        "model": field_metrics(cl_pred[:, sl], cl_truth[:, sl], dy),
                        "copy": field_metrics(cl_copy[:, sl], cl_truth[:, sl], dy),
                    })
                cl["blocks"] = blocks
            entry["closed_loop_40"] = cl
            entry["seconds"] = time.time() - started
            case_result["cells"][name] = entry
            print("[%s] %s done in %.1fs" % (key, name, entry["seconds"]), flush=True)

        results[key] = case_result
        del model_frames, physical_frames, inputs, truth_norm, truth_phys

    payload = {
        "protocol": {
            "window_frames": [TEST_START, TEST_STOP - 1],
            "window_us": [TEST_START * 0.015, (TEST_STOP - 1) * 0.015],
            "direct10_starts": starts,
            "closed_loop_blocks": CLOSED_LOOP_BLOCKS,
            "checkpoint": args.checkpoint,
            "baseline": "copy/persistence of the last input frame",
            "q_grid": [0.30, 1.50, 49],
            "radial_bands": 4,
            "note": "identical 5-channel samples for all cells; U-* strip conditions in-model",
        },
        "cells": dict(
            (k, dict((kk, str(vv) if isinstance(vv, Path) else vv)
                     for kk, vv in v.items()))
            for k, v in CELLS.items()),
        "results": results,
    }
    out = args.output_dir / ("factorial_evaluation_" + args.checkpoint + ".json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[written] " + str(out), flush=True)


if __name__ == "__main__":
    main()
