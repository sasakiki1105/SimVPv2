#!/usr/bin/env python3
"""Audit q/transport gradients and one FiLM-conditioned GPU update."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss
from openstl.models.simvp_model import SimVP_Model


TARGET_GRADIENT_RATIO = 0.075
E_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31


def raw_condition(case: dict) -> np.ndarray:
    electric_v_m = float(case["Ez_kVm"]) * 1.0e3
    magnetic_t = float(case["B_mT"]) * 1.0e-3
    ly_m = float(case.get("Ly_m", 1.28e-2))
    drift_velocity = electric_v_m / magnetic_t
    mode_n0 = (
        E_CHARGE * magnetic_t**2 * ly_m
        / (2.0 * np.pi * ELECTRON_MASS * electric_v_m)
    )
    return np.log(np.asarray([drift_velocity, mode_n0], dtype=np.float64))


def standardized_condition(manifest: dict, case: dict) -> np.ndarray:
    normalization = manifest["condition_normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    std = np.asarray(normalization["std"], dtype=np.float64)
    return ((raw_condition(case) - mean) / std).astype(np.float32)


def load_window(case: dict, start: int, frames: int) -> torch.Tensor:
    channels = [str(value) for value in case["channels"]]
    low = np.asarray(case["normalization_low"], dtype=np.float64)
    high = np.asarray(case["normalization_high"], dtype=np.float64)
    valid_height, valid_width = 257, 256
    model_height = int(case.get("model_height", 260))
    model_width = int(case.get("model_width", 256))
    output = np.empty(
        (frames, len(channels), model_height, model_width), dtype=np.float32
    )
    with h5py.File(case["path"], "r") as handle:
        for channel_index, channel in enumerate(channels):
            raw = np.asarray(
                handle[f"fields/{channel}"][
                    start : start + frames, :valid_height, :valid_width
                ],
                dtype=np.float64,
            )
            normalized = (raw - low[channel_index]) / (
                high[channel_index] - low[channel_index]
            )
            target = output[:, channel_index]
            target[:, :valid_height, :valid_width] = normalized.astype(np.float32)
            target[:, valid_height:, :valid_width] = target[
                :, valid_height - 1 : valid_height, :valid_width
            ]
    return torch.from_numpy(output)


def condition_channels(values: np.ndarray, frames: int, height: int, width: int):
    return torch.from_numpy(
        np.broadcast_to(
            values.reshape(1, -1, 1, 1),
            (frames, len(values), height, width),
        ).astype(np.float32, copy=True)
    )


def gradient_norm(loss: torch.Tensor, values: torch.Tensor, retain: bool) -> float:
    gradient = torch.autograd.grad(loss, values, retain_graph=retain)[0]
    return float(torch.linalg.vector_norm(gradient).detach().cpu())


def audit_case(loss_module, manifest, case, start, device):
    fields = load_window(case, start, 20)
    condition = standardized_condition(manifest, case)
    cond = condition_channels(condition, 10, fields.shape[-2], fields.shape[-1])
    batch_x = torch.cat((fields[:10], cond), dim=1).unsqueeze(0).to(device)
    target = fields[10:].unsqueeze(0).to(device)
    torch.manual_seed(42)
    prediction = (target + 0.01 * torch.randn_like(target)).requires_grad_(True)
    components = loss_module(prediction, target, batch_x=batch_x)
    data_loss = F.mse_loss(prediction, target)
    gradients = {"data": gradient_norm(data_loss, prediction, True)}
    names = tuple(components)
    for index, name in enumerate(names):
        gradients[name] = gradient_norm(
            components[name], prediction, index < len(names) - 1
        )
    if not all(np.isfinite(value) and value > 0.0 for value in gradients.values()):
        raise ValueError(f"Invalid gradients for {case['case_key']}: {gradients}")
    recommended = {
        name: TARGET_GRADIENT_RATIO * gradients["data"] / gradients[name]
        for name in names
    }
    return {
        "case": case["case_key"],
        "standardized_condition": condition.tolist(),
        "losses": {
            name: float(value.detach().cpu()) for name, value in components.items()
        },
        "gradient_norms": gradients,
        "recommended_lambda_at_7p5pct": recommended,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start", type=int, default=1200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_cases = [case for case in manifest["cases"] if case["role"] == "source"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loss_module = PEPAPICSpectralLoss(
        args.manifest,
        max_mode=64,
        radial_bands=4,
        radial_min_m=0.09e-2,
        radial_max_m=1.19e-2,
        coordinate_system="q_normalized",
        q_min=0.30,
        q_max=1.50,
        q_bins=49,
    ).to(device)
    audits = [
        audit_case(loss_module, manifest, case, args.start, device)
        for case in source_cases
    ]
    names = tuple(audits[0]["recommended_lambda_at_7p5pct"])
    recommended = {
        name: float(
            np.median(
                [audit["recommended_lambda_at_7p5pct"][name] for audit in audits]
            )
        )
        for name in names
    }

    case = source_cases[0]
    fields = load_window(case, args.start, 20)
    cond_values = standardized_condition(manifest, case)
    cond = condition_channels(cond_values, 10, fields.shape[-2], fields.shape[-1])
    inputs = torch.cat((fields[:10], cond), dim=1).unsqueeze(0).to(device)
    targets = fields[10:].unsqueeze(0).to(device)
    model = SimVP_Model(
        in_shape=tuple(inputs.shape[1:]),
        hid_S=64,
        hid_T=512,
        N_S=4,
        N_T=8,
        model_type="gSTA",
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        aft_seq_length=10,
        simvp_direct_aft_seq=True,
        out_channels=3,
        condition_dim=2,
        condition_film=True,
        condition_hidden_dim=64,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    data_loss = F.mse_loss(predictions, targets)
    components = loss_module(predictions, targets, batch_x=inputs)
    total = data_loss + sum(
        recommended[name] * value for name, value in components.items()
    )
    if not torch.isfinite(total):
        raise ValueError(f"Non-finite total loss: {float(total)}")
    total.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024.0**2
        if device.type == "cuda"
        else 0.0
    )

    payload = {
        "manifest": str(args.manifest),
        "target_gradient_ratio": TARGET_GRADIENT_RATIO,
        "audits": audits,
        "recommended_weights": recommended,
        "smoke": {
            "device": str(device),
            "output_shape": list(predictions.shape),
            "total_loss": float(total.detach().cpu()),
            "step_seconds": elapsed,
            "peak_memory_mb": peak_mb,
        },
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PASS: q-normalized loss and temporal FiLM are finite and differentiable")


if __name__ == "__main__":
    main()
