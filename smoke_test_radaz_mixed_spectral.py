#!/usr/bin/env python3
"""Audit mixed-manifest spectral losses and one GPU update before training."""

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


WEIGHTS = {
    "amplitude": 2.0e-4,
    "phase": 2.5e-6,
    "cross_phase": 4.0e-5,
}


def load_window(case: dict, start: int, frames: int) -> torch.Tensor:
    channels = [str(value) for value in case["channels"]]
    low = np.asarray(case["normalization_low"], dtype=np.float64)
    high = np.asarray(case["normalization_high"], dtype=np.float64)
    valid_height = 257
    valid_width = 256
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


def gradient_norm(loss: torch.Tensor, values: torch.Tensor, retain: bool) -> float:
    gradient = torch.autograd.grad(loss, values, retain_graph=retain)[0]
    return float(torch.linalg.vector_norm(gradient).detach().cpu())


def audit_case(
    spectral: PEPAPICSpectralLoss,
    case: dict,
    start: int,
    device: torch.device,
) -> dict:
    target = load_window(case, start, 10).unsqueeze(0).to(device)
    torch.manual_seed(42)
    prediction = (target + 0.01 * torch.randn_like(target)).requires_grad_(True)
    components = spectral(prediction, target)
    data_loss = F.mse_loss(prediction, target)
    gradients = {"data": gradient_norm(data_loss, prediction, True)}
    names = tuple(components)
    for index, name in enumerate(names):
        gradients[name] = gradient_norm(
            components[name], prediction, index < len(names) - 1
        )
    weighted_ratios = {
        name: WEIGHTS[name] * gradients[name] / gradients["data"]
        for name in names
    }
    values = {name: float(value.detach().cpu()) for name, value in components.items()}
    if not all(np.isfinite(value) and value > 0.0 for value in gradients.values()):
        raise ValueError(f"Invalid gradients for {case['case_key']}: {gradients}")
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"Invalid losses for {case['case_key']}: {values}")
    return {
        "case": case["case_key"],
        "losses": values,
        "gradient_norms": gradients,
        "weighted_gradient_over_data": weighted_ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start", type=int, default=1200)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spectral = PEPAPICSpectralLoss(args.manifest).to(device)
    audits = [
        audit_case(spectral, case, args.start, device)
        for case in manifest["cases"]
    ]

    sample = load_window(manifest["cases"][0], args.start, 20)
    inputs = sample[:10].unsqueeze(0).to(device)
    targets = sample[10:].unsqueeze(0).to(device)
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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    data_loss = F.mse_loss(predictions, targets)
    components = spectral(predictions, targets)
    total = data_loss + sum(WEIGHTS[name] * value for name, value in components.items())
    if not torch.isfinite(total):
        raise ValueError(f"Non-finite GPU smoke loss: {float(total)}")
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

    print(json.dumps({"manifest": str(args.manifest), "audits": audits}, indent=2))
    print(f"[PASS] output_shape={tuple(predictions.shape)}")
    print(f"[PASS] total_loss={float(total.detach().cpu()):.8g}")
    print(f"[PASS] step_sec={elapsed:.3f} peak_memory_mb={peak_mb:.1f}")
    print("PASS: mixed-manifest spectral loss is finite and differentiable")


if __name__ == "__main__":
    main()
