#!/usr/bin/env python3
"""Run one forward/backward update for the 128x128 RadAz direct model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from openstl.models.simvp_model import SimVP_Model
from openstl.datasets.dataloader_pepapic_h5 import _build_disjoint_starts
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-resolution", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--spectral", action="store_true")
    parser.add_argument("--spectral-amplitude-lambda", type=float, default=5.0e-4)
    parser.add_argument("--spectral-phase-lambda", type=float, default=5.0e-6)
    parser.add_argument("--spectral-cross-lambda", type=float, default=1.0e-4)
    args = parser.parse_args()

    with h5py.File(args.h5_path, "r") as handle:
        shape = tuple(handle["data_tchw"].shape)
        sample = np.asarray(
            handle["data_tchw"][0 : 20 + args.batch_size - 1], dtype=np.float32
        )
    if shape[0:2] != (2001, 3):
        raise ValueError(f"Unexpected training data shape: {shape}")
    model_height = args.model_resolution or shape[2]
    model_width = args.model_resolution or shape[3]
    split = _build_disjoint_starts(2001, 20)
    info = split[-1]
    expected = {
        "train_frame_range": (0, 1599),
        "val_frame_range": (1600, 1799),
        "test_frame_range": (1800, 2000),
        "n_train": 1581,
        "n_val": 181,
        "n_test": 182,
    }
    for key, value in expected.items():
        if info[key] != value:
            raise ValueError(f"Split mismatch for {key}: {info[key]} != {value}")

    device = torch.device(args.device)
    model = SimVP_Model(
        in_shape=(10, 3, model_height, model_width),
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
    spectral_loss = (
        PEPAPICSpectralLoss(args.h5_path).to(device) if args.spectral else None
    )
    input_windows = np.stack(
        [sample[start : start + 10] for start in range(args.batch_size)]
    )
    target_windows = np.stack(
        [sample[start + 10 : start + 20] for start in range(args.batch_size)]
    )
    inputs = torch.from_numpy(input_windows).to(device)
    targets = torch.from_numpy(target_windows).to(device)
    if (model_height, model_width) != tuple(inputs.shape[-2:]):
        batch, length, channels, height, width = inputs.shape
        inputs = F.interpolate(
            inputs.reshape(batch * length, channels, height, width),
            size=(model_height, model_width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, length, channels, model_height, model_width)
        targets = F.interpolate(
            targets.reshape(batch * length, channels, height, width),
            size=(model_height, model_width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, length, channels, model_height, model_width)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    predictions = None
    loss = None
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Direct output shape mismatch: {tuple(predictions.shape)} != {tuple(targets.shape)}"
            )
        data_loss = torch.mean((predictions - targets) ** 2)
        loss = data_loss
        spectral_components = None
        if spectral_loss is not None:
            spectral_components = spectral_loss(predictions, targets)
            loss = (
                loss
                + args.spectral_amplitude_lambda
                * spectral_components["amplitude"]
                + args.spectral_phase_lambda * spectral_components["phase"]
                + args.spectral_cross_lambda
                * spectral_components["cross_phase"]
            )
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite smoke loss: {loss}")
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    print(f"[PASS] shape={shape}")
    print(f"[PASS] split={info}")
    print(f"[PASS] direct_output={tuple(predictions.shape)}")
    print(f"[PASS] data_loss={float(data_loss):.8g}")
    if spectral_components is not None:
        for name, value in spectral_components.items():
            print(f"[PASS] spectral_{name}_loss={float(value):.8g}")
    print(f"[PASS] one_step_loss={float(loss):.8g}")
    print(f"[PASS] peak_cuda_memory_mb={peak_memory_mb:.1f}")
    print(f"[PASS] forward_backward_step_sec={elapsed / args.steps:.4f}")
    if device.type == "cuda":
        total_memory_mb = torch.cuda.get_device_properties(device).total_memory / (1024.0**2)
        print(f"[PASS] total_cuda_memory_mb={total_memory_mb:.1f}")


if __name__ == "__main__":
    main()
