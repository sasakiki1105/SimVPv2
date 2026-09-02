#!/usr/bin/env python3
"""Check a raw RadAz manifest with one SimVPv2 forward/backward update."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from openstl.datasets.dataloader_pepapic_h5 import _PEPAPICMultiCaseWindows
from openstl.models.simvp_model import SimVP_Model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = _PEPAPICMultiCaseWindows(
        str(args.manifest),
        pre_seq_length=10,
        aft_seq_length=10,
        split="val",
    )
    if dataset.C != 3 or (dataset.H, dataset.W) != (260, 256):
        raise ValueError(
            f"Unexpected manifest shape C,H,W={(dataset.C, dataset.H, dataset.W)}"
        )
    if len(dataset.case_info) != 2:
        raise ValueError(f"Expected two source cases, got {len(dataset.case_info)}")
    x_numpy, y_numpy = dataset[0]
    if x_numpy.shape != (10, 3, 260, 256):
        raise ValueError(f"Unexpected input shape: {x_numpy.shape}")
    if y_numpy.shape != x_numpy.shape:
        raise ValueError(f"Input/target shape mismatch: {x_numpy.shape}, {y_numpy.shape}")
    if not all(bool(item["n_samples"] == 181) for item in dataset.case_info):
        raise ValueError(f"Unexpected validation split: {dataset.case_info}")

    device = torch.device(args.device)
    model = SimVP_Model(
        in_shape=(10, 3, 260, 256),
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
    x = torch.from_numpy(x_numpy[None]).to(device)
    y = torch.from_numpy(y_numpy[None]).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    loss = torch.mean((prediction - y) ** 2)
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite smoke loss: {loss}")
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    print(f"[PASS] manifest={args.manifest}")
    print(f"[PASS] cases={dataset.case_info}")
    print(f"[PASS] input_range={float(x.min()):.6g}..{float(x.max()):.6g}")
    print(f"[PASS] loss={float(loss):.8g}")
    print(f"[PASS] elapsed_sec={elapsed:.3f} peak_cuda_memory_mb={peak_mb:.1f}")
    del prediction, loss, x, y, model, optimizer, dataset
    gc.collect()


if __name__ == "__main__":
    main()
