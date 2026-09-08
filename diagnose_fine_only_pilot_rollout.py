"""Locate the failure that aborted evaluate_radaz_e20_native_fine_only_pilot.py.

The pilot exited 1 with "No 27-30 us rollout frames".  Nothing printed an
"[A rollout]" line, so rollout() broke on its very first model call: the
first prediction contained a non-finite value, pieces stayed empty, and the
27-30 us window was therefore never reached.

This script re-runs that first call three ways -- fp16 autocast (what the
pilot used), plain fp32, and fp32 with the native-G2 history -- and reports
where the non-finite values appear.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

import evaluate_radaz_e20_native_fine_only_pilot as pilot

WORKDIR = Path(r"C:\Users\astro\research\SimVPv2\workdirs\radaz_e20_native_g2_fine_only_pilot")


class Args:
    pass


def describe(name, array):
    finite = np.isfinite(array)
    print("  %-26s finite=%-6s min=%-12.5g max=%-12.5g nan=%d inf=%d"
          % (name, bool(finite.all()), np.nanmin(array), np.nanmax(array),
             int(np.isnan(array).sum()), int(np.isinf(array).sum())))


def main():
    protocol = json.loads((WORKDIR / "protocol.json").read_text(encoding="utf-8"))
    paths = protocol["paths"]
    args = Args()
    args.config = pilot.CONFIG
    args.workdir = Path(protocol["checkpoint"]).parent.parent
    device = torch.device("cuda:0")

    manifest = Path(paths["manifest"])
    low, high, _ = pilot.load_normalization(manifest)
    print("normalization low =", low, " high =", high)

    fine, fine_time, fine_x, fine_y = pilot.load_fine(
        Path(paths["fine_h5"]), 24.0, 30.0, full_radial=True)
    native, native_time, native_x, native_y = pilot.load_native(
        Path(paths["native_h5"]), 24.0, 30.0)
    print("fine frames=%d  %.3f-%.3f us" % (len(fine), fine_time[0] * 1e6,
                                            fine_time[-1] * 1e6))
    print("steps requested by run_experiment_a = len(fine) - PRE =", len(fine) - pilot.PRE)

    native_physical = pilot.interpolate_native_full_to_model(
        native, native_x, native_y[:128], fine_x, fine_y)
    fine_norm = pilot.normalize_model(fine, low, high)
    native_norm = pilot.normalize_model(native_physical, low, high)
    describe("fine history (normalized)", fine_norm[:pilot.PRE])
    describe("native history (normalized)", native_norm[:pilot.PRE])

    model = pilot.build_model(args, device)

    for label, history, amp in (
        ("fine  / fp16 autocast (as run)", fine_norm[:pilot.PRE], True),
        ("fine  / fp32", fine_norm[:pilot.PRE], False),
        ("native/ fp16 autocast (as run)", native_norm[:pilot.PRE], True),
        ("native/ fp32", native_norm[:pilot.PRE], False),
    ):
        tensor = torch.from_numpy(np.ascontiguousarray(history)[None]).to(device)
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=amp):
                prediction = model(tensor)[0].float().cpu().numpy()
        print()
        print(label)
        describe("prediction step 1", prediction)

        # second step: feed the prediction back, which is what rollout does
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=amp):
                second = model(torch.from_numpy(prediction[-pilot.PRE:][None]
                                                ).to(device))[0].float().cpu().numpy()
        describe("prediction step 2", second)

    # how far does an fp32 rollout actually get before diverging?
    print()
    print("fp32 free rollout from the fine history, first 60 steps:")
    current = np.array(fine_norm[:pilot.PRE], dtype=np.float32)
    produced = 0
    for block in range(6):
        with torch.inference_mode():
            out = model(torch.from_numpy(current[None]).to(device))[0].float().cpu().numpy()
        if not np.all(np.isfinite(out)):
            print("  block %d: NON-FINITE (nan=%d inf=%d)"
                  % (block, int(np.isnan(out).sum()), int(np.isinf(out).sum())))
            break
        produced += len(out)
        print("  block %d: ok, frames=%d, range %.4g .. %.4g"
              % (block, produced, out.min(), out.max()))
        current = out[-pilot.PRE:]


if __name__ == "__main__":
    main()
