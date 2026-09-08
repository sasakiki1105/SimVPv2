"""One train-window forward/backward for each corrected cell; no weight updates."""
import json
import os
import runpy
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
from evaluate_radaz_conditioned_factorial import MANIFEST, condition_vector, load_case_frames
from openstl.methods.simvp import SimVP


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(MANIFEST.read_text())
    case = next(c for c in manifest["cases"] if c["role"] == "source")
    low, high = (np.asarray(manifest["normalization"][k]) for k in ("low", "high"))
    frames = load_case_frames(case, low, high, 1400, 1420)[0]
    inputs = np.empty((1, 10, 5, 260, 256), dtype=np.float32)
    inputs[0, :, :3] = frames[:10]
    inputs[0, :, 3:] = condition_vector(case, manifest)[0][None, :, None, None]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, truth = torch.from_numpy(inputs).to(dev), torch.from_numpy(frames[10:][None]).to(dev)
    rows = {}
    for cell in "ABCD":
        cfg = {k: v for k, v in runpy.run_path(str(root / f"configs/custom/pepapic/SimVP_gSTA_radaz_v3_{cell}_60ep.py")).items()
               if not k.startswith("__")}
        cfg.update(in_shape=(10, 5, 260, 256), dataname="pepapic_h5", data_root=str(MANIFEST))
        model = SimVP(**cfg).to(dev).train()
        if any(isinstance(m, torch.nn.BatchNorm2d) for m in model.model.hid.modules()):
            raise AssertionError("BatchNorm survived in corrected translator")
        pred = model(x)
        loss, data, _, _, auxiliary = model._total_loss(pred, truth, batch_x=x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError("Missing/nonfinite training gradients")
        with torch.no_grad():
            evaluated = model.eval()(x)
            torch.testing.assert_close(pred.detach(), evaluated, rtol=1e-5, atol=1e-6)
        rows[cell] = {"shape": list(pred.shape), "parameters": sum(p.numel() for p in model.parameters()),
                      "total_loss": float(loss.detach()), "data_loss": float(data.detach()),
                      "spectral_losses": {k: float(v.detach()) for k, v in auxiliary.items()},
                      "finite_parameter_gradients": True, "train_eval_predictions_match": True}
        print(cell, rows[cell], flush=True)
        del model, pred, evaluated, loss, data, auxiliary, grads
    output = root / "workdirs/2D_RadAz/radaz_arch_v3_plan/training_smoke.json"
    output.write_text(json.dumps({"status": "smoke_only_no_optimizer_no_checkpoint",
        "case": case["case_key"], "source_train_frames": [1400, 1419], "cells": rows}, indent=2))


if __name__ == "__main__":
    main()
