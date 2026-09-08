"""Paired real-train-window loss/gradient check; no optimizer or checkpoint."""
import json
import os
import runpy
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
from evaluate_radaz_conditioned_factorial import condition_vector, load_case_frames
from openstl.methods.simvp import SimVP
from run_radaz_paired_pilot import ROOT, OUT, verify_bundle


def main():
    bundle = verify_bundle()
    torch.set_num_threads(4)
    manifest = json.loads(Path(bundle["manifest"]).read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    low, high = (np.asarray(manifest["normalization"][k]) for k in ("low", "high"))
    frames = load_case_frames(case, low, high, 1400, 1420)[0]
    inputs = np.empty((1, 10, 5, 260, 256), dtype=np.float32)
    inputs[0, :, :3] = frames[:10]
    inputs[0, :, 3:] = condition_vector(case, manifest)[0][None, :, None, None]
    device = torch.device("cuda:0")
    x, target = torch.from_numpy(inputs).to(device), torch.from_numpy(frames[10:][None]).to(device)
    reference, rows = None, {}
    for cell in ("D", "P"):
        cfg = {k:v for k,v in runpy.run_path(str(ROOT / bundle["configs"][cell])).items() if not k.startswith("__")}
        cfg.update(in_shape=(10,5,260,256), dataname="pepapic_h5", data_root=bundle["manifest"])
        torch.manual_seed(42)
        model = SimVP(**cfg).to(device).train()
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.model.hid.modules())
        prediction = model(x)
        loss, data, _, _, auxiliary = model._total_loss(prediction, target, batch_x=x)
        if cell == "D":
            torch.testing.assert_close(loss, data, rtol=0, atol=0)
            assert auxiliary is None
            reference = prediction.detach().cpu()
        else:
            torch.testing.assert_close(prediction.detach().cpu(), reference, rtol=0, atol=0)
            assert loss > data and auxiliary is not None
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        with torch.no_grad():
            torch.testing.assert_close(prediction.detach(), model.eval()(x), rtol=1e-5, atol=1e-6)
        rows[cell] = dict(total_loss=float(loss.detach()), data_loss=float(data.detach()),
            spectral_losses={k:float(v.detach()) for k,v in (auxiliary or {}).items()},
            finite_parameter_gradients=True, train_eval_predictions_match=True)
        print(cell, rows[cell], flush=True)
        del model, prediction, loss, data, auxiliary, gradients
    report = dict(status="smoke_only_no_optimizer_no_checkpoint", source_train_frames=[1400,1419],
                  initial_predictions_identical=True, cells=rows)
    (OUT / "training_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
