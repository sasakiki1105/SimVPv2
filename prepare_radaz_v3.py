"""Prepare corrected loss calibration and a four-cell research plan; no training.

All calibration uses source TRAIN frames. Fixed Gaussian perturbation output-
gradient ratios set loss weights; these are not parameter-gradient shares or
evidence that the weights stay balanced during optimization.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import MANIFEST, load_case_frames
from evaluate_radaz_corrected_A import validate_source_window, digest
from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "workdirs/2D_RadAz/radaz_arch_v3_plan"
STARTS = (1200, 1300, 1400, 1500)
SEEDS = (42, 43, 44)
CELLS = {"A": (4, 64), "B": (2, 64), "C": (4, 128), "D": (2, 128)}


def calibrate():
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    low, high = (np.asarray(manifest["normalization"][k], dtype=float) for k in ("low", "high"))
    lm = PEPAPICSpectralLoss(str(MANIFEST), max_mode=64, coordinate_system="integer_power_cross",
                            radial_reduction="local_product").to(dev)
    rows = []
    for case in manifest["cases"]:
        if case["role"] != "source":
            continue
        for start in STARTS:
            validate_source_window(case, manifest, start, start+20, "calibration")
            frames = load_case_frames(case, low, high, start, start+20)[0]
            truth = torch.from_numpy(frames[10:20][None]).to(dev)
            generator = torch.Generator().manual_seed(0)
            prediction = (truth + .05*torch.randn(truth.shape, generator=generator)).requires_grad_()
            data = (prediction-truth).square().mean()
            terms = lm(prediction, truth)
            gradients = {}
            losses = {"data": data, "power": terms["power"], "crossspec": terms["cross_spectrum"]}
            for name, loss in losses.items():
                grad = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
                if not torch.isfinite(grad).all():
                    raise RuntimeError("Nonfinite calibration gradient")
                gradients[name] = float(torch.linalg.vector_norm(grad))
            rows.append({"case": case["case_key"], "input_start": start, "target_frames": [start+10, start+19],
                         "normalized_frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
                         "losses": {k: float(v.detach()) for k, v in losses.items()},
                         "output_gradient_norms": gradients,
                         "power_ratio": gradients["power"]/gradients["data"],
                         "crossspec_ratio": gradients["crossspec"]/gradients["data"]})
            del prediction, truth, terms, losses, grad, frames
        print("TRAIN-only gradient calibration:", case["case_key"], flush=True)
    weights = {k: float(.05/np.median([r[k+"_ratio"] for r in rows])) for k in ("power", "crossspec")}
    for r in rows:
        r["weighted_output_gradient_ratios"] = {k: r[k+"_ratio"]*weights[k] for k in weights}
    return {"radial_reduction": "local_product", "role": "source_train_only",
            "starts": STARTS, "noise_std_normalized": .05, "noise_seed": 0,
            "target_median_output_gradient_ratio": .05, "weights": weights, "windows": rows,
            "scope": "synthetic perturbed targets; NOT actual parameter-gradient contributions during training"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "bundle.json").exists():
        raise SystemExit("Prepared bundle exists; do not overwrite a frozen experiment plan")
    calibration = calibrate()
    (OUT / "gradient_calibration_train.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    # A clean config: old v2 comments/thresholds are not inherited.
    base = dict(method="SimVP", model_type="gSTA", hid_T=256, N_T=4, N_S=4,
                spatio_kernel_enc=3, spatio_kernel_dec=3, simvp_direct_aft_seq=True,
                out_channels=3, condition_dim=2, condition_film=False, condition_hidden_dim=64,
                translator_norm="group", translator_norm_groups=8,
                lr=1e-3, batch_size=1, drop_path=0, drop_path_schedule="zero_to_max", sched="onecycle", epoch=60,
                pre_seq_length=10, aft_seq_length=10, in_shape=None,
                pepapic_condition_channels="log_vE,log_n0",
                pepapic_spectral_loss="integer_power_cross", pepapic_spectral_coordinate_system="integer_power_cross",
                pepapic_spectral_radial_reduction="local_product", pepapic_spectral_max_mode=64,
                pepapic_spectral_radial_bands=4, pepapic_spectral_radial_min_m=.0009, pepapic_spectral_radial_max_m=.0119,
                pepapic_spectral_power_eps_relative=1e-8, pepapic_spectral_cross_mask_kappa=1e-3,
                pepapic_spectral_power_lambda=calibration["weights"]["power"],
                pepapic_spectral_crossspec_lambda=calibration["weights"]["crossspec"],
                pepapic_spectral_complex_lambda=0.0, pepapic_transport_lambda=0.0,
                radaz_validation_diagnostics=True, metrics=["mse", "mae"],
                snapshot_epochs="5,10,15,20,30,40,50,60", snapshot_epoch_numbering="completed")
    configs, commands = {}, []
    for cell, (downsample, channels) in CELLS.items():
        cfg = dict(base, spatio_azimuth_downsample=downsample, hid_S=channels)
        path = ROOT / f"configs/custom/pepapic/SimVP_gSTA_radaz_v3_{cell}_60ep.py"
        if path.exists():
            raise FileExistsError(path)
        path.write_text("# Corrected 2x2 pilot; primary checkpoint is terminal epoch index 59.\n"
                        "# Loss weights from SOURCE-TRAIN synthetic OUTPUT-gradient audit, not parameter gradients.\n"
                        + "\n".join(k+" = "+repr(v) for k, v in cfg.items())+"\n", encoding="utf-8")
        configs[cell] = str(path.relative_to(ROOT))
        for seed in SEEDS:
            commands.append(["python", "tools/train_only.py", "--dataname", "pepapic_h5",
                "--config_file", configs[cell], "--data_root", str(MANIFEST),
                "--res_dir", str(ROOT / "workdirs/2D_RadAz"), "--ex_name", f"radaz_v3_{cell}_seed{seed}_60ep",
                "--method", "simvp", "--pre_seq_length", "10", "--aft_seq_length", "10",
                "--total_length", "20", "--epoch", "60", "--batch_size", "1", "--val_batch_size", "1",
                "--num_workers", "0", "--gpus", "0", "--seed", str(seed), "--no_display_method_info"])
    bundle_files = [ROOT / p for p in configs.values()] + [Path(__file__), MANIFEST,
        OUT / "gradient_calibration_train.json", ROOT / "radaz_metrics_v3.py", ROOT / "evaluate_radaz_corrected_A.py",
        ROOT / "evaluate_radaz_conditioned_factorial.py", ROOT / "openstl/models/simvp_factory.py",
        ROOT / "openstl/models/simvp_model.py", ROOT / "openstl/modules/simvp_modules.py",
        ROOT / "openstl/methods/simvp.py", ROOT / "openstl/methods/pepapic_spectral_loss.py",
        ROOT / "openstl/methods/radaz_validation.py", ROOT / "openstl/utils/callbacks.py",
        ROOT / "openstl/api/exp.py", ROOT / "openstl/datasets/dataloader_pepapic_h5.py"]
    protocol_path = ROOT / "RADAZ_V3_PROTOCOL.md"
    if not protocol_path.exists():
        raise FileNotFoundError("Write the research protocol before freezing the bundle")
    bundle_files.append(protocol_path)
    bundle = {"status": "prepared_not_trained", "purpose": "exploratory factorial pilot on existing trajectories",
        "seeds": SEEDS, "factors": {"azimuth_latent_width": [64, 128], "encoder_channels": [64, 128]},
        "cells": CELLS, "configs": configs, "translator_width": 256, "translator_depth": 4,
        "normalization": "GroupNorm 8 groups, same training/evaluation, all cells",
        "primary_checkpoint": "last.ckpt, require epoch index 59 in all 12 runs",
        "checkpoint_selection": "fixed completed epoch 60; best.ckpt and trajectory are secondary only",
        "old_A_thresholds": "not transferable to changed normalization, radial product or O aggregation",
        "confirmation": "requires independent PIC realizations and untouched condition/time evaluation; old holdouts are development data",
        "sha256": {str(p.relative_to(ROOT)): digest(p) for p in bundle_files},
        "commands": commands, "commands_executed": False}
    (OUT / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print("Prepared", OUT / "bundle.json", flush=True)


if __name__ == "__main__":
    main()
