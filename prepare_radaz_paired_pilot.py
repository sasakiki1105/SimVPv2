"""Freeze the authorized D/P pilot without starting research training."""
import copy
import json
import os
import runpy
import shutil
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from evaluate_radaz_corrected_A import digest
from evaluate_radaz_conditioned_factorial import MANIFEST
from openstl.models.simvp_factory import build_simvp_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "workdirs/2D_RadAz/radaz_paired_pilot_v2"


def source_manifest(parent):
    manifest = copy.deepcopy(parent)
    manifest["cases"] = [c for c in manifest["cases"] if c["role"] == "source"]
    # BaseExperiment constructs three loaders even for train_only.py. The old
    # factorial manifest reserved test membership for condition holdouts.
    # Register existing source temporal test segments; never reassign frames.
    for case in manifest["cases"]:
        case["splits"] = ["train", "val", "test"]
    manifest["name"] = "radaz_paired_pilot_source_only"
    manifest["description"] = "Source-only D/P pilot; inherited source-train normalization. Old development source-test is not confirmatory."
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bundle.json"
    if path.exists():
        raise SystemExit("Frozen pilot exists; refusing overwrite")
    manifest = source_manifest(json.loads(MANIFEST.read_text(encoding="utf-8")))
    subset = OUT / "source_manifest.json"
    subset.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    base = {k:v for k,v in runpy.run_path(str(ROOT / "configs/custom/pepapic/SimVP_gSTA_radaz_v3_A_60ep.py")).items() if not k.startswith("__")}
    configs, commands, signatures = {}, {}, {}
    torch.set_num_threads(4)
    reference = None
    for cell in ("D", "P"):
        cfg = copy.deepcopy(base)
        if cell == "D":
            cfg.update(pepapic_spectral_loss="none", pepapic_spectral_power_lambda=0., pepapic_spectral_crossspec_lambda=0.)
        config = ROOT / f"configs/custom/pepapic/SimVP_gSTA_radaz_pilot_{cell}_60ep.py"
        config_text = "# A-fixed D/P source-only pilot, terminal index 59, seed 42.\n"+"\n".join(k+" = "+repr(v) for k,v in cfg.items())+"\n"
        if config.exists() and config.read_text(encoding="utf-8") != config_text:
            raise RuntimeError("Existing config differs; refusing overwrite: " + str(config))
        if not config.exists():
            config.write_text(config_text, encoding="utf-8")
        configs[cell] = str(config.relative_to(ROOT))
        torch.manual_seed(42)
        m = build_simvp_model(dict(cfg, in_shape=(10,5,260,256))).eval()
        state = m.state_dict()
        if reference is None:
            reference = {k:v.clone() for k,v in state.items()}
        else:
            for k,v in state.items():
                torch.testing.assert_close(v, reference[k], rtol=0, atol=0)
        signatures[cell] = sum(p.numel() for p in m.parameters())
        del m
        commands[cell] = ["tools/train_only.py", "--dataname", "pepapic_h5", "--config_file", str(config),
            "--data_root", str(subset), "--res_dir", str(ROOT / "workdirs/2D_RadAz"),
            "--ex_name", f"radaz_pilot_{cell}_GN_A_seed42_60ep", "--method", "simvp",
            "--pre_seq_length", "10", "--aft_seq_length", "10", "--total_length", "20",
            "--epoch", "60", "--batch_size", "1", "--val_batch_size", "1", "--num_workers", "0",
            "--gpus", "0", "--seed", "42", "--drop_path", "0", "--no_display_method_info"]
    files = [Path(__file__), ROOT / "RADAZ_PAIRED_PILOT_PROTOCOL.md", subset,
        ROOT / "run_radaz_paired_pilot.py", ROOT / "evaluate_radaz_paired_pilot.py",
        ROOT / "analyze_radaz_nextstep_baselines.py", ROOT / "radaz_metrics_v3.py",
        ROOT / "evaluate_radaz_conditioned_factorial.py", ROOT / "openstl/models/simvp_factory.py",
        ROOT / "openstl/models/simvp_model.py", ROOT / "openstl/modules/simvp_modules.py",
        ROOT / "openstl/methods/simvp.py", ROOT / "openstl/methods/pepapic_spectral_loss.py",
        ROOT / "openstl/methods/radaz_validation.py", ROOT / "openstl/datasets/dataloader_pepapic_h5.py",
        ROOT / "openstl/utils/callbacks.py", ROOT / "openstl/api/exp.py", ROOT / "tools/train_only.py",
        ROOT / "evaluate_radaz_corrected_A.py",
        ROOT / "tests/test_radaz_paired_pilot.py", ROOT / "tests/smoke_radaz_paired_pilot.py",
        *[ROOT / p for p in configs.values()]]
    # Include parser, scheduler and inherited method behavior, not only the edited modules.
    files = sorted(set(files + list((ROOT / "openstl").rglob("*.py"))))
    baseline_dir = ROOT / "workdirs/2D_RadAz/radaz_nextstep_baselines_v1"
    files += [baseline_dir / "results.json", baseline_dir / "protocol.json",
              *sorted(baseline_dir.glob("*_predictions.npz"))]
    bundle = {"status": "frozen_before_training", "seed": 42, "epochs": 60,
        "version": 2, "supersedes": "radaz_paired_pilot_v1: loader construction failed before model initialization/optimizer steps",
        "amendment": "Explicit source temporal test membership required by existing three-loader API; model, loss, train/val frames unchanged",
        "primary_checkpoint": "last.ckpt index 59", "initial_model_states_identical": True,
        "parameters": signatures, "configs": configs, "commands": commands,
        "manifest": str(subset), "parent_manifest_sha256": digest(MANIFEST),
        "later_unscheduled_seeds": [43,44], "source_fields": {},
        "sha256": {str(f.relative_to(ROOT)): digest(f) for f in files}}
    for c in manifest["cases"]:
        stat = Path(c["path"]).stat()
        bundle["source_fields"][c["case_key"]] = {"path": c["path"], "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    for source in files:
        if source.suffix in (".py", ".md", ".json"):
            destination = OUT / "executed_source" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    print("Frozen", path)


if __name__ == "__main__":
    main()
