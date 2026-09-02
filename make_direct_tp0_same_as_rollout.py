"""Build a teacher-forced TP1 sequence aligned with the TP0 rollout timeline."""

from pathlib import Path
import argparse
import runpy

import numpy as np
import torch


def build_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    from openstl.models.simvp_model import SimVP_Model

    cfg = runpy.run_path(str(config_path))
    model = SimVP_Model(
        in_shape=(10, 3, 100, 100),
        hid_S=cfg.get("hid_S", 64),
        hid_T=cfg.get("hid_T", 512),
        N_S=cfg.get("N_S", 4),
        N_T=cfg.get("N_T", 8),
        model_type=cfg.get("model_type", "gSTA"),
        drop_path=float(cfg.get("drop_path", 0.0)),
        spatio_kernel_enc=int(cfg.get("spatio_kernel_enc", 3)),
        spatio_kernel_dec=int(cfg.get("spatio_kernel_dec", 3)),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    state = {
        (key.removeprefix("model.") if key.startswith("model.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/custom/pepapic/SimVP_gSTA_pepapic.py"),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    rollout_dir = args.workdir / "rollout_tp0_quads_assets"
    saved_dir = args.workdir / "saved"
    args.outdir.mkdir(parents=True, exist_ok=True)

    rollout_inputs = np.load(rollout_dir / "inputs_roll.npy", mmap_mode="r")
    rollout_trues = np.load(rollout_dir / "trues_roll.npy", mmap_mode="r")
    direct_preds = np.load(saved_dir / "preds.npy", mmap_mode="r")

    # Truth timeline = validation tail seed followed by the complete test truth.
    seed = np.asarray(rollout_inputs[0], dtype=np.float32)
    test_truth = np.asarray(rollout_trues[:, 0], dtype=np.float32)
    timeline = np.concatenate([seed, test_truth], axis=0)
    count = test_truth.shape[0]
    inputs = np.stack([timeline[k:k + 10] for k in range(count)])
    trues = test_truth[:, None]
    preds = np.empty_like(trues)

    # Existing direct-test TP1 predictions cover rollout indices 10..391 exactly.
    preds[10:10 + direct_preds.shape[0], 0] = direct_preds[:, 0]
    missing = list(range(10)) + list(range(10 + direct_preds.shape[0], count))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.config, args.workdir / "checkpoints/best.ckpt", device)
    with torch.no_grad():
        for k in missing:
            output = model(torch.from_numpy(inputs[k:k + 1]).to(device))
            if isinstance(output, (tuple, list)):
                output = output[0]
            preds[k, 0] = output[0, 0].detach().cpu().numpy()
            print(f"inferred missing aligned frame {k + 1}/{count}")

    np.save(args.outdir / "inputs.npy", inputs)
    np.save(args.outdir / "preds.npy", preds)
    np.save(args.outdir / "trues.npy", trues)
    print("saved", args.outdir, inputs.shape, preds.shape, trues.shape)


if __name__ == "__main__":
    main()
