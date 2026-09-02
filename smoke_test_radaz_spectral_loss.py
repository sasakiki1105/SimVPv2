import argparse
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

from openstl.methods.pepapic_spectral_loss import PEPAPICSpectralLoss


ROOT = Path(__file__).resolve().parent
DEFAULT_H5 = (
    ROOT.parent
    / "PEPAPIC"
    / "test"
    / "results"
    / "2D_Landmark"
    / "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
    / "2D_RadAz_Xe1p_Bx20mT_Ez10kVm_dt15ps_out15ns"
    / "SimVPv2_inputs"
    / "radaz_3ch_trainfixed_margin20_native257x256_pad260x256.h5"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check RadAz spectral-loss selectivity and gradients."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_H5)
    parser.add_argument("--start", type=int, default=1601)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.01)
    return parser.parse_args()


def load_frames(path, start, frames):
    with h5py.File(path, "r") as handle:
        data_key = "data_tchw" if "data_tchw" in handle else "data"
        end = min(start + frames, int(handle[data_key].shape[0]))
        if end - start < 2:
            raise ValueError(
                f"Need at least two frames from start={start}; data has "
                f"{handle[data_key].shape[0]} frames"
            )
        values = handle[data_key][start:end]
    return torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)


def scalar_values(losses):
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def gradient_norm(loss, values, retain_graph):
    gradient = torch.autograd.grad(
        loss, values, retain_graph=retain_graph, allow_unused=False
    )[0]
    return float(torch.linalg.vector_norm(gradient).detach().cpu())


def main():
    args = parse_args()
    torch.manual_seed(42)
    target = load_frames(args.data_root, args.start, args.frames)
    spectral = PEPAPICSpectralLoss(args.data_root)

    identity = scalar_values(spectral(target, target))

    amplitude_pred = target.clone()
    phi_index = spectral.phi_index
    phi_mean = amplitude_pred[:, :, phi_index].mean(dim=-1, keepdim=True)
    amplitude_pred[:, :, phi_index] = (
        phi_mean + 1.15 * (amplitude_pred[:, :, phi_index] - phi_mean)
    )
    amplitude = scalar_values(spectral(amplitude_pred, target))

    shifted_frames = []
    for time_index in range(target.shape[1]):
        shifted_frames.append(
            torch.roll(target[:, time_index], shifts=time_index, dims=-1)
        )
    phase_pred = torch.stack(shifted_frames, dim=1)
    phase = scalar_values(spectral(phase_pred, target))

    cross_pred = target.clone()
    cross_pred[:, :, spectral.electron_index] = torch.roll(
        cross_pred[:, :, spectral.electron_index], shifts=7, dims=-1
    )
    cross = scalar_values(spectral(cross_pred, target))

    noisy = (
        target + args.noise_std * torch.randn_like(target)
    ).requires_grad_(True)
    noisy_losses = spectral(noisy, target)
    data_loss = F.mse_loss(noisy, target)
    grad_norms = {
        "data": gradient_norm(data_loss, noisy, retain_graph=True),
        "amplitude": gradient_norm(
            noisy_losses["amplitude"], noisy, retain_graph=True
        ),
        "phase": gradient_norm(noisy_losses["phase"], noisy, retain_graph=True),
        "cross_phase": gradient_norm(
            noisy_losses["cross_phase"], noisy, retain_graph=False
        ),
    }
    target_fraction = 0.05
    lambda_proxies = {
        name: (
            target_fraction * grad_norms["data"] / value
            if value > 0.0
            else float("inf")
        )
        for name, value in grad_norms.items()
        if name != "data"
    }

    print(f"data_root={args.data_root}")
    print(f"target_shape={tuple(target.shape)}")
    print(f"valid_shape={(spectral.valid_height, spectral.valid_width)}")
    print(f"radial_bands={spectral.radial_bands} max_mode={spectral.max_mode}")
    print(f"mode_weight_sum={float(spectral.mode_weights.sum()):.8f}")
    print(f"identity={identity}")
    print(f"amplitude_perturbation={amplitude}")
    print(f"time_dependent_roll={phase}")
    print(f"electron_only_roll={cross}")
    print(f"noisy_losses={scalar_values(noisy_losses)}")
    print(f"gradient_norms={grad_norms}")
    print(f"lambda_proxies_for_5pct_output_gradient={lambda_proxies}")

    tolerance = 2e-5
    if any(value > tolerance for value in identity.values()):
        raise AssertionError(f"Identity losses should be near zero: {identity}")
    if not amplitude["amplitude"] > identity["amplitude"] + tolerance:
        raise AssertionError("Amplitude perturbation was not detected")
    if not phase["phase"] > identity["phase"] + tolerance:
        raise AssertionError("Time-dependent azimuthal roll was not detected")
    if not cross["cross_phase"] > identity["cross_phase"] + tolerance:
        raise AssertionError("Electron/E_y cross-phase perturbation was not detected")
    if not all(torch.isfinite(value) for value in noisy_losses.values()):
        raise AssertionError(f"Non-finite spectral loss: {scalar_values(noisy_losses)}")
    if not all(value > 0.0 for value in grad_norms.values()):
        raise AssertionError(f"Expected non-zero gradients: {grad_norms}")
    print("PASS: RadAz spectral losses are finite, selective, and differentiable")


if __name__ == "__main__":
    main()
