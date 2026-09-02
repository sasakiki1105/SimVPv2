import json
from pathlib import Path

import h5py
import numpy as np
import torch
from openstl.models import SimVP_Model
from .base_method import Base_method
from .pepapic_spectral_loss import PEPAPICSpectralLoss


EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19


class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)
        self._setup_pepapic_poisson_loss()
        self._setup_pepapic_spectral_loss()

    def _build_model(self, **args):
        return SimVP_Model(**args)

    def _setup_pepapic_poisson_loss(self):
        mode = str(getattr(self.hparams, "pepapic_poisson_loss", "none")).lower()
        weight = float(getattr(self.hparams, "pepapic_poisson_lambda", 0.0))
        efield_mode = str(getattr(self.hparams, "pepapic_efield_loss", "none")).lower()
        efield_weight = float(getattr(self.hparams, "pepapic_efield_lambda", 0.0))
        self.pepapic_poisson_loss_mode = mode
        self.pepapic_poisson_loss_weight = weight
        self.pepapic_efield_loss_mode = efield_mode
        self.pepapic_efield_loss_weight = efield_weight
        self.pepapic_poisson_enabled = not (
            mode in ("none", "off", "false", "0") or weight <= 0.0
        )
        self.pepapic_efield_enabled = not (
            efield_mode in ("none", "off", "false", "0") or efield_weight <= 0.0
        )
        if not self.pepapic_poisson_enabled and not self.pepapic_efield_enabled:
            return

        data_root = Path(str(self.hparams.data_root))
        with h5py.File(data_root, "r") as f:
            props = f["props"][()] if "props" in f else [b"electron_den", b"ion_den", b"phi"]
            props = [p.decode() if isinstance(p, (bytes, bytearray)) else str(p) for p in props]
            train_min = np.asarray(f["train_min"][()], dtype=np.float32)
            train_max = np.asarray(f["train_max"][()], dtype=np.float32)
            margin = float(f["margin"][()]) if "margin" in f else 0.0

        value_range = train_max - train_min
        lo = train_min - margin * value_range
        hi = train_max + margin * value_range
        scale = hi - lo

        for required in ("electron_den", "ion_den", "phi"):
            if required not in props:
                raise ValueError(f"pepapic_poisson_loss requires channel {required}, got {props}")
        self.pepapic_electron_index = props.index("electron_den")
        self.pepapic_ion_index = props.index("ion_den")
        self.pepapic_phi_index = props.index("phi")

        dx, dy = self._load_pepapic_spacing(data_root)
        self.register_buffer(
            "pepapic_denorm_scale",
            torch.as_tensor(scale, dtype=torch.float32).view(1, 1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pepapic_denorm_offset",
            torch.as_tensor(lo, dtype=torch.float32).view(1, 1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer("pepapic_dx", torch.tensor(float(dx), dtype=torch.float32), persistent=False)
        self.register_buffer("pepapic_dy", torch.tensor(float(dy), dtype=torch.float32), persistent=False)
        self.pepapic_poisson_floor = float(getattr(self.hparams, "pepapic_poisson_floor", 0.086))
        self.pepapic_poisson_floor_alpha = float(
            getattr(self.hparams, "pepapic_poisson_floor_alpha", 1.1)
        )
        self.pepapic_poisson_eps = float(getattr(self.hparams, "pepapic_poisson_eps", 1e-12))
        self.pepapic_efield_eps = float(getattr(self.hparams, "pepapic_efield_eps", 1e-12))

    def _load_pepapic_spacing(self, data_root):
        in_shape = getattr(self.hparams, "in_shape", None)
        if in_shape is None:
            h, w = 100, 100
        else:
            h, w = int(in_shape[-2]), int(in_shape[-1])

        domain_path = getattr(self.hparams, "pepapic_domain_info", None)
        if domain_path is None:
            # Expected H5 layout:
            # result_dir/SimVPv2_inputs/global_norm_*.h5
            domain_path = data_root.parent.parent / "Domain_info" / "Global_domain_info.json"
        else:
            domain_path = Path(str(domain_path))

        try:
            with open(domain_path, "r", encoding="utf-8") as f:
                domain = json.load(f)
            mins = np.asarray(domain["min_zones"][0], dtype=np.float64)
            maxs = np.asarray(domain["max_zones"][0], dtype=np.float64)
            lx = float(maxs[0] - mins[0])
            ly = float(maxs[1] - mins[1])
        except Exception:
            lx = float(w - 1)
            ly = float(h - 1)
        dx = lx / float(w - 1) if w > 1 else 1.0
        dy = ly / float(h - 1) if h > 1 else 1.0
        return dx, dy

    def _setup_pepapic_spectral_loss(self):
        mode = str(getattr(self.hparams, "pepapic_spectral_loss", "none")).lower()
        self.pepapic_spectral_amplitude_weight = float(
            getattr(self.hparams, "pepapic_spectral_amplitude_lambda", 0.0)
        )
        self.pepapic_spectral_phase_weight = float(
            getattr(self.hparams, "pepapic_spectral_phase_lambda", 0.0)
        )
        self.pepapic_spectral_cross_weight = float(
            getattr(self.hparams, "pepapic_spectral_cross_lambda", 0.0)
        )
        self.pepapic_spectral_complex_weight = float(
            getattr(self.hparams, "pepapic_spectral_complex_lambda", 0.0)
        )
        self.pepapic_transport_weight = float(
            getattr(self.hparams, "pepapic_transport_lambda", 0.0)
        )
        self.pepapic_spectral_enabled = (
            mode not in ("none", "off", "false", "0")
            and max(
                self.pepapic_spectral_amplitude_weight,
                self.pepapic_spectral_phase_weight,
                self.pepapic_spectral_cross_weight,
                self.pepapic_spectral_complex_weight,
                self.pepapic_transport_weight,
            )
            > 0.0
        )
        self.pepapic_spectral_loss_mode = mode
        if not self.pepapic_spectral_enabled:
            self.pepapic_spectral_loss_module = None
            return

        self.pepapic_spectral_loss_module = PEPAPICSpectralLoss(
            data_root=self.hparams.data_root,
            max_mode=int(getattr(self.hparams, "pepapic_spectral_max_mode", 30)),
            radial_bands=int(getattr(self.hparams, "pepapic_spectral_radial_bands", 4)),
            radial_min_m=float(
                getattr(self.hparams, "pepapic_spectral_radial_min_m", 0.09e-2)
            ),
            radial_max_m=float(
                getattr(self.hparams, "pepapic_spectral_radial_max_m", 1.19e-2)
            ),
            mtsi_mode_min=int(
                getattr(self.hparams, "pepapic_spectral_mtsi_mode_min", 1)
            ),
            mtsi_mode_max=int(
                getattr(self.hparams, "pepapic_spectral_mtsi_mode_max", 6)
            ),
            ecdi_mode_min=int(
                getattr(self.hparams, "pepapic_spectral_ecdi_mode_min", 9)
            ),
            ecdi_mode_max=int(
                getattr(self.hparams, "pepapic_spectral_ecdi_mode_max", 21)
            ),
            coordinate_system=str(
                getattr(self.hparams, "pepapic_spectral_coordinate_system", "fixed_n")
            ),
            q_min=float(getattr(self.hparams, "pepapic_spectral_q_min", 0.30)),
            q_max=float(getattr(self.hparams, "pepapic_spectral_q_max", 1.50)),
            q_bins=int(getattr(self.hparams, "pepapic_spectral_q_bins", 49)),
        )

    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if getattr(self.hparams, "simvp_direct_aft_seq", False):
            pred_y = self.model(batch_x)
            if pred_y.shape[1] != aft_seq_length:
                raise RuntimeError(
                    "simvp_direct_aft_seq=True requires the model output length "
                    f"to match aft_seq_length={aft_seq_length}, got {pred_y.shape[1]}."
                )
            return pred_y
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        loss, data_loss, poisson_loss, efield_loss, spectral_losses = self._total_loss(
            pred_y, batch_y, batch_x=batch_x
        )
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        if poisson_loss is not None:
            self.log('train_data_loss', data_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log('train_poisson_loss', poisson_loss, on_step=True, on_epoch=True, prog_bar=False)
        if efield_loss is not None:
            self.log('train_data_loss', data_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log('train_efield_loss', efield_loss, on_step=True, on_epoch=True, prog_bar=False)
        if spectral_losses is not None:
            self.log('train_data_loss', data_loss, on_step=True, on_epoch=True, prog_bar=False)
            for name, value in spectral_losses.items():
                self.log(
                    f'train_spectral_{name}_loss',
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss, data_loss, poisson_loss, efield_loss, spectral_losses = self._total_loss(
            pred_y, batch_y, batch_x=batch_x
        )
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        if poisson_loss is not None:
            self.log('val_data_loss', data_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log('val_poisson_loss', poisson_loss, on_step=True, on_epoch=True, prog_bar=False)
        if efield_loss is not None:
            self.log('val_data_loss', data_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log('val_efield_loss', efield_loss, on_step=True, on_epoch=True, prog_bar=False)
        if spectral_losses is not None:
            self.log('val_data_loss', data_loss, on_step=True, on_epoch=True, prog_bar=False)
            for name, value in spectral_losses.items():
                self.log(
                    f'val_spectral_{name}_loss',
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )
        return loss

    def _total_loss(self, pred_y, batch_y, batch_x=None):
        data_loss = self.criterion(pred_y, batch_y)
        total = data_loss
        poisson_loss = None
        efield_loss = None
        spectral_losses = None
        if getattr(self, "pepapic_poisson_enabled", False):
            poisson_loss = self._pepapic_poisson_loss(pred_y)
            total = total + self.pepapic_poisson_loss_weight * poisson_loss
        if getattr(self, "pepapic_efield_enabled", False):
            efield_loss = self._pepapic_efield_loss(pred_y, batch_y)
            total = total + self.pepapic_efield_loss_weight * efield_loss
        if getattr(self, "pepapic_spectral_enabled", False):
            spectral_losses = self.pepapic_spectral_loss_module(
                pred_y, batch_y, batch_x=batch_x
            )
            total = (
                total
                + self.pepapic_spectral_amplitude_weight
                * spectral_losses.get("amplitude", 0.0)
                + self.pepapic_spectral_phase_weight
                * spectral_losses.get("phase", 0.0)
                + self.pepapic_spectral_cross_weight
                * spectral_losses.get("cross_phase", 0.0)
                + self.pepapic_spectral_complex_weight
                * spectral_losses.get("complex_mode", 0.0)
                + self.pepapic_transport_weight
                * spectral_losses.get("transport", 0.0)
            )
        return total, data_loss, poisson_loss, efield_loss, spectral_losses

    def _denorm_pepapic(self, y):
        return y * self.pepapic_denorm_scale.to(dtype=y.dtype) + self.pepapic_denorm_offset.to(dtype=y.dtype)

    def _pepapic_laplacian(self, phi):
        dx2 = self.pepapic_dx.to(dtype=phi.dtype) ** 2
        dy2 = self.pepapic_dy.to(dtype=phi.dtype) ** 2
        return (
            (phi[..., 1:-1, 2:] - 2.0 * phi[..., 1:-1, 1:-1] + phi[..., 1:-1, :-2]) / dx2
            + (phi[..., 2:, 1:-1] - 2.0 * phi[..., 1:-1, 1:-1] + phi[..., :-2, 1:-1]) / dy2
        )

    def _pepapic_electric_field(self, phi):
        dx = self.pepapic_dx.to(dtype=phi.dtype)
        dy = self.pepapic_dy.to(dtype=phi.dtype)
        ex = -(phi[..., 1:-1, 2:] - phi[..., 1:-1, :-2]) / (2.0 * dx)
        ey = -(phi[..., 2:, 1:-1] - phi[..., :-2, 1:-1]) / (2.0 * dy)
        return ex, ey

    def _pepapic_poisson_loss(self, pred_y):
        physical = self._denorm_pepapic(pred_y)
        phi = physical[:, :, self.pepapic_phi_index]
        ne = physical[:, :, self.pepapic_electron_index]
        ni = physical[:, :, self.pepapic_ion_index]

        lap = self._pepapic_laplacian(phi)
        source = E_CHARGE * (ni[..., 1:-1, 1:-1] - ne[..., 1:-1, 1:-1]) / EPS0
        residual = lap + source

        residual_rms = torch.sqrt(torch.mean(residual * residual, dim=(-2, -1)))
        source_rms = torch.sqrt(torch.mean(source * source, dim=(-2, -1)))
        relative_residual = residual_rms / torch.clamp(source_rms, min=self.pepapic_poisson_eps)

        if self.pepapic_poisson_loss_mode in ("zero", "residual", "relative_zero"):
            return torch.mean(relative_residual * relative_residual)
        if self.pepapic_poisson_loss_mode in ("floor_hinge", "hinge", "floor"):
            floor = self.pepapic_poisson_floor * self.pepapic_poisson_floor_alpha
            excess = torch.relu(relative_residual - floor)
            return torch.mean(excess * excess)
        raise ValueError(f"Unknown pepapic_poisson_loss: {self.pepapic_poisson_loss_mode}")

    def _pepapic_efield_loss(self, pred_y, true_y):
        pred_physical = self._denorm_pepapic(pred_y)
        true_physical = self._denorm_pepapic(true_y)
        pred_phi = pred_physical[:, :, self.pepapic_phi_index]
        true_phi = true_physical[:, :, self.pepapic_phi_index]

        pred_ex, pred_ey = self._pepapic_electric_field(pred_phi)
        true_ex, true_ey = self._pepapic_electric_field(true_phi)

        numerator = torch.mean((pred_ex - true_ex) ** 2 + (pred_ey - true_ey) ** 2)
        denominator = torch.mean(true_ex ** 2 + true_ey ** 2)
        loss = numerator / torch.clamp(denominator, min=self.pepapic_efield_eps)

        if self.pepapic_efield_loss_mode in ("normalized_mse", "relative_mse", "mse"):
            return loss
        raise ValueError(f"Unknown pepapic_efield_loss: {self.pepapic_efield_loss_mode}")
