import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn


class PEPAPICSpectralLoss(nn.Module):
    """Differentiable azimuthal-mode losses for radial-azimuthal PIC data."""

    def __init__(
        self,
        data_root,
        max_mode=30,
        radial_bands=4,
        radial_min_m=0.09e-2,
        radial_max_m=1.19e-2,
        mtsi_mode_min=1,
        mtsi_mode_max=6,
        ecdi_mode_min=9,
        ecdi_mode_max=21,
        coordinate_system="fixed_n",
        q_min=0.30,
        q_max=1.50,
        q_bins=49,
        eps=1e-8,
    ):
        super().__init__()
        data_root = Path(str(data_root))
        if data_root.suffix.lower() not in (".h5", ".hdf5", ".json"):
            raise ValueError(
                "PEPAPICSpectralLoss requires a RadAz H5 or multi-case manifest, "
                f"got {data_root}"
            )

        metadata = self._load_metadata(data_root)
        props = metadata["props"]
        for required in ("electron_den", "phi"):
            if required not in props:
                raise ValueError(
                    f"PEPAPICSpectralLoss requires channel {required}, got {props}"
                )

        self.electron_index = props.index("electron_den")
        self.phi_index = props.index("phi")
        self.valid_height, self.valid_width = metadata["valid_spatial_shape"]
        self.model_height, self.model_width = metadata["model_spatial_shape"]
        self.max_mode = min(int(max_mode), self.valid_width // 2)
        self.radial_bands = int(radial_bands)
        self.coordinate_system = str(coordinate_system).lower()
        self.eps = float(eps)
        if self.max_mode < 1:
            raise ValueError("max_mode must include at least azimuthal mode n=1")
        if self.radial_bands < 1:
            raise ValueError("radial_bands must be positive")

        band_pool = self._make_radial_band_pool(
            metadata["x_m"],
            self.model_height,
            self.radial_bands,
            float(radial_min_m),
            float(radial_max_m),
        )
        mode_weights = self._make_mode_weights(
            self.max_mode,
            int(mtsi_mode_min),
            int(mtsi_mode_max),
            int(ecdi_mode_min),
            int(ecdi_mode_max),
        )

        y_m = np.asarray(metadata["y_m"], dtype=np.float64)
        if y_m.size < 2:
            raise ValueError("H5 y_m must contain at least two azimuthal coordinates")
        dy = float(np.median(np.diff(y_m)))
        if not np.isfinite(dy) or dy <= 0.0:
            raise ValueError(f"Invalid azimuthal spacing inferred from y_m: {dy}")

        self.register_buffer(
            "denorm_scale",
            torch.as_tensor(metadata["norm_high"] - metadata["norm_low"], dtype=torch.float32)
            .view(1, 1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "denorm_offset",
            torch.as_tensor(metadata["norm_low"], dtype=torch.float32)
            .view(1, 1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "radial_band_pool",
            torch.as_tensor(band_pool, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "mode_weights",
            torch.as_tensor(mode_weights, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "azimuthal_spacing_m",
            torch.tensor(dy, dtype=torch.float32),
            persistent=False,
        )
        if self.coordinate_system in ("q", "q_normalized", "q_complex_transport"):
            condition_names = metadata.get("condition_names", [])
            condition_mean = metadata.get("condition_mean")
            condition_std = metadata.get("condition_std")
            for required in ("log_vE", "log_n0"):
                if required not in condition_names:
                    raise ValueError(
                        f"q-normalized spectral loss requires condition {required}, "
                        f"got {condition_names}"
                    )
            self.log_ve_index = condition_names.index("log_vE")
            self.log_n0_index = condition_names.index("log_n0")
            self.condition_dim = len(condition_names)
            self.register_buffer(
                "condition_mean",
                torch.as_tensor(condition_mean, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "condition_std",
                torch.as_tensor(condition_std, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "q_grid",
                torch.linspace(float(q_min), float(q_max), int(q_bins)),
                persistent=False,
            )
        elif self.coordinate_system not in ("fixed", "fixed_n", "n"):
            raise ValueError(
                f"Unknown spectral coordinate_system={coordinate_system}"
            )

    @staticmethod
    def _decode_strings(values):
        return [
            value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
            for value in values
        ]

    @classmethod
    def _load_metadata(cls, data_root):
        if data_root.suffix.lower() == ".json":
            return cls._load_manifest_metadata(data_root)

        with h5py.File(data_root, "r") as handle:
            props = (
                cls._decode_strings(handle["props"][()])
                if "props" in handle
                else ["electron_den", "ion_den", "phi"]
            )
            train_min = np.asarray(handle["train_min"][()], dtype=np.float32)
            train_max = np.asarray(handle["train_max"][()], dtype=np.float32)
            margin = float(handle["margin"][()]) if "margin" in handle else 0.0
            if "norm_low" in handle and "norm_high" in handle:
                norm_low = np.asarray(handle["norm_low"][()], dtype=np.float32)
                norm_high = np.asarray(handle["norm_high"][()], dtype=np.float32)
            else:
                span = train_max - train_min
                norm_low = train_min - margin * span
                norm_high = train_max + margin * span

            if "valid_spatial_shape" in handle:
                valid_shape = tuple(
                    int(value) for value in np.asarray(handle["valid_spatial_shape"][()]).ravel()
                )
            else:
                valid_shape = tuple(int(value) for value in handle["data"].shape[-2:])
            if "model_spatial_shape" in handle:
                model_shape = tuple(
                    int(value) for value in np.asarray(handle["model_spatial_shape"][()]).ravel()
                )
            else:
                model_shape = tuple(int(value) for value in handle["data"].shape[-2:])

            x_m = np.asarray(handle["x_m"][()], dtype=np.float64)
            y_m = np.asarray(handle["y_m"][()], dtype=np.float64)

        if len(valid_shape) != 2 or len(model_shape) != 2:
            raise ValueError(
                f"Expected 2D valid/model spatial shapes, got {valid_shape}, {model_shape}"
            )
        if x_m.size != valid_shape[0] or y_m.size != valid_shape[1]:
            raise ValueError(
                "Coordinate lengths do not match valid_spatial_shape: "
                f"x={x_m.size}, y={y_m.size}, valid={valid_shape}"
            )
        if norm_low.shape != norm_high.shape or norm_low.size != len(props):
            raise ValueError(
                f"Normalization metadata does not match props: {norm_low.shape}, "
                f"{norm_high.shape}, {props}"
            )
        return {
            "props": props,
            "norm_low": norm_low,
            "norm_high": norm_high,
            "valid_spatial_shape": valid_shape,
            "model_spatial_shape": model_shape,
            "x_m": x_m,
            "y_m": y_m,
            "condition_names": [],
            "condition_mean": None,
            "condition_std": None,
        }

    @classmethod
    def _load_manifest_metadata(cls, manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = manifest.get("cases")
        if not cases:
            raise ValueError(
                f"PEPAPIC manifest must contain at least one case: {manifest_path}"
            )

        normalization = manifest.get("normalization", {})
        condition_normalization = manifest.get("condition_normalization", {})
        props = normalization.get("channels", cases[0].get("channels"))
        norm_low = normalization.get("low", cases[0].get("normalization_low"))
        norm_high = normalization.get("high", cases[0].get("normalization_high"))
        if props is None or norm_low is None or norm_high is None:
            raise ValueError(
                "Manifest spectral loss requires channels and common normalization "
                f"bounds: {manifest_path}"
            )
        props = [str(value) for value in props]
        norm_low = np.asarray(norm_low, dtype=np.float32)
        norm_high = np.asarray(norm_high, dtype=np.float32)
        if norm_low.shape != (len(props),) or norm_high.shape != (len(props),):
            raise ValueError(
                "Manifest normalization bounds do not match channels: "
                f"{norm_low.shape}, {norm_high.shape}, {props}"
            )
        if np.any(~np.isfinite(norm_low)) or np.any(~np.isfinite(norm_high)):
            raise ValueError("Manifest normalization bounds must be finite")
        if np.any(norm_high <= norm_low):
            raise ValueError("Manifest normalization bounds must have positive spans")

        reference = None
        base = manifest_path.resolve().parent
        for index, case in enumerate(cases):
            case_props = [str(value) for value in case.get("channels", props)]
            case_low = np.asarray(
                case.get("normalization_low", norm_low), dtype=np.float32
            )
            case_high = np.asarray(
                case.get("normalization_high", norm_high), dtype=np.float32
            )
            if case_props != props:
                raise ValueError(
                    f"Manifest case {index} channels differ from the common channels"
                )
            if not np.allclose(case_low, norm_low) or not np.allclose(
                case_high, norm_high
            ):
                raise ValueError(
                    f"Manifest case {index} does not use the common normalization"
                )

            h5_path = Path(str(case["path"]))
            if not h5_path.is_absolute():
                h5_path = base / h5_path
            if not h5_path.is_file():
                raise FileNotFoundError(h5_path)

            case_format = str(case.get("format", "canonical_h5")).lower()
            if case_format not in (
                "radaz_consolidated",
                "radaz_consolidated_fields",
            ):
                raise ValueError(
                    "Manifest spectral loss currently requires consolidated RadAz "
                    f"cases, got {case_format} for {h5_path}"
                )

            spatial_stride = int(case.get("spatial_stride", 1))
            if spatial_stride <= 0 or 256 % spatial_stride:
                raise ValueError(
                    f"RadAz spatial_stride must divide 256, got {spatial_stride}"
                )
            native_grid = spatial_stride == 1
            valid_height = 257 if native_grid else 256 // spatial_stride
            valid_width = 256 // spatial_stride
            model_height = int(
                case.get("model_height", 260 if native_grid else valid_height)
            )
            model_width = int(case.get("model_width", valid_width))

            with h5py.File(h5_path, "r") as handle:
                if "axes/x_m" not in handle or "axes/y_m" not in handle:
                    raise KeyError(f"Missing RadAz coordinates in {h5_path}")
                x_source = np.asarray(handle["axes/x_m"][()], dtype=np.float64)
                y_source = np.asarray(handle["axes/y_m"][()], dtype=np.float64)

            if native_grid:
                x_m = x_source[:valid_height]
                y_m = y_source[:valid_width]
            else:
                x_m = x_source[:256].reshape(valid_height, spatial_stride).mean(1)
                y_m = y_source[:256].reshape(valid_width, spatial_stride).mean(1)

            metadata = {
                "valid_spatial_shape": (valid_height, valid_width),
                "model_spatial_shape": (model_height, model_width),
                "x_m": x_m,
                "y_m": y_m,
            }
            if reference is None:
                reference = metadata
            else:
                for key in ("valid_spatial_shape", "model_spatial_shape"):
                    if metadata[key] != reference[key]:
                        raise ValueError(
                            f"Manifest case {index} has inconsistent {key}: "
                            f"{metadata[key]} != {reference[key]}"
                        )
                if not np.allclose(metadata["x_m"], reference["x_m"]) or not np.allclose(
                    metadata["y_m"], reference["y_m"]
                ):
                    raise ValueError(
                        f"Manifest case {index} uses a different spatial grid"
                    )

        return {
            "props": props,
            "norm_low": norm_low,
            "norm_high": norm_high,
            "condition_names": [
                str(value) for value in condition_normalization.get("names", [])
            ],
            "condition_mean": condition_normalization.get("mean"),
            "condition_std": condition_normalization.get("std"),
            **reference,
        }

    @staticmethod
    def _make_radial_band_pool(
        x_m,
        model_height,
        radial_bands,
        radial_min_m,
        radial_max_m,
    ):
        x_m = np.asarray(x_m, dtype=np.float64)
        if radial_min_m >= radial_max_m:
            raise ValueError("radial_min_m must be smaller than radial_max_m")
        edges = np.linspace(radial_min_m, radial_max_m, radial_bands + 1)
        pool = np.zeros((radial_bands, int(model_height)), dtype=np.float32)
        for band_index in range(radial_bands):
            if band_index == radial_bands - 1:
                selected = (x_m >= edges[band_index]) & (x_m <= edges[band_index + 1])
            else:
                selected = (x_m >= edges[band_index]) & (x_m < edges[band_index + 1])
            indices = np.flatnonzero(selected)
            if indices.size == 0:
                raise ValueError(
                    "No radial cells in spectral-loss band "
                    f"{band_index}: [{edges[band_index]}, {edges[band_index + 1]}]"
                )
            pool[band_index, indices] = 1.0 / float(indices.size)
        return pool

    @staticmethod
    def _make_mode_weights(max_mode, mtsi_min, mtsi_max, ecdi_min, ecdi_max):
        modes = np.arange(1, max_mode + 1)
        masks = [
            (modes >= mtsi_min) & (modes <= mtsi_max),
            (modes >= ecdi_min) & (modes <= ecdi_max),
        ]
        masks.append(~(masks[0] | masks[1]))
        group_masses = (0.4, 0.4, 0.2)
        weights = np.zeros(max_mode, dtype=np.float32)
        active_mass = 0.0
        for mask, mass in zip(masks, group_masses):
            count = int(np.count_nonzero(mask))
            if count:
                weights[mask] = mass / float(count)
                active_mass += mass
        if active_mass <= 0.0:
            raise ValueError("No azimuthal modes were selected")
        weights /= np.sum(weights)
        return weights

    def _denormalize(self, values):
        return (
            values * self.denorm_scale.to(dtype=values.dtype)
            + self.denorm_offset.to(dtype=values.dtype)
        )

    def _magnitude(self, values):
        return torch.sqrt(torch.sum(values * values, dim=-1))

    def _unit_pair(self, values):
        magnitude = self._magnitude(values)
        return values / torch.clamp(magnitude.unsqueeze(-1), min=self.eps)

    @staticmethod
    def _multiply_conjugate(left, right):
        real = left[..., 0] * right[..., 0] + left[..., 1] * right[..., 1]
        imag = left[..., 1] * right[..., 0] - left[..., 0] * right[..., 1]
        return torch.stack((real, imag), dim=-1)

    def _normalize_reliability(self, values, dims):
        scale = torch.mean(values.detach(), dim=dims, keepdim=True)
        return torch.clamp(values.detach() / torch.clamp(scale, min=self.eps), max=10.0)

    def _band_coefficients(self, normalized, physical_units=True):
        if normalized.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(normalized.shape)}")
        if normalized.shape[-2] < self.valid_height or normalized.shape[-1] < self.valid_width:
            raise ValueError(
                f"Input spatial shape {tuple(normalized.shape[-2:])} is smaller than "
                f"valid shape {(self.valid_height, self.valid_width)}"
            )

        values = self._denormalize(normalized) if physical_units else normalized
        values = values[..., : self.valid_height, : self.valid_width]
        phi = values[:, :, self.phi_index]
        electron_den = values[:, :, self.electron_index]
        # With common source normalization, normalized-space ne_hat*Ey_hat is
        # proportional to the physical modal transport by one global constant.
        # Omitting dy in q mode keeps gradients numerically comparable across
        # operating conditions without changing relative modal structure.
        dy = (
            self.azimuthal_spacing_m.to(dtype=phi.dtype)
            if physical_units
            else torch.tensor(1.0, dtype=phi.dtype, device=phi.device)
        )
        electric_y = -(
            torch.roll(phi, shifts=-1, dims=-1)
            - torch.roll(phi, shifts=1, dims=-1)
        ) / (2.0 * dy)

        fields = torch.stack((phi, electron_den, electric_y), dim=2)
        pool = self.radial_band_pool[:, : self.valid_height].to(dtype=fields.dtype)
        band_fields = torch.einsum("rh,btfhw->btfrw", pool, fields)
        band_fluctuations = band_fields - torch.mean(
            band_fields, dim=-1, keepdim=True
        )
        coefficients = torch.fft.rfft(band_fluctuations, dim=-1, norm="forward")
        coefficients = coefficients[..., 1 : self.max_mode + 1]
        return torch.view_as_real(coefficients)

    def _amplitude_loss(self, pred_coeff, true_coeff):
        pred_power = torch.sum(pred_coeff * pred_coeff, dim=-1)
        true_power = torch.sum(true_coeff * true_coeff, dim=-1)
        power_scale = torch.mean(
            true_power.detach(), dim=(1, 3, 4), keepdim=True
        )
        pred_log_power = torch.log1p(
            pred_power / torch.clamp(power_scale, min=self.eps)
        )
        true_log_power = torch.log1p(
            true_power / torch.clamp(power_scale, min=self.eps)
        )
        error = (pred_log_power - true_log_power) ** 2
        weights = self.mode_weights.to(dtype=error.dtype).view(1, 1, 1, 1, -1)
        return torch.mean(torch.sum(error * weights, dim=-1))

    def _phase_increment_loss(self, pred_coeff, true_coeff):
        pred_unit = self._unit_pair(pred_coeff)
        true_unit = self._unit_pair(true_coeff)
        pred_increment = self._multiply_conjugate(
            pred_unit[:, 1:], pred_unit[:, :-1]
        )
        true_increment = self._multiply_conjugate(
            true_unit[:, 1:], true_unit[:, :-1]
        )
        circular_error = torch.clamp(
            1.0 - torch.sum(pred_increment * true_increment, dim=-1),
            min=0.0,
        )

        true_amplitude = self._magnitude(true_coeff)
        reliability = torch.sqrt(
            true_amplitude[:, 1:] * true_amplitude[:, :-1]
        )
        reliability = self._normalize_reliability(
            reliability, dims=(1, 3, 4)
        )
        weights = self.mode_weights.to(dtype=circular_error.dtype).view(
            1, 1, 1, 1, -1
        )
        return torch.mean(
            torch.sum(circular_error * reliability * weights, dim=-1)
        )

    def _cross_phase_loss(self, pred_coeff, true_coeff):
        # Coefficient field order is phi, electron density, E_y.
        pred_cross = self._multiply_conjugate(
            self._unit_pair(pred_coeff[:, :, 1]),
            self._unit_pair(pred_coeff[:, :, 2]),
        )
        true_cross = self._multiply_conjugate(
            self._unit_pair(true_coeff[:, :, 1]),
            self._unit_pair(true_coeff[:, :, 2]),
        )
        circular_error = torch.clamp(
            1.0 - torch.sum(pred_cross * true_cross, dim=-1),
            min=0.0,
        )

        true_cross_magnitude = (
            self._magnitude(true_coeff[:, :, 1])
            * self._magnitude(true_coeff[:, :, 2])
        )
        reliability = self._normalize_reliability(
            true_cross_magnitude, dims=(1, 2, 3)
        )
        weights = self.mode_weights.to(dtype=circular_error.dtype).view(
            1, 1, 1, -1
        )
        return torch.mean(
            torch.sum(circular_error * reliability * weights, dim=-1)
        )

    def _physical_conditions(self, batch_x):
        if batch_x is None:
            raise ValueError("q-normalized loss requires batch_x condition channels")
        if batch_x.shape[2] < self.condition_dim:
            raise ValueError(
                f"batch_x has {batch_x.shape[2]} channels but condition_dim={self.condition_dim}"
            )
        standardized = batch_x[:, :, -self.condition_dim :].mean(dim=(1, 3, 4))
        raw = (
            standardized * self.condition_std.to(dtype=batch_x.dtype)
            + self.condition_mean.to(dtype=batch_x.dtype)
        )
        log_ve = raw[:, self.log_ve_index]
        log_n0 = raw[:, self.log_n0_index]
        return torch.exp(log_ve), torch.exp(log_n0)

    def _interpolate_to_q(self, coefficients, mode_n0):
        # coefficients: [B,T,F,R,N,2], with N corresponding to n=1..max_mode.
        batch, time, fields, radial, modes, complex_dim = coefficients.shape
        positions = (
            mode_n0[:, None] * self.q_grid.to(dtype=coefficients.dtype)[None, :]
            - 1.0
        )
        valid = (positions >= 0.0) & (positions <= float(modes - 1))
        lower = torch.floor(positions).long().clamp(0, modes - 1)
        upper = (lower + 1).clamp(0, modes - 1)
        fraction = (positions - lower.to(positions.dtype)).clamp(0.0, 1.0)

        index_shape = (batch, time, fields, radial, -1, complex_dim)
        lower_index = lower[:, None, None, None, :, None].expand(index_shape)
        upper_index = upper[:, None, None, None, :, None].expand(index_shape)
        lower_value = torch.gather(coefficients, dim=4, index=lower_index)
        upper_value = torch.gather(coefficients, dim=4, index=upper_index)
        fraction = fraction[:, None, None, None, :, None]
        interpolated = lower_value + fraction * (upper_value - lower_value)
        valid_mask = valid[:, None, None, None, :, None]
        return interpolated, valid_mask

    def _complex_mode_loss(self, pred_coeff, true_coeff, valid_mask):
        # The three input fields already share one source-train affine
        # normalization across every case. Keep this absolute normalized-space
        # error rather than dividing by each case's q-band power; the latter
        # makes weak-mode cases dominate the gradient.
        error = (pred_coeff - true_coeff) ** 2
        mask = valid_mask.to(dtype=error.dtype)
        denominator = torch.clamp(
            mask.sum() * error.shape[1] * error.shape[2] * error.shape[3] * 2,
            min=1.0,
        )
        return torch.sum(error * mask) / denominator

    def _transport_loss(self, pred_coeff, true_coeff, valid_mask, drift_velocity, mode_n0):
        # From n0 = e B Ly / (2 pi m_e v_E), recover B for physical Gamma.
        electron_mass = 9.1093837015e-31
        electron_charge = 1.602176634e-19
        ly_m = 1.28e-2
        magnetic_t = (
            mode_n0 * (2.0 * torch.pi * electron_mass) * drift_velocity
            / (electron_charge * ly_m)
        ).to(torch.float64)

        # ne_hat * Ey_hat is O(1e20--1e22) in SI units. Squaring it in
        # float32 overflows even though the final relative loss is moderate.
        # This block is small (radial bands x q bins), so evaluate it in
        # float64 and cast the scalar back while retaining autograd.
        pred_coeff64 = pred_coeff.to(torch.float64)
        true_coeff64 = true_coeff.to(torch.float64)

        pred_cross = self._multiply_conjugate(
            pred_coeff64[:, :, 1], pred_coeff64[:, :, 2]
        )
        true_cross = self._multiply_conjugate(
            true_coeff64[:, :, 1], true_coeff64[:, :, 2]
        )
        # Express B in units of the 20 mT reference. Since field channels use
        # common normalization, this is proportional to physical Gamma by one
        # global constant while preserving the required 1/B dependence.
        relative_b = magnetic_t / 0.020
        pred_transport = -pred_cross[..., 0] / relative_b[:, None, None, None]
        true_transport = -true_cross[..., 0] / relative_b[:, None, None, None]
        error = (pred_transport - true_transport) ** 2
        mask = valid_mask[:, :, 0, :, :, 0].to(dtype=error.dtype)
        denominator = torch.clamp(
            mask.sum() * error.shape[1] * error.shape[2], min=1.0
        )
        return (torch.sum(error * mask) / denominator).to(dtype=pred_coeff.dtype)

    def forward(self, pred_y, true_y, batch_x=None):
        if self.coordinate_system in ("q", "q_normalized", "q_complex_transport"):
            pred_coeff = self._band_coefficients(pred_y, physical_units=False)
            true_coeff = self._band_coefficients(true_y, physical_units=False)
            drift_velocity, mode_n0 = self._physical_conditions(batch_x)
            pred_q, valid_mask = self._interpolate_to_q(pred_coeff, mode_n0)
            true_q, _ = self._interpolate_to_q(true_coeff, mode_n0)
            return {
                "complex_mode": self._complex_mode_loss(
                    pred_q, true_q, valid_mask
                ),
                "transport": self._transport_loss(
                    pred_q, true_q, valid_mask, drift_velocity, mode_n0
                ),
            }
        pred_coeff = self._band_coefficients(pred_y, physical_units=True)
        true_coeff = self._band_coefficients(true_y, physical_units=True)
        return {
            "amplitude": self._amplitude_loss(pred_coeff, true_coeff),
            "phase": self._phase_increment_loss(pred_coeff, true_coeff),
            "cross_phase": self._cross_phase_loss(pred_coeff, true_coeff),
        }
