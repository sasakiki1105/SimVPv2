#!/usr/bin/env python3
"""Reusable electric-history hidden-band ROM components for RadAz data.

The deployment input is an ideal G8-observable Fourier state (n=0--16) at
every coarse-PIC output time.  A causal recurrent observer uses that state and
the known electric-field history to reconstruct modes n=17--21.  No fine-grid
coefficient from the current or previous time is supplied to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn


VISIBLE_MODES = tuple(range(17))
HIDDEN_MODES = tuple(range(17, 22))
FIELD_NAMES = ("phi", "electron_den", "ion_den", "efy")
INPUT_FIELD_NAMES = ("phi", "electron_den", "ion_den")
ELECTRIC_MEMORY_TIMES_US = (0.30, 1.50, 5.00)
PARAMETER_CENTER_KVM = 25.0
PARAMETER_SCALE_KVM = 10.0
TIME_SCALE_US = 10.0
CONTROL_DIMENSION = 9


def electric_history_controls(
    time_us: np.ndarray,
    current_ez_kvm: float,
    source_ez_kvm: float,
    transition: bool,
    step_time_us: float = 30.0,
) -> np.ndarray:
    """Return causal controls derived only from the prescribed Ez history."""
    time_us = np.asarray(time_us, dtype=np.float64)
    count = len(time_us)
    current = np.full(
        count,
        (current_ez_kvm - PARAMETER_CENTER_KVM) / PARAMETER_SCALE_KVM,
        dtype=np.float64,
    )
    source = np.full(
        count,
        (source_ez_kvm - PARAMETER_CENTER_KVM) / PARAMETER_SCALE_KVM,
        dtype=np.float64,
    )
    delta = current - source
    flag = np.full(count, float(transition), dtype=np.float64)
    age_us = (
        np.maximum(time_us - step_time_us, 0.0)
        if transition
        else np.zeros(count, dtype=np.float64)
    )
    memories = [
        delta * np.exp(-age_us / timescale)
        for timescale in ELECTRIC_MEMORY_TIMES_US
    ]
    result = np.column_stack(
        (
            current,
            source,
            delta,
            flag,
            age_us / TIME_SCALE_US,
            np.log1p(age_us) / np.log1p(TIME_SCALE_US),
            *memories,
        )
    )
    if result.shape != (count, CONTROL_DIMENSION):
        raise AssertionError("electric-history control dimension mismatch")
    return result.astype(np.float32)


def complex_to_real(coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients)
    return np.stack((values.real, values.imag), axis=-1).reshape(len(values), -1)


def real_to_complex(
    values: np.ndarray, fields: int, radial_bands: int, modes: int
) -> np.ndarray:
    shaped = np.asarray(values).reshape(len(values), fields, radial_bands, modes, 2)
    return shaped[..., 0] + 1j * shaped[..., 1]


def invariant_observable_features(
    visible_coefficients: np.ndarray, amplitude_scale: np.ndarray
) -> np.ndarray:
    """Rotation-invariant amplitude and causal phase-increment descriptors."""
    values = np.asarray(visible_coefficients)
    amplitude = np.abs(values).astype(np.float64)
    scale = np.asarray(amplitude_scale, dtype=np.float64)
    log_amplitude = np.log1p(amplitude / scale[None, ...])
    previous = np.concatenate((values[:1], values[:-1]), axis=0)
    previous_amplitude = np.abs(previous).astype(np.float64)
    cross = values.astype(np.complex128) * np.conj(previous.astype(np.complex128))
    denominator = amplitude * previous_amplitude
    unit_increment = cross / np.maximum(denominator, np.finfo(np.float64).tiny)
    reliability = np.sqrt(denominator) / (
        np.sqrt(denominator) + scale[None, ...]
    )
    unit_increment *= reliability
    return np.concatenate(
        (
            log_amplitude.reshape(len(values), -1),
            unit_increment.real.reshape(len(values), -1),
            unit_increment.imag.reshape(len(values), -1),
        ),
        axis=1,
    )


def apply_hidden_amplitude_to_carrier(
    carrier_coefficients: np.ndarray, predicted_amplitude: np.ndarray
) -> np.ndarray:
    """Keep a carrier's phase and replace its n=17--21 amplitudes."""
    carrier = np.asarray(carrier_coefficients)
    amplitude = np.asarray(predicted_amplitude)
    if carrier.shape != amplitude.shape:
        raise ValueError(f"carrier/amplitude shapes differ: {carrier.shape}, {amplitude.shape}")
    magnitude = np.abs(carrier)
    unit = np.zeros_like(carrier, dtype=np.result_type(carrier, np.complex128))
    np.divide(carrier, magnitude, out=unit, where=magnitude > 0.0)
    return unit * amplitude


def electric_history_blend_weight(controls: np.ndarray) -> np.ndarray:
    """Conservative, pre-primary gate for blending ROM and G8 amplitudes.

    The frozen G8 model is trusted at its E25 stationary training point.  The
    ROM weight grows with distance from E25 and with the still-active causal
    electric-step memories.  Constants are fixed before the E25->E22.5
    confirmatory trajectory is available and must not be tuned on that case.
    """
    controls = np.asarray(controls, dtype=np.float64)
    if controls.shape[-1] != CONTROL_DIMENSION:
        raise ValueError(f"expected {CONTROL_DIMENSION} controls, received {controls.shape}")
    distance_from_e25 = np.abs(controls[..., 0]) / 0.5
    active_step_memory = np.max(np.abs(controls[..., 6:9]), axis=-1) / 0.25
    return np.clip(np.maximum(distance_from_e25, active_step_memory), 0.0, 1.0)


def blend_carrier_and_rom_amplitude(
    carrier_coefficients: np.ndarray,
    rom_amplitude: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Blend amplitudes while preserving the supplied carrier phase."""
    carrier = np.asarray(carrier_coefficients)
    rom_amplitude = np.asarray(rom_amplitude)
    weight = np.asarray(weight, dtype=np.float64)
    while weight.ndim < carrier.ndim:
        weight = weight[..., None]
    amplitude = (1.0 - weight) * np.abs(carrier) + weight * rom_amplitude
    return apply_hidden_amplitude_to_carrier(carrier, amplitude)


class HistoryHiddenBandROM(nn.Module):
    """Causal GRU observer for modes absent from an ideal G8 input."""

    def __init__(
        self,
        observable_dimension: int,
        output_dimension: int,
        hidden_dimension: int = 128,
        layers: int = 2,
        control_dimension: int = CONTROL_DIMENSION,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.observable_dimension = int(observable_dimension)
        self.control_dimension = int(control_dimension)
        self.recurrent = nn.GRU(
            self.observable_dimension + self.control_dimension,
            hidden_dimension,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, observable: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        sequence = torch.cat((observable, controls), dim=-1)
        encoded, _ = self.recurrent(sequence)
        return self.head(encoded[:, -1])


class InstantHiddenBandROM(nn.Module):
    """Current-state ablation with no temporal state."""

    def __init__(
        self,
        observable_dimension: int,
        output_dimension: int,
        hidden_dimension: int = 128,
        control_dimension: int = CONTROL_DIMENSION,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observable_dimension + control_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, observable: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observable[:, -1], controls[:, -1]), dim=-1))


@dataclass
class HiddenBandBundle:
    model: nn.Module
    observable_scaler: object
    observable_pca: object
    hidden_scaler: object
    hidden_pca: object
    history_steps: int
    device: torch.device

    @classmethod
    def load(
        cls,
        artifact_directory: Path | str,
        device: str | torch.device = "cpu",
        model_name: str = "history_electric",
    ) -> "HiddenBandBundle":
        directory = Path(artifact_directory)
        transforms = joblib.load(directory / "transforms.joblib")
        checkpoint = torch.load(
            directory / f"{model_name}.pt", map_location=torch.device(device)
        )
        model_type = checkpoint["model_type"]
        kwargs = checkpoint["model_kwargs"]
        if model_type == "history":
            model = HistoryHiddenBandROM(**kwargs)
        elif model_type == "instant":
            model = InstantHiddenBandROM(**kwargs)
        else:
            raise ValueError(f"unknown model type: {model_type}")
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        return cls(
            model=model,
            observable_scaler=transforms["observable_scaler"],
            observable_pca=transforms["observable_pca"],
            hidden_scaler=transforms["hidden_scaler"],
            hidden_pca=transforms["hidden_pca"],
            history_steps=int(checkpoint["history_steps"]),
            device=torch.device(device),
        )

    def encode_observable(self, visible_coefficients: np.ndarray) -> np.ndarray:
        flattened = complex_to_real(visible_coefficients)
        scaled = self.observable_scaler.transform(flattened)
        return self.observable_pca.transform(scaled).astype(np.float32)

    def decode_hidden(self, encoded: np.ndarray) -> np.ndarray:
        scaled = self.hidden_pca.inverse_transform(np.asarray(encoded))
        flattened = self.hidden_scaler.inverse_transform(scaled)
        fields = len(FIELD_NAMES)
        radial_bands = flattened.shape[1] // (fields * len(HIDDEN_MODES) * 2)
        return real_to_complex(flattened, fields, radial_bands, len(HIDDEN_MODES))

    def predict(
        self,
        visible_coefficients: np.ndarray,
        controls: np.ndarray,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict all times with a complete causal history window.

        Returns ``(indices, hidden_coefficients)``; early indices lacking the
        requested history are intentionally omitted.
        """
        observable = self.encode_observable(visible_coefficients)
        controls = np.asarray(controls, dtype=np.float32)
        indices = np.arange(self.history_steps - 1, len(observable), dtype=np.int64)
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for begin in range(0, len(indices), batch_size):
                selected = indices[begin : begin + batch_size]
                x = np.stack(
                    [observable[i - self.history_steps + 1 : i + 1] for i in selected]
                )
                u = np.stack(
                    [controls[i - self.history_steps + 1 : i + 1] for i in selected]
                )
                value = self.model(
                    torch.from_numpy(x).to(self.device),
                    torch.from_numpy(u).to(self.device),
                )
                outputs.append(value.cpu().numpy())
        return indices, self.decode_hidden(np.concatenate(outputs, axis=0))


@dataclass
class HiddenBandEnvelopeBundle:
    """Deployable electric-history amplitude ROM for a supplied phase carrier."""

    model: nn.Module
    transforms: dict
    history_steps: int
    device: torch.device

    @classmethod
    def load(
        cls,
        artifact_directory: Path | str,
        device: str | torch.device = "cpu",
        model_name: str = "history_electric",
    ) -> "HiddenBandEnvelopeBundle":
        directory = Path(artifact_directory)
        transforms = joblib.load(directory / "transforms.joblib")
        if transforms.get("representation") != "rotation_invariant_carrier_envelope":
            raise ValueError("artifact is not a carrier-envelope hidden-band ROM")
        checkpoint = torch.load(
            directory / f"{model_name}.pt",
            map_location=torch.device(device),
            weights_only=False,
        )
        if checkpoint["model_type"] != "history":
            raise ValueError("envelope deployment expects a recurrent history model")
        model = HistoryHiddenBandROM(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        return cls(
            model=model,
            transforms=transforms,
            history_steps=int(checkpoint["history_steps"]),
            device=torch.device(device),
        )

    def encode_observable(self, visible_coefficients: np.ndarray) -> np.ndarray:
        raw = invariant_observable_features(
            visible_coefficients,
            self.transforms["observable_amplitude_scale"],
        )
        scaled = self.transforms["observable_scaler"].transform(raw)
        return self.transforms["observable_pca"].transform(scaled).astype(np.float32)

    def decode_amplitude(self, encoded: np.ndarray) -> np.ndarray:
        standardized = self.transforms["hidden_pca"].inverse_transform(encoded)
        log_amplitude = self.transforms["hidden_scaler"].inverse_transform(standardized)
        log_amplitude = np.clip(
            log_amplitude.reshape(len(encoded), len(FIELD_NAMES), 8, len(HIDDEN_MODES)),
            0.0,
            self.transforms["hidden_log_upper"][None, ...],
        )
        return self.transforms["hidden_amplitude_scale"][None, ...] * np.expm1(
            log_amplitude
        )

    def predict_amplitude(
        self,
        visible_coefficients: np.ndarray,
        controls: np.ndarray,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        observable = self.encode_observable(visible_coefficients)
        controls = np.asarray(controls, dtype=np.float32)
        indices = np.arange(self.history_steps - 1, len(observable), dtype=np.int64)
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for begin in range(0, len(indices), batch_size):
                selected = indices[begin : begin + batch_size]
                x = np.stack(
                    [observable[i-self.history_steps+1:i+1] for i in selected]
                )
                u = np.stack(
                    [controls[i-self.history_steps+1:i+1] for i in selected]
                )
                outputs.append(
                    self.model(
                        torch.from_numpy(x).to(self.device),
                        torch.from_numpy(u).to(self.device),
                    ).cpu().numpy()
                )
        return indices, self.decode_amplitude(np.concatenate(outputs))

    def correct_carrier(
        self,
        visible_coefficients: np.ndarray,
        hidden_carrier: np.ndarray,
        controls: np.ndarray,
        batch_size: int = 256,
        conservative_gate: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices, amplitude = self.predict_amplitude(
            visible_coefficients, controls, batch_size=batch_size
        )
        carrier = np.asarray(hidden_carrier)[indices]
        if conservative_gate:
            weight = electric_history_blend_weight(np.asarray(controls)[indices])
            corrected = blend_carrier_and_rom_amplitude(carrier, amplitude, weight)
        else:
            corrected = apply_hidden_amplitude_to_carrier(carrier, amplitude)
        return indices, corrected


def fuse_visible_and_hidden(
    visible_coefficients: np.ndarray, hidden_coefficients: np.ndarray
) -> np.ndarray:
    """Assemble n=0--21 coefficients without altering the observed modes."""
    visible = np.asarray(visible_coefficients)
    hidden = np.asarray(hidden_coefficients)
    if visible.shape[:-1] != hidden.shape[:-1]:
        raise ValueError(f"visible/hidden shapes do not align: {visible.shape}, {hidden.shape}")
    return np.concatenate((visible, hidden), axis=-1)
