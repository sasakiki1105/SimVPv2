"""Band-aggregated tensor state for the adapter arms (tensor_adapter_design_20260928.md §1, §3).

Per frame the R3 extraction cache provides `tensor_features[frame, band, component, stat]`
with components (A, D, Txy, Txz, Tyz) and stats (mean, RMS) (r3_baseline_20260915/common.py).
This module builds, per arm, a per-frame state vector padded to ADAPTER_INPUT_DIM, standardised
with TRAIN-only statistics, and returns aligned 10-frame histories for field windows.

Arms: F0 (no adapter), Fconst (ones), Fnoise (fixed noise by frame, seed), FA (A only), F_mom
(low-order moments used by the hatNG estimator: caller supplies), Fhat (A + hatNG), Fraw (A + NG4),
Fres (A + (NG4 - hatNG)).  raw == hat + residual holds by construction and is asserted.
Frames outside the state cache raise; nothing is imputed.
"""
import numpy as np
import torch

from radaz_adapter_model import ADAPTER_INPUT_DIM

ARMS = ('F0', 'Fconst', 'Fnoise', 'FA', 'F_mom', 'Fhat', 'Fraw', 'Fres')
NOISE_SEED = 20260928


class TensorState:
    def __init__(self, frame, time_us, tensor_features, train_window_us=(12.0, 20.0), hat=None, moments=None, history=10):
        frame = np.asarray(frame, int); tf = np.asarray(tensor_features, float)
        if tf.ndim != 4 or tf.shape[1:] != (4, 5, 2) or len(tf) != len(frame):
            raise ValueError(f'tensor_features must be [n,4,5,2] aligned with frame, got {tf.shape}')
        if not np.all(np.diff(frame) == 1):
            raise ValueError('state frames must be consecutive')
        self.frame0, self.n, self.history = int(frame[0]), len(frame), int(history)
        self.time_us = np.asarray(time_us, float)
        self.train_mask = (self.time_us >= train_window_us[0] - 1e-10) & (self.time_us < train_window_us[1] - 1e-10)
        if self.train_mask.sum() < 2:
            raise ValueError('TRAIN window contains fewer than two frames')
        A = tf[:, :, 0, :].reshape(self.n, 8)
        NG = tf[:, :, 1:, :].reshape(self.n, 32)
        if hat is not None:
            hat = np.asarray(hat, float)
            if hat.shape != NG.shape:
                raise ValueError('hat must match NG4 shape [n,32]')
            res = NG - hat
            np.testing.assert_allclose(hat + res, NG, rtol=0, atol=1e-12 * max(1.0, np.abs(NG).max()))
        noise = np.random.default_rng(NOISE_SEED).standard_normal((self.n, 32))
        raw_states = {'F0': None, 'Fconst': np.ones((self.n, 1)), 'Fnoise': noise, 'FA': A, 'Fraw': np.concatenate([A, NG], 1)}
        if hat is not None:
            raw_states['Fhat'] = np.concatenate([A, hat], 1)
            raw_states['Fres'] = np.concatenate([A, res], 1)
        if moments is not None:
            moments = np.asarray(moments, float)
            if len(moments) != self.n or moments.shape[1] + 8 > ADAPTER_INPUT_DIM:
                raise ValueError('moments must be [n,<=32]')
            raw_states['F_mom'] = np.concatenate([A, moments], 1)
        self.stats, self.states = {}, {}
        for arm, X in raw_states.items():
            if X is None:
                continue
            if arm == 'Fconst':
                m, s = np.zeros(1), np.ones(1)          # constant input is not standardised (it would become all-zero)
            else:
                m = X[self.train_mask].mean(0); s = X[self.train_mask].std(0)
                s = np.where(s > 1e-12 * (np.abs(m) + 1), s, 1.0)
            Z = (X - m) / s
            padded = np.zeros((self.n, ADAPTER_INPUT_DIM)); padded[:, :Z.shape[1]] = Z
            self.stats[arm] = {'mean': m, 'std': s, 'dim': int(Z.shape[1]), 'train_frames': int(self.train_mask.sum())}
            self.states[arm] = padded

    def available_arms(self):
        return ['F0'] + [a for a in ARMS if a in self.states]

    def window(self, arm, first_input_frame):
        """Aligned [history, ADAPTER_INPUT_DIM] state for a field window whose input frames start at first_input_frame."""
        if arm == 'F0':
            raise ValueError('F0 has no adapter input')
        i0 = int(first_input_frame) - self.frame0
        if i0 < 0 or i0 + self.history > self.n:
            raise IndexError(f'frames {first_input_frame}..{first_input_frame + self.history - 1} outside the state cache')
        return torch.from_numpy(self.states[arm][i0: i0 + self.history].astype(np.float32))

    def batch(self, arm, first_input_frames):
        return torch.stack([self.window(arm, f) for f in first_input_frames])
