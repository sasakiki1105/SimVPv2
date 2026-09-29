"""Explicit saved-field transport inputs, isolated from frozen RadAz workflows.

Channels are [ne, ni, phi, (gamma), log_vE, log_n0]; only three fields are
predicted. Gamma is computed from the exact float32 input fields, before
padding, using the periodic central difference. No future frame is needed.
"""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.datasets.dataloader_pepapic_h5 import (
    _condition_values, _radaz_consolidated_metadata, _load_radaz_consolidated_segment,
)
from openstl.models.simvp_factory import build_simvp_model
from radaz_metrics_v3 import band_pool, local_observables

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'protocols/radaz_transport_channels_20260929'
OUT = ROOT / 'workdirs/2D_RadAz/radaz_transport_channels_20260929'
SOURCE = ROOT / 'workdirs/2D_RadAz/radaz_paired_pilot_v2/source_manifest.json'
BASE_CONFIG = ROOT / 'configs/custom/pepapic/SimVP_gSTA_radaz_bc_Donly_B_60ep.py'
ARMS = ('none', 'raw', 'corr')
SPLITS = {'train': [0, 1600], 'val': [1600, 1800], 'test': [1800, 2000]}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def validate_manifest(manifest):
    if manifest['condition_channels'] != ['log_vE', 'log_n0']:
        raise ValueError('Expected the two source-normalized E/B coordinates')
    if manifest['normalization'].get('clip', False):
        raise ValueError('Clipped fields are not the registered input representation')
    keys = [c['case_key'] for c in manifest['cases']]
    if len(keys) != len(set(keys)) or not keys:
        raise ValueError('Empty or duplicate cases')
    for c in manifest['cases']:
        if c.get('role') != 'source':
            raise ValueError('Preparation/training accepts source cases only')
        if c['channels'] != ['electron_den', 'ion_den', 'phi'] or c.get('spatial_stride', 1) != 1:
            raise ValueError('This first comparison uses native fine three-field inputs')
        if (c['model_height'], c['model_width']) != (260, 256):
            raise ValueError('Expected 257 valid radial nodes padded to 260, width 256')
        if c.get('normalization_clip', False) or c['B_mT'] <= 0 or c['Ez_kVm'] <= 0:
            raise ValueError('Invalid source metadata')
        if not np.isclose(c['dt_frame_ns'], 15) or not np.isclose(c['Ly_m'], .0128):
            raise ValueError('Unexpected temporal cadence/domain')
        for key in ('low', 'high'):
            np.testing.assert_array_equal(c['normalization_' + key], manifest['normalization'][key])


def physical_fields(frames, manifest):
    """[...,3,H,W] normalized -> float64 physical values, without padding."""
    if frames.shape[-3] != 3:
        raise ValueError('Expected three field channels')
    low = np.asarray(manifest['normalization']['low'], dtype=np.float64)
    span = np.asarray(manifest['normalization']['high'], dtype=np.float64) - low
    return np.asarray(frames[..., :257, :256], dtype=np.float64) * span[:, None, None] + low[:, None, None]


def transport_image(physical, dy, magnetic_t, arm):
    """[...,3,x,y] -> [...,x,y], signed particle-flux dimensions."""
    if arm not in ('raw', 'corr') or dy <= 0 or magnetic_t <= 0:
        raise ValueError('Invalid transport definition/grid/B')
    p = np.asarray(physical, dtype=np.float64)
    if p.shape[-3] != 3 or not np.isfinite(p).all():
        raise ValueError('Invalid physical fields')
    ne, phi = p[..., 0, :, :], p[..., 2, :, :]
    ey = -(np.roll(phi, -1, axis=-1) - np.roll(phi, 1, axis=-1)) / (2 * dy)
    if arm == 'corr':
        ne = ne - ne.mean(axis=-1, keepdims=True)
        ey = ey - ey.mean(axis=-1, keepdims=True)
    return -ne * ey / magnetic_t


def make_input(frames, case, contract, arm):
    """Build inputs from PAST frames only. Target construction is separate."""
    if arm not in ARMS or frames.ndim != 4 or frames.shape[1:] != (3, 260, 256):
        raise ValueError('Expected [T,3,260,256] and a registered arm')
    manifest = contract['manifest']
    parts = [np.asarray(frames, dtype=np.float32)]
    if arm != 'none':
        gamma = transport_image(physical_fields(frames, manifest), case['Ly_m']/256,
                                case['B_mT']*.001, arm)
        scale = float(contract['gamma_rms'][arm])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid TRAIN gamma RMS')
        # Replicate the last valid radial row exactly as for the three fields.
        gamma = np.pad(gamma / scale, ((0, 0), (0, 3), (0, 0)), mode='edge')
        parts.append(gamma[:, None].astype(np.float32))
    vector = _condition_values(case, manifest['condition_channels'],
                               normalization=manifest['condition_normalization'])
    cond = np.broadcast_to(vector[None, :, None, None], (len(frames), 2, 260, 256))
    parts.append(cond)
    result = np.concatenate(parts, axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite model input')
    return result


def load_frames(case, start, stop):
    meta = _radaz_consolidated_metadata(case['path'], case)
    return _load_radaz_consolidated_segment(case['path'], meta, start, stop)


class TransportWindows(Dataset):
    """Frame-disjoint source windows. In-memory fields, gamma derived per input.

    Explicit ranges avoid the legacy 2001st frame changing the test sample count.
    Training does not instantiate/load any TEST dataset.
    """
    def __init__(self, contract, split, arm, preloaded=None):
        self.contract, self.arm = contract, arm
        self.cases = contract['manifest']['cases']
        self.pre, self.aft = contract['pre'], contract['aft']
        start, stop = contract['splits'][split]
        self.data = (preloaded if preloaded is not None else
                     [load_frames(c, start, stop) for c in self.cases])
        if len(self.data) != len(self.cases) or any(len(d) != stop-start for d in self.data):
            raise ValueError('Preloaded frames do not match the split')
        self.samples = [(i, s) for i in range(len(self.cases))
                        for s in range(stop-start-self.pre-self.aft+1)]
        self.frame_start = start
        if not self.samples:
            raise ValueError('No complete windows inside split')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        i, s = self.samples[index]
        x = make_input(self.data[i][s:s+self.pre], self.cases[i], self.contract, self.arm)
        y = self.data[i][s+self.pre:s+self.pre+self.aft]
        return x, y


def model_config(contract, arm):
    cfg = {k: v for k, v in runpy.run_path(str(BASE_CONFIG)).items() if not k.startswith('__')}
    cfg.update(in_shape=(contract['pre'], 5 + (arm != 'none'), 260, 256),
               out_channels=3, condition_dim=2, condition_film=True,
               aft_seq_length=contract['aft'], radaz_validation_diagnostics=False)
    return cfg


def matched_model(contract, arm, seed):
    """Identical shared tensors; additional input weights initially zero.

    All arms have byte-identical predictions before training. Initialization of
    unrelated parameters and epoch sample order cannot depend on extra channels.
    """
    torch.manual_seed(seed)
    base = build_simvp_model(model_config(contract, 'none'))
    if arm == 'none':
        return base
    target = build_simvp_model(model_config(contract, arm))
    source, state = base.state_dict(), target.state_dict()
    changed = []
    for key, value in state.items():
        if value.shape == source[key].shape:
            value.copy_(source[key])
        elif value.ndim == 4 and source[key].shape[1] == 3 and value.shape[1] == 4:
            value.zero_()
            value[:, :3].copy_(source[key])
            changed.append(key)
        else:
            raise ValueError('Unexpected architecture difference: ' + key)
    if len(changed) != 1:
        raise ValueError('Expected exactly one extra encoder input slice')
    target.load_state_dict(state, strict=True)
    return target


def observables(frames, case, contract):
    """Per-frame full flux and fixed-T10 flux; full modal arrays for audits."""
    physical = physical_fields(frames, contract['manifest'])
    leading = physical.shape[:-3]
    flat = physical.reshape(-1, 3, 257, 256)
    pool = band_pool(np.arange(257) * (case['Ly_m']/256))
    obs = local_observables(flat[None], pool, case['Ly_m']/256, case['B_mT']*.001)
    gamma = obs['gamma'][0].reshape(*leading, 4, 64)
    full = obs['flux_full'][0].reshape(*leading, 4)
    t10 = np.stack([gamma[..., b, np.asarray(contract['t10'][str(b)])-1].sum(-1)
                    for b in range(4)], axis=-1)
    result = {'full': full, 't10': t10, 'modal': gamma}
    for name in ('pn', 'pe', 'cross'):
        result[name] = obs[name][0].reshape(*leading, 4, 64)
    return result


def forecast_windows(series, pre=10, aft=10):
    """No overlap across splits: callers pass ONE disjoint segment."""
    n = len(series)-pre-aft+1
    if n <= 0:
        raise ValueError('Segment too short')
    x = np.stack([series[s:s+pre].reshape(-1) for s in range(n)])
    y = np.stack([series[s+pre:s+pre+aft].mean(0) for s in range(n)])
    return x, y


def fit_ridge(x, y, xv, yv):
    """Direct future-window AR10, all scalers TRAIN-only, alpha VAL-only."""
    xm, ym = x.mean(0), y.mean(0)
    xs, ys = x.std(0), y.std(0)
    xs = np.where(xs > 1e-12, xs, 1.)
    ys = np.where(ys > 1e-12, ys, 1.)
    z, target = (x-xm)/xs, (y-ym)/ys
    zv, tv = (xv-xm)/xs, (yv-ym)/ys
    gram, rhs = z.T @ z / len(z), z.T @ target / len(z)
    best = None
    for alpha in (1e-4, 1e-3, .01, .1, 1., 10., 100.):
        weights = np.linalg.solve(gram + alpha*np.eye(gram.shape[0]), rhs)
        score = float(np.mean((zv @ weights-tv)**2))
        if best is None or score < best['validation_scaled_mse']:
            best = dict(alpha=alpha, weights=weights.tolist(), x_mean=xm.tolist(),
                        x_scale=xs.tolist(), y_mean=ym.tolist(), y_scale=ys.tolist(),
                        validation_scaled_mse=score)
    return best


def predict_ridge(fit, x):
    return ((x-np.asarray(fit['x_mean']))/np.asarray(fit['x_scale']) @
            np.asarray(fit['weights'])) * np.asarray(fit['y_scale']) + np.asarray(fit['y_mean'])


def score_transport(pred, truth, null):
    error, ref = (pred-truth)**2, (null-truth)**2
    den = ref.sum()
    by_band = [float(1-a/b) if b > 0 else None for a, b in zip(error.sum(0), ref.sum(0))]
    return {'skill': float(1-error.sum()/den) if den > 0 else None,
            'skill_by_band': by_band, 'rmse': float(np.sqrt(error.mean())),
            'bias_by_band': (pred-truth).mean(0).tolist(),
            'sse_by_band': error.sum(0).tolist(), 'null_sse_by_band': ref.sum(0).tolist()}


def verify_prepared(folder=OUT):
    folder = Path(folder)
    bundle = read_json(folder/'bundle.json')
    for name, value in bundle['assets_sha256'].items():
        if digest(name) != value:
            raise RuntimeError('Prepared asset changed; make a new version: ' + name)
    for name, value in bundle['data_identity'].items():
        stat = Path(name).stat()
        if [stat.st_size, stat.st_mtime_ns] != value:
            raise RuntimeError('Source data size/mtime changed: ' + name)
    contract = read_json(folder/'contract.json')
    return contract, bundle
