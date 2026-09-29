"""Four-field transport prediction and autonomous feedback (separate v2).

Frozen input-only code is reused without modification. Raw/corr now predict
[ne, ni, phi, gamma]; none remains the three-field control. The gamma output
uses exactly the same TRAIN scale as the gamma input.
"""
from __future__ import annotations

import numpy as np
import torch

import radaz_transport_channels as old
from openstl.models.simvp_factory import build_simvp_model

ROOT = old.ROOT
OUT = ROOT/'workdirs/2D_RadAz/radaz_transport_output_20260929'
PLAN = ROOT/'protocols/radaz_transport_output_20260929'
ARMS = old.ARMS


def make_target(frames, case, contract, arm):
    """Future gamma is a LABEL only; make_input is called separately on past."""
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    if arm == 'none':
        return np.asarray(frames, dtype=np.float32).copy()
    return old.make_input(frames, case, contract, arm)[:, :4].copy()


class TransportWindows(old.TransportWindows):
    def __getitem__(self, index):
        x, fields = super().__getitem__(index)
        i, _ = self.samples[index]
        return x, make_target(fields, self.cases[i], self.contract, self.arm)


def model_config(contract, arm):
    cfg = old.model_config(contract, arm)
    cfg['out_channels'] = 3 if arm == 'none' else 4
    return cfg


def matched_model(contract, arm, seed):
    source_model = old.matched_model(contract, arm, seed)
    if arm == 'none':
        return source_model
    target = build_simvp_model(model_config(contract, arm))
    source, state = source_model.state_dict(), target.state_dict()
    changed = []
    for key, value in state.items():
        if value.shape == source[key].shape:
            value.copy_(source[key])
        elif key in ('dec.readout.weight', 'dec.readout.bias'):
            if source[key].shape[0] != 3 or value.shape[0] != 4:
                raise ValueError('Unexpected decoder readout')
            value.zero_()
            value[:3].copy_(source[key])
            changed.append(key)
        else:
            raise ValueError('Unexpected output architecture change: '+key)
    if set(changed) != {'dec.readout.weight', 'dec.readout.bias'}:
        raise ValueError('Expected one new readout channel')
    target.load_state_dict(state, strict=True)
    return target


def loss_parts(pred, target, gamma_weight=1/3):
    """Keep the old three-field objective's coefficient unchanged.

    L = mean(MSE_ne, MSE_ni, MSE_phi) + (1/3) MSE_gamma.
    Gamma MSE excludes padding. No consistency loss or future feedback.
    """
    if pred.shape != target.shape or pred.ndim != 5 or pred.shape[2] not in (3,4):
        raise ValueError('Expected matching [B,T,3 or 4,H,W]')
    if not np.isfinite(gamma_weight) or gamma_weight <= 0:
        raise ValueError('Gamma weight must be finite and positive')
    field = torch.nn.functional.mse_loss(pred[:,:,:3], target[:,:,:3])
    gamma = (torch.nn.functional.mse_loss(pred[:,:,3,:257,:256], target[:,:,3,:257,:256])
             if pred.shape[2] == 4 else pred.new_zeros(()))
    return {'total':field+gamma_weight*gamma, 'field':field, 'gamma':gamma}


def checked_step(model, optimizer, x, y, gamma_weight, check_gradients=False):
    optimizer.zero_grad(set_to_none=True)
    parts = loss_parts(model(x), y, gamma_weight)
    if not all(torch.isfinite(v) for v in parts.values()):
        raise RuntimeError('Nonfinite training loss')
    parts['total'].backward()
    if check_gradients and any(p.grad is not None and not torch.isfinite(p.grad).all()
                               for p in model.parameters()):
        raise RuntimeError('Nonfinite gradient')
    optimizer.step()
    return {k:float(v.detach()) for k,v in parts.items()}


@torch.inference_mode()
def autonomous_rollout(model, initial_input, blocks, valid_height=257):
    """Only initial history and fixed known conditions; NO future truth argument.

    Predictions of ALL physical channels, including gamma, feed the next block.
    Only invalid radial padding is replaced with the final valid row. Gamma
    on physical nodes is never recomputed, projected, clipped or normalized anew.
    Returns CPU predictions to avoid holding a long trajectory on the GPU.
    """
    if model.training:
        raise ValueError('Call model.eval() before autonomous rollout')
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
        raise ValueError('blocks must be a positive integer')
    if initial_input.ndim != 5 or initial_input.shape[2] not in (5,6):
        raise ValueError('Expected three/four physical fields plus two conditions')
    batch, steps, channels, height, width = initial_input.shape
    if not 0 < valid_height <= height or not torch.isfinite(initial_input).all():
        raise ValueError('Invalid initial state/valid height')
    cond = initial_input[:,:,-2:]
    constant = cond[:,:1,:,:1,:1]
    if not torch.allclose(cond, constant.expand_as(cond), rtol=0, atol=1e-6):
        raise ValueError('This rollout requires fixed E/B within each trajectory')
    x = initial_input
    result = []
    for block in range(blocks):
        p = model(x)
        if p.shape != (batch,steps,channels-2,height,width):
            raise RuntimeError('Rollout requires matching input/output physical state and time length')
        if not torch.isfinite(p).all():
            raise RuntimeError(f'Nonfinite autonomous prediction at block {block+1}')
        if height > valid_height:
            tail = p[...,valid_height-1:valid_height,:].expand(*p.shape[:-2],height-valid_height,width)
            p = torch.cat((p[...,:valid_height,:],tail),dim=-2)
        result.append(p.cpu())
        x = torch.cat((p,constant.expand(batch,steps,2,height,width)),dim=2)
    return torch.cat(result,dim=1)


def direct_full_flux(pred, case, contract, arm):
    """Average the *predicted gamma*, not a recomputation from ne/phi.

    Only full flux is defined this way. A spatial FFT of gamma does NOT recover
    the density--electric-field cross-spectrum's T10 decomposition.
    """
    if arm == 'none' or pred.shape[-3] != 4:
        raise ValueError('Direct gamma requires a four-field prediction')
    image = np.asarray(pred[...,3,:257,:256],dtype=np.float64)*contract['gamma_rms'][arm]
    pool = old.band_pool(np.arange(257)*(case['Ly_m']/256))
    return np.einsum('rh,...h->...r',pool,image.mean(axis=-1))


def diagnose(pred, true_fields, case, contract, arm):
    """Field-derived spectra, direct flux, and local consistency kept separate."""
    if pred.ndim != 4 or true_fields.ndim != 4 or pred.shape[0] != len(true_fields):
        raise ValueError('Expected [time,channels,H,W]')
    expected_channels = 3 if arm == 'none' else 4
    if pred.shape[1] != expected_channels or true_fields.shape[1] != 3:
        raise ValueError('Incorrect observable channels')
    fields = pred[:,:3]
    po, to = old.observables(fields,case,contract), old.observables(true_fields,case,contract)
    arrays = {f'{prefix}_{k}':v for prefix,obs in (('derived',po),('truth',to)) for k,v in obs.items()}
    field_mse = np.square(fields[:,:,:257].astype(float)-true_fields[:,:,:257]).mean((0,2,3))
    metrics = {'field_mse_by_channel':field_mse.tolist()}
    if arm != 'none':
        true_state = make_target(true_fields,case,contract,arm)
        derived_state = make_target(fields,case,contract,arm)
        direct = pred[:,3,:257,:256].astype(float)
        truth_gamma = true_state[:,3,:257,:256].astype(float)
        derived_gamma = derived_state[:,3,:257,:256].astype(float)
        metrics.update(direct_gamma_normalized_mse=float(np.square(direct-truth_gamma).mean()),
            derived_gamma_normalized_mse=float(np.square(derived_gamma-truth_gamma).mean()),
            direct_derived_local_normalized_mse=float(np.square(direct-derived_gamma).mean()))
        arrays['direct_full'] = direct_full_flux(pred,case,contract,arm)
        arrays['direct_minus_derived_full'] = arrays['direct_full']-po['full']
        metrics['direct_derived_band_rmse'] = float(np.sqrt(np.square(arrays['direct_minus_derived_full']).mean()))
    return metrics, arrays
