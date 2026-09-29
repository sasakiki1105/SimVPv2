"""Five-field state: ne, ni, phi, saved-field gamma_corr, electron u_z.

The frozen four-field implementation remains unchanged. Velocity uses one
source-TRAIN RMS in both directions; no per-case scaling or clipping.
"""
from __future__ import annotations

import numpy as np
import torch

import radaz_transport_channels as old
import radaz_transport_output as v2
from openstl.models.simvp_factory import build_simvp_model

ROOT = old.ROOT
OUT = ROOT/'workdirs/2D_RadAz/radaz_transport_velocity_20260929'
PLAN = ROOT/'protocols/radaz_transport_velocity_20260929'
ARMS = ('corr_uez',)


def load_frames(case, start, stop, contract):
    """Return normalized [ne,ni,phi,u_ez], at exactly the same saved times."""
    scale = float(contract['velocity_rms'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid source TRAIN velocity scale')
    metadata_case = dict(case, channels=['electron_den','ion_den','phi','electron_wd'],
        normalization_low=list(contract['manifest']['normalization']['low'])+[0.],
        normalization_high=list(contract['manifest']['normalization']['high'])+[scale],
        normalization_clip=False)
    meta = old._radaz_consolidated_metadata(case['path'], metadata_case)
    return old._load_radaz_consolidated_segment(case['path'],meta,start,stop)


def make_target(frames, case, contract, arm='corr_uez'):
    if arm not in ARMS or frames.ndim != 4 or frames.shape[1:] != (4,260,256):
        raise ValueError('Expected normalized ne,ni,phi,u_ez on the native padded grid')
    if not np.isfinite(frames).all():
        raise ValueError('Nonfinite state')
    state = v2.make_target(frames[:,:3],case,contract,'corr')
    return np.concatenate((state,frames[:,3:4]),axis=1)


def make_input(frames, case, contract, arm='corr_uez'):
    state = make_target(frames,case,contract,arm)
    vector = old._condition_values(case,contract['manifest']['condition_channels'],
        normalization=contract['manifest']['condition_normalization'])
    cond = np.broadcast_to(vector[None,:,None,None],(len(frames),2,260,256))
    return np.concatenate((state,cond),axis=1).astype(np.float32,copy=False)


class TransportWindows(old.TransportWindows):
    def __init__(self, contract, split, arm='corr_uez', preloaded=None):
        start,stop = contract['splits'][split]
        if preloaded is None:
            preloaded=[]
            for case in contract['manifest']['cases']:
                print('LOAD',split,case['case_key'],start,stop,flush=True)
                preloaded.append(load_frames(case,start,stop,contract))
        if any(d.shape[1:] != (4,260,256) for d in preloaded):
            raise ValueError('Four stored fields required, including velocity')
        super().__init__(contract,split,arm,preloaded=preloaded)

    def __getitem__(self,index):
        i,s=self.samples[index]
        # Each call sees its own time window; labels cannot change past inputs.
        return (make_input(self.data[i][s:s+self.pre],self.cases[i],self.contract,self.arm),
                make_target(self.data[i][s+self.pre:s+self.pre+self.aft],
                            self.cases[i],self.contract,self.arm))


def model_config(contract,arm='corr_uez'):
    if arm not in ARMS:
        raise ValueError('Only the combined five-channel model is scheduled')
    cfg = v2.model_config(contract,'corr')
    cfg.update(in_shape=(contract['pre'],7,260,256),out_channels=5)
    return cfg


def matched_model(contract,arm,seed):
    base = v2.matched_model(contract,'corr',seed)
    target = build_simvp_model(model_config(contract,arm))
    source,state = base.state_dict(),target.state_dict()
    changed=[]
    for key,value in state.items():
        if value.shape == source[key].shape:
            value.copy_(source[key])
        elif key in ('dec.readout.weight','dec.readout.bias'):
            if value.shape[0]!=5 or source[key].shape[0]!=4:
                raise ValueError('Unexpected readout shape')
            value.zero_(); value[:4].copy_(source[key]); changed.append(key)
        elif value.ndim==4 and value.shape[1]==5 and source[key].shape[1]==4:
            value.zero_(); value[:,:4].copy_(source[key]); changed.append(key)
        else:
            raise ValueError('Unexpected architecture change: '+key)
    if len(changed)!=3:
        raise ValueError('Expected one input slice and readout weight/bias')
    target.load_state_dict(state,strict=True)
    return target


def loss_parts(pred,target,gamma_weight=1/3,velocity_weight=1/3):
    if pred.shape!=target.shape or pred.ndim!=5 or pred.shape[2]!=5:
        raise ValueError('Expected matching five-field predictions and labels')
    if not np.isfinite(velocity_weight) or velocity_weight<=0:
        raise ValueError('Invalid velocity weight')
    parts=v2.loss_parts(pred[:,:,:4],target[:,:,:4],gamma_weight)
    velocity=torch.nn.functional.mse_loss(pred[:,:,4,:257,:256],target[:,:,4,:257,:256])
    return dict(parts,total=parts['total']+velocity_weight*velocity,velocity=velocity)


def checked_step(model,optimizer,x,y,gamma_weight=1/3,velocity_weight=1/3,check_gradients=False):
    optimizer.zero_grad(set_to_none=True)
    parts=loss_parts(model(x),y,gamma_weight,velocity_weight)
    if not all(torch.isfinite(v) for v in parts.values()):
        raise RuntimeError('Nonfinite training loss')
    parts['total'].backward()
    if check_gradients and any(p.grad is not None and not torch.isfinite(p.grad).all()
                              for p in model.parameters()):
        raise RuntimeError('Nonfinite gradient')
    optimizer.step()
    return {k:float(v.detach()) for k,v in parts.items()}


@torch.inference_mode()
def autonomous_rollout(model,initial_input,blocks,valid_height=257):
    if model.training or isinstance(blocks,bool) or not isinstance(blocks,int) or blocks<1:
        raise ValueError('An eval model and positive integer block count are required')
    if initial_input.ndim!=5 or initial_input.shape[2]!=7:
        raise ValueError('Five physical channels and two fixed conditions required')
    batch,steps,_,height,width=initial_input.shape
    if not 0<valid_height<=height or not torch.isfinite(initial_input).all():
        raise ValueError('Invalid initial state')
    cond=initial_input[:,:,-2:]; constant=cond[:,:1,:,:1,:1]
    if not torch.allclose(cond,constant.expand_as(cond),rtol=0,atol=1e-6):
        raise ValueError('Conditions must be fixed in time and space')
    result=[]; x=initial_input
    for block in range(blocks):
        p=model(x)
        if p.shape!=(batch,steps,5,height,width) or not torch.isfinite(p).all():
            raise RuntimeError(f'Invalid autonomous prediction at block {block+1}')
        if height>valid_height:
            tail=p[...,valid_height-1:valid_height,:].expand(*p.shape[:-2],height-valid_height,width)
            p=torch.cat((p[...,:valid_height,:],tail),dim=-2)
        result.append(p.cpu())
        x=torch.cat((p,constant.expand(batch,steps,2,height,width)),dim=2)
    return torch.cat(result,dim=1)


def diagnose(pred,true_fields,case,contract,arm='corr_uez'):
    if arm not in ARMS or pred.shape[1]!=5 or true_fields.shape[1]!=4:
        raise ValueError('Expected five predictions and four stored fields')
    metrics,arrays=v2.diagnose(pred[:,:4],true_fields[:,:3],case,contract,'corr')
    error=pred[:,4,:257].astype(float)-true_fields[:,3,:257]
    metrics.update(velocity_normalized_mse=float(np.square(error).mean()),
                   velocity_rmse_m_s=float(np.sqrt(np.square(error).mean())*contract['velocity_rms']))
    return metrics,arrays
