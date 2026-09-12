"""GPU feasibility on one SOURCE-TRAIN window; ephemeral models, no checkpoint.

Five measured optimizer steps after two warmup steps; nothing is selected from
validation/test. One identical float32 recipe for A/B/C.
"""
import gc
import json
import os
import runpy
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import torch
from openstl.methods.simvp import SimVP
from evaluate_radaz_conditioned_factorial import condition_vector, load_case_frames
from run_radaz_paired_pilot import ROOT, verify_bundle, digest


def main():
    bundle = verify_bundle()
    out = ROOT.parent/'research_results/audits/bc_temporal_preflight_20260912/training_smoke.json'
    if out.exists():
        raise RuntimeError('Completed smoke exists; do not silently overwrite')
    torch.set_num_threads(4)
    manifest = json.loads(open(bundle['manifest'], encoding='utf-8').read())
    c = manifest['cases'][0]
    low, high = (np.asarray(manifest['normalization'][k]) for k in ('low','high'))
    frames = load_case_frames(c, low, high, 1400, 1420)[0]
    x = np.empty((1,10,5,260,256), np.float32)
    x[0,:,:3] = frames[:10]
    x[0,:,3:] = condition_vector(c, manifest)[0][None,:,None,None]
    x, y = torch.from_numpy(x).cuda(), torch.from_numpy(frames[10:][None]).cuda()
    rows = {}
    base = {k:v for k,v in runpy.run_path(str(ROOT/bundle['configs']['D'])).items() if not k.startswith('__')}
    for cell in ('A','B','C'):
        path = ROOT/(bundle['configs']['D'] if cell == 'A' else 'configs/custom/pepapic/SimVP_gSTA_radaz_bc_Donly_'+cell+'_60ep.py')
        cfg = {k:v for k,v in runpy.run_path(str(path)).items() if not k.startswith('__')}
        differences = {k:[base.get(k),v] for k,v in cfg.items() if v != base.get(k)}
        expected = {'A':set(), 'B':{'spatio_azimuth_downsample'}, 'C':{'hid_S'}}[cell]
        if set(differences) != expected:
            raise RuntimeError('Unexpected D configuration change: '+str(differences))
        cfg.update(in_shape=(10,5,260,256), dataname='pepapic_h5', data_root=bundle['manifest'])
        torch.manual_seed(42)
        model = SimVP(**cfg).cuda().train()
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.model.hid.modules())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        torch.cuda.reset_peak_memory_stats()
        times, losses, shapes = [], [], []
        hook = model.model.enc.register_forward_hook(lambda mod, args, output: shapes.append(list(output[0].shape)))
        for step in range(7):
            torch.cuda.synchronize()
            began = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, data, _, _, auxiliary = model._total_loss(pred,y,batch_x=x)
            assert auxiliary is None
            torch.testing.assert_close(loss,data,rtol=0,atol=0)
            assert pred.shape == y.shape and torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic()-began
            if step >= 2:
                times.append(elapsed)
            losses.append(float(loss.detach()))
        hook.remove()
        with torch.no_grad():
            before = model.train()(x)
            after = model.eval()(x)
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        rows[cell] = {'config_sha256':digest(path), 'parameters':sum(p.numel() for p in model.parameters()),
                     'changes_from_D':differences, 'encoder_output':shapes[0],
                     'measured_step_seconds':times, 'median_step_seconds':float(np.median(times)),
                     'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                     'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
                     'finite_gradients':True, 'train_eval_agree':True, 'smoke_losses':losses}
        print(cell, json.dumps(rows[cell]),flush=True)
        del before,after,model,optimizer,pred,loss,data
        gc.collect()
        torch.cuda.empty_cache()
    out.write_text(json.dumps({'kind':'ephemeral_train_window_smoke_not_research_training',
        'source_case':c['case_key'], 'source_train_frames':[1400,1420], 'precision':'float32',
        'cells':rows, 'timing_caveat':'includes explicit gradient-finiteness checks; excludes full data loading, validation and checkpoint I/O'},indent=2),encoding='utf-8')
    verify_bundle()


if __name__ == '__main__':
    main()
