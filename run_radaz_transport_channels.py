"""Local transport-channel smoke, training/resume, evaluation and CLI status.

No automatic remote access or hidden job submission. ``train --arm all`` runs
the six registered jobs sequentially in this process. Training reads TRAIN/VAL
only; TEST is a separate explicit evaluation command.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from radaz_transport_channels import (
    ROOT, OUT, PLAN, ARMS, read_json, digest, atomic_json, verify_prepared,
    TransportWindows, load_frames, make_input, model_config, matched_model,
    observables, forecast_windows, predict_ridge, score_transport, _condition_values,
    physical_fields, transport_image,
)


def now():
    return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def exclusive(path):
    """OS lock is released on crash; a stale file alone cannot block recovery."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def epoch_order(length, seed, epoch):
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed+1000003*epoch))


class OrderedSampler(Sampler):
    def __init__(self, size, seed):
        self.size, self.seed, self.epoch = size, seed, 0

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(epoch_order(self.size, self.seed, self.epoch).tolist())


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def checked_step(model, optimizer, x, y, check_gradients=False):
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    if pred.shape != y.shape:
        raise RuntimeError('Three-field output shape mismatch')
    loss = torch.nn.functional.mse_loss(pred, y)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss')
    loss.backward()
    if check_gradients and any(p.grad is not None and not torch.isfinite(p.grad).all()
                               for p in model.parameters()):
        raise RuntimeError('Nonfinite gradient')
    optimizer.step()
    return float(loss.detach())


def smoke(folder, device):
    """Full-size three-arm TRAIN-only check; does not select any model."""
    contract = read_json(folder/'contract.json')
    if (folder/'smoke.json').exists():
        raise RuntimeError('Smoke result already exists; do not overwrite')
    case = contract['manifest']['cases'][0]
    frames = load_frames(case, 1400, 1420)
    y = torch.from_numpy(frames[10:][None]).to(device)
    results, initial = {}, None
    for arm in ARMS:
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        model = matched_model(contract, arm, 42).to(device)
        x = torch.from_numpy(make_input(frames[:10],case,contract,arm)[None]).to(device)
        model.eval()
        with torch.no_grad():
            before = model(x).cpu()
        if initial is None:
            initial = before
        else:
            torch.testing.assert_close(before, initial, rtol=1e-5, atol=2e-6)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        begin = time.monotonic()
        losses = [checked_step(model,optimizer,x,y,True) for _ in range(3)]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = (time.monotonic()-begin)/3
        gamma_grad = None
        if arm != 'none':
            first = next(p for p in model.parameters() if p.ndim == 4 and p.shape[1] == 4)
            gamma_grad = float(first.grad[:,3].abs().sum())
            if not gamma_grad > 0:
                raise RuntimeError('Fourth channel has no learning signal')
        model.eval()
        with torch.no_grad():
            prediction = model(x).cpu().numpy()
        checkpoint = folder/'smoke_checkpoints'/f'{arm}.pt'
        save_checkpoint(checkpoint, dict(kind='technical_smoke_not_research_checkpoint',
            model=model.state_dict(), arm=arm, contract_sha256=digest(folder/'contract.json')))
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model'], strict=True)
        with torch.no_grad():
            np.testing.assert_array_equal(model(x).cpu().numpy(), prediction)
        # Exercise the real evaluator's physical reconstruction and fixed T10.
        obs = observables(prediction[0],case,contract)
        if not all(np.isfinite(v).all() for v in obs.values()):
            raise RuntimeError('Nonfinite transport diagnostics')
        results[arm] = dict(input_shape=list(x.shape), output_shape=list(prediction.shape),
            parameters=sum(p.numel() for p in model.parameters()), losses=losses,
            mean_step_seconds=seconds, gamma_weight_gradient_l1=gamma_grad,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None,
            checkpoint_roundtrip_exact=True, transport_diagnostics_finite=True)
        print('SMOKE',arm,results[arm],flush=True)
        del model, optimizer, x, prediction, before
        gc.collect()
    atomic_json(folder/'smoke.json',dict(status='passed',created_utc=now(),
        purpose='engineering_only_not_forecast_skill', source=case['case_key'],frames=[1400,1420],
        device=str(device), torch=torch.__version__, numpy=np.__version__, initial_predictions_equal=True,
        code_sha256={str(ROOT/n):digest(ROOT/n) for n in ('radaz_transport_channels.py',
            'prepare_radaz_transport_channels.py','run_radaz_transport_channels.py')}, arms=results))


def train_job(folder, arm, seed, device, datasets=None):
    contract, bundle = verify_prepared(folder)
    job = arm+str(seed)
    path = folder/'jobs'/job
    path.mkdir(parents=True, exist_ok=True)
    ck_path = path/'last.pt'
    identity = dict(job=job, arm=arm, seed=seed, bundle_sha256=digest(folder/'bundle.json'))
    model = matched_model(contract,arm,seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=contract['training']['lr'], weight_decay=0.)
    if datasets is None:
        train = TransportWindows(contract,'train',arm)
        val = TransportWindows(contract,'val',arm)
    else:
        train = TransportWindows(contract,'train',arm,preloaded=datasets[0])
        val = TransportWindows(contract,'val',arm,preloaded=datasets[1])
    sampler = OrderedSampler(len(train),seed)
    loader = DataLoader(train,batch_size=1,sampler=sampler,num_workers=0,pin_memory=device.type=='cuda',
                        generator=torch.Generator().manual_seed(2000003+seed))
    val_loader = DataLoader(val,batch_size=1,shuffle=False,num_workers=0)
    epochs = contract['training']['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=contract['training']['lr'],
        total_steps=len(loader)*epochs,pct_start=contract['training']['pct_start'])
    first, global_step = 0, 0
    if ck_path.exists():
        ck = torch.load(ck_path,map_location=device,weights_only=False)
        if ck['identity'] != identity:
            raise RuntimeError('Checkpoint identity differs')
        model.load_state_dict(ck['model'],strict=True)
        optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        first, global_step = ck['completed_epochs'], ck['global_step']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    for epoch in range(first, epochs):
        model.train(); sampler.epoch=epoch
        begin, total = time.monotonic(), 0.
        for step,(x,y) in enumerate(loader):
            loss = checked_step(model,optimizer,x.to(device),y.to(device))
            scheduler.step(); total += loss; global_step += 1
            if step == 0 or (step+1)%50 == 0 or step+1 == len(loader):
                elapsed = time.monotonic()-begin
                progress = dict(**identity,status='training',epoch=epoch+1,epochs=epochs,
                    step=step+1,steps_per_epoch=len(loader),global_step=global_step,loss=loss,
                    mean_step_seconds=elapsed/(step+1),updated_utc=now(),
                    remaining_epoch_seconds=elapsed/(step+1)*(len(loader)-step-1))
                atomic_json(path/'status.json',progress)
                print(job,f'epoch {epoch+1}/{epochs} step {step+1}/{len(loader)} loss {loss:.6g}',flush=True)
        model.eval(); val_sum=0.
        with torch.no_grad():
            for x,y in val_loader:
                val_sum += float(torch.nn.functional.mse_loss(model(x.to(device)),y.to(device)))
        save_checkpoint(ck_path,dict(kind='transport_channel_training',identity=identity,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            completed_epochs=epoch+1,global_step=global_step,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            model_config=model_config(contract,arm)))
        row=dict(**identity,completed_epochs=epoch+1,train_mse=total/len(loader),
                 val_mse=val_sum/len(val_loader),elapsed_seconds=time.monotonic()-begin,updated_utc=now())
        with (path/'epochs.jsonl').open('a',encoding='utf-8') as handle:
            import json
            handle.write(json.dumps(row)+'\n')
        atomic_json(path/'status.json',dict(**row,status='complete' if epoch+1==epochs else 'epoch_saved'))
        if (folder/'PAUSE_AFTER_EPOCH').exists():
            atomic_json(path/'status.json',dict(**row,status='paused'))
            return False
    return True


def evaluate(folder, arm, seed, split, device, transfer_manifest=None):
    contract, _ = verify_prepared(folder)
    path = folder/'jobs'/(arm+str(seed))
    ck_path = path/'last.pt'
    ck = torch.load(ck_path,map_location='cpu',weights_only=False)
    expected = dict(job=arm+str(seed),arm=arm,seed=seed,bundle_sha256=digest(folder/'bundle.json'))
    if (ck.get('kind') != 'transport_channel_training' or ck['identity'] != expected or
            ck['completed_epochs'] != contract['training']['epochs']):
        raise RuntimeError('Matching terminal checkpoint required for research evaluation')
    transfer = read_json(transfer_manifest) if transfer_manifest else None
    label = f'transfer_{digest(transfer_manifest)[:12]}_{split}' if transfer else split
    target = path/f'evaluation_{label}.json'
    if target.exists():
        raise RuntimeError('Evaluation exists; preserve it')
    model = matched_model(contract,arm,seed).to(device)
    model.load_state_dict(ck['model'],strict=True); model.eval()
    baseline = read_json(folder/'baselines.json')
    start,stop = contract['splits'][split]
    rows = {}
    cases = contract['manifest']['cases']
    if transfer is not None:
        import copy
        cases = copy.deepcopy(transfer['cases'])
        source_conditions = {(c['Ez_kVm'],c['B_mT']) for c in contract['manifest']['cases']}
        keys = [c['case_key'] for c in cases]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError('Transfer manifest must have distinct case keys')
        for c in cases:
            if (c['Ez_kVm'],c['B_mT']) in source_conditions:
                raise ValueError('Transfer requires held-out E/B; source realizations are a separate experiment')
            if c.get('evaluation_authorized') is not True or not c.get('exposure') or not c.get('transfer_kind'):
                raise ValueError('Each transfer case needs evaluation_authorized=true, exposure and transfer_kind')
            if c.get('spatial_stride',1) != 1 or not np.isclose(c.get('Ly_m',0),.0128):
                raise ValueError('Only the same native fine grid/domain is supported')
            if not np.isclose(c.get('dt_frame_ns',0),15):
                raise ValueError('Transfer cadence must be 15 ns')
            # Always use source-only scales, regardless of a target manifest's normalization.
            c.update(channels=['electron_den','ion_den','phi'],model_height=260,model_width=256,
                     normalization_low=contract['manifest']['normalization']['low'],
                     normalization_high=contract['manifest']['normalization']['high'],normalization_clip=False)
    for case in cases:
        frames = load_frames(case,start,stop)
        true_obs = observables(frames,case,contract)
        pred_obs = {k:[] for k in ('full','t10','modal','pn','pe','cross')}
        field_sse = np.zeros(3)
        gamma_sse = {'raw':0.,'corr':0.}
        field_count = 0
        with torch.no_grad():
            for s in range(len(frames)-19):
                x=make_input(frames[s:s+10],case,contract,arm)
                pred=model(torch.from_numpy(x[None]).to(device)).cpu().numpy()[0]
                field_sse += np.square(pred[:,:,:257].astype(float)-frames[s+10:s+20,:,:257]).sum((0,2,3))
                field_count += 10*257*256
                pp=physical_fields(pred,contract['manifest'])
                tp=physical_fields(frames[s+10:s+20],contract['manifest'])
                for definition in ('raw','corr'):
                    pg=transport_image(pp,.00005,case['B_mT']*.001,definition)
                    tg=transport_image(tp,.00005,case['B_mT']*.001,definition)
                    gamma_sse[definition]+=float(np.square((pg-tg)/contract['gamma_rms'][definition]).sum())
                obs=observables(pred,case,contract)
                for name in pred_obs:
                    pred_obs[name].append(obs[name])
        predicted={k:np.stack(v) for k,v in pred_obs.items()}
        cond=_condition_values(case,contract['manifest']['condition_channels'],
            normalization=contract['manifest']['condition_normalization'])
        metrics={}
        saved={}
        for name in ('full','t10'):
            x,y=forecast_windows(true_obs[name])
            p=predicted[name].mean(1)
            pooled=baseline[name]['pooled']
            nulls=dict(persistence=x[:,-4:],
                train_mean=np.broadcast_to(pooled['y_mean'],y.shape),
                pooled_AR10_EB=predict_ridge(pooled,np.column_stack((x,np.tile(cond,(len(x),1))))))
            if transfer is None:
                local=baseline[name]['local'][case['case_key']]
                nulls.update(train_mean=np.broadcast_to(local['y_mean'],y.shape),local_AR10=predict_ridge(local,x))
            metrics[name]={key:score_transport(p,y,v) for key,v in nulls.items()}
            saved.update({name+'_truth':y,name+'_prediction':p})
            for key,value in nulls.items():
                saved[name+'_'+key]=value
        archive=path/f'{label}_{case["case_key"]}_transport.npz'
        for name in ('modal','pn','pe','cross'):
            saved['pred_'+name]=predicted[name]
            saved['truth_'+name]=true_obs[name]
        np.savez_compressed(archive,**saved)
        rows[case['case_key']]=dict(field_mse_by_channel=(field_sse/field_count).tolist(),
            local_gamma_source_rms_mse={a:v/field_count for a,v in gamma_sse.items()},
            windows=len(predicted['full']),transport=metrics,archive=str(archive),archive_sha256=digest(archive))
        print(case['case_key'],metrics['full']['pooled_AR10_EB'],flush=True)
    summary={}
    for obs in ('full','t10'):
        summary[obs]={null:float(np.median([r['transport'][obs][null]['skill'] for r in rows.values()]))
                      for null in next(iter(rows.values()))['transport'][obs]
                      if all(r['transport'][obs][null]['skill'] is not None for r in rows.values())}
    atomic_json(target,dict(**expected,split=split,checkpoint_sha256=digest(ck_path),
        frame_range=[start,stop],primary='full',secondary='t10',per_condition=rows,
        transfer_manifest=str(transfer_manifest) if transfer else None,
        transfer_manifest_sha256=digest(transfer_manifest) if transfer else None,
        transfer_metadata=[{k:c.get(k) for k in ('case_key','exposure','transfer_kind')} for c in cases] if transfer else None,
        mean_null='source-pooled TRAIN mean' if transfer else 'per-condition TRAIN mean',
        median_condition_skill=summary,created_utc=now(),
        caveat='Overlapping origins, descriptive; source cases previously examined. Target exposure recorded separately. No population CI.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['smoke','train','evaluate','status'])
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--arm',choices=[*ARMS,'all'],default='all')
    parser.add_argument('--seed',type=int,choices=[42,43])
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--transfer-manifest',type=Path,
                        help='Explicitly authorized held-out cases; no target fitting or scaling')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.action=='status':
        for path in sorted((args.out/'jobs').glob('*/status.json')):
            print(path.parent.name,read_json(path),flush=True)
        if not list((args.out/'jobs').glob('*/status.json')):
            print('No long training has started. Prepared:',(args.out/'bundle.json').exists())
        return
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for technical checks')
    with exclusive(args.out/'execution.lock'):
        if args.action=='smoke':
            smoke(args.out,device)
        elif args.action=='train':
            contract,_=verify_prepared(args.out)
            if (args.out/'PAUSE_AFTER_EPOCH').exists():
                raise RuntimeError('PAUSE_AFTER_EPOCH exists; explicitly remove it to resume')
            shared=(TransportWindows(contract,'train','none').data,
                    TransportWindows(contract,'val','none').data)
            for seed in ([args.seed] if args.seed else [42,43]):
                for arm in (ARMS if args.arm=='all' else [args.arm]):
                    if not train_job(args.out,arm,seed,device,shared):
                        return
                    gc.collect()
                    if device.type=='cuda':
                        torch.cuda.empty_cache()
        else:
            if args.arm=='all' or args.seed is None:
                raise ValueError('Evaluation requires --arm and --seed')
            evaluate(args.out,args.arm,args.seed,args.split,device,args.transfer_manifest)


if __name__=='__main__':
    main()
