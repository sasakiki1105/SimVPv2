"""Prepare, verify, train, evaluate and roll out the five-field transport model.

All commands are local. Long training and scientific evaluation are explicit;
smoke only uses source TRAIN and never starts a queue in the background.
"""
from __future__ import annotations

import argparse
import copy
import gc
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import radaz_transport_channels as old
import radaz_transport_velocity as v2
import radaz_transport_output as parent_output
import h5py
from run_radaz_transport_channels import exclusive, OrderedSampler, save_checkpoint, now


def verify(folder):
    contract,bundle=old.verify_prepared(folder)
    if contract.get('version')!=3 or contract.get('gamma_feedback')!='predicted_channel':
        raise RuntimeError('Not a five-field prediction contract')
    return contract,bundle


def prepare(folder):
    if (folder/'contract.json').exists():
        raise RuntimeError('Prepared version exists; preserve it')
    parent,bundle=old.verify_prepared(parent_output.OUT)
    contract=copy.deepcopy(parent)
    total=count=0; stats={}
    # Calibration reads TRAIN only, all valid nodes, excluding duplicate y endpoint.
    for case in contract['manifest']['cases']:
        subtotal=subcount=subsum=0
        minimum=float('inf'); maximum=-float('inf')
        start,stop=contract['splits']['train']
        with h5py.File(case['path'],'r') as handle:
            field=handle['fields/electron_wd']
            if field.shape!=(len(handle['axes/time_s']),257,257):
                raise ValueError('Velocity is not on the same consolidated grid')
            for lo in range(start,stop,32):
                data=np.asarray(field[lo:min(lo+32,stop),:257,:256],dtype=np.float64)
                if not np.isfinite(data).all():
                    raise ValueError('Nonfinite source TRAIN velocity')
                subtotal+=float(np.square(data).sum()); subcount+=data.size
                subsum+=float(data.sum()); minimum=min(minimum,float(data.min()))
                maximum=max(maximum,float(data.max()))
        total+=subtotal; count+=subcount
        stats[case['case_key']]=dict(count=subcount,rms=float(np.sqrt(subtotal/subcount)),
            mean=subsum/subcount,min=minimum,max=maximum,frames=[start,stop])
        print('VELOCITY CALIBRATION',case['case_key'],stats[case['case_key']],flush=True)
    scale=float(np.sqrt(total/count))
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Invalid TRAIN velocity scale')
    contract.update(version=3,created_utc=now(),status='combined_five_field_first',
        parent_bundle=str(parent_output.OUT/'bundle.json'),
        parent_bundle_sha256=old.digest(parent_output.OUT/'bundle.json'),
        physical_channels=['electron_den','ion_den','phi','gamma_corr','electron_wd'],
        output_channels={'corr_uez':5},gamma_definition='corr',
        velocity_rms=scale,velocity_normalization='zero offset / source TRAIN global RMS, unclipped',
        velocity_calibration=dict(count=count,per_condition=stats),
        velocity_feedback='predicted_channel',
        primary='future 10-frame mean of direct gamma_corr, pooled into four radial bands: Gamma_full',
        priority='combined state first; channel ablations deferred by user',
        claim_limit='joint model performance only; no attribution to an individual channel')
    contract['training'].update(velocity_weight=1/3,
        loss='field_MSE + gamma_MSE/3 + velocity_MSE/3; auxiliary losses on valid nodes')
    old.atomic_json(folder/'contract.json',contract)
    print('PREPARED five-field state; velocity RMS',scale,flush=True)


def seal(folder):
    if (folder/'bundle.json').exists():
        raise RuntimeError('Already sealed; preserve this version')
    _,parent=old.verify_prepared(parent_output.OUT)
    smoke=old.read_json(folder/'smoke.json')
    files=[v2.ROOT/name for name in ('radaz_transport_velocity.py','run_radaz_transport_velocity.py')]
    if smoke['status']!='passed' or smoke['code_sha256']!={str(p):old.digest(p) for p in files}:
        raise RuntimeError('Current five-field GPU smoke required')
    files += [v2.ROOT/'tests/test_radaz_transport_velocity.py',v2.PLAN/'PROTOCOL.md',
              folder/'contract.json',folder/'smoke.json',parent_output.OUT/'bundle.json']
    assets=dict(parent['assets_sha256'])
    assets.update({str(p):old.digest(p) for p in files})
    old.atomic_json(folder/'bundle.json',dict(status='sealed_five_field_experiment',
        created_utc=now(),assets_sha256=assets,data_identity=parent['data_identity'],
        training_order=['corr_uez42','corr_uez43'],
        git_parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=v2.ROOT,text=True).strip()))
    print('SEALED',folder/'bundle.json',flush=True)


def smoke(folder,device):
    if (folder/'smoke.json').exists():
        raise RuntimeError('Smoke already exists; preserve it')
    contract=old.read_json(folder/'contract.json')
    case=contract['manifest']['cases'][0]
    frames=v2.load_frames(case,1400,1420,contract)
    arm='corr_uez'
    model=v2.matched_model(contract,arm,42).to(device)
    x=torch.from_numpy(v2.make_input(frames[:10],case,contract)[None]).to(device)
    y=torch.from_numpy(v2.make_target(frames[10:],case,contract)[None]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start=time.monotonic()
    losses=[v2.checked_step(model,optimizer,x,y,check_gradients=True) for _ in range(3)]
    if device.type=='cuda':
        torch.cuda.synchronize()
    elapsed=(time.monotonic()-start)/3
    gradients={name:float(model.dec.readout.weight.grad[index].abs().sum())
               for name,index in (('gamma',3),('velocity',4))}
    if not all(value>0 for value in gradients.values()):
        raise RuntimeError('Both added output heads must learn')
    model.eval(); captured=[]
    hook=model.register_forward_pre_hook(lambda mod,args:captured.append(args[0].detach().cpu().clone()))
    rolled=v2.autonomous_rollout(model,x,2); hook.remove()
    torch.testing.assert_close(captured[1][:,:,:5,:257],rolled[:,:10,:,:257],rtol=0,atol=0)
    torch.testing.assert_close(captured[1][:,:,-2:],captured[0][:,:,-2:],rtol=0,atol=0)
    checkpoint=folder/'smoke_checkpoints'/f'{arm}.pt'
    save_checkpoint(checkpoint,dict(kind='technical_five_field_smoke',model=model.state_dict()))
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model'],strict=True)
    restored=v2.autonomous_rollout(model,x,2)
    torch.testing.assert_close(rolled,restored,rtol=0,atol=0)
    metrics,arrays=v2.diagnose(rolled[0,:10].numpy(),frames[10:],case,contract)
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise RuntimeError('Nonfinite diagnostics')
    result=dict(input_shape=list(x.shape),output_shape=list(y.shape),
        parameters=sum(p.numel() for p in model.parameters()),losses=losses,
        readout_gradient_l1=gradients,mean_step_seconds=elapsed,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
        rollout_shape=list(rolled.shape),predicted_state_feedback_exact=True,
        checkpoint_rollout_roundtrip_exact=True,diagnostics_finite=True,diagnostic_only_metrics=metrics)
    print('SMOKE five-field',result,flush=True)
    old.atomic_json(folder/'smoke.json',dict(status='passed',created_utc=now(),
        purpose='engineering_only_no_forecast_performance_claim',frames=[1400,1420],
        case=case['case_key'],device=str(device),torch=torch.__version__,result=result,
        code_sha256={str(v2.ROOT/n):old.digest(v2.ROOT/n) for n in
                     ('radaz_transport_velocity.py','run_radaz_transport_velocity.py')}))


def train_job(folder,arm,seed,device,preloaded=None):
    contract,_=verify(folder)
    path=folder/'jobs'/(arm+str(seed)); path.mkdir(parents=True,exist_ok=True)
    identity=dict(job=arm+str(seed),arm=arm,seed=seed,bundle_sha256=old.digest(folder/'bundle.json'))
    model=v2.matched_model(contract,arm,seed).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=contract['training']['lr'],weight_decay=0.)
    train=v2.TransportWindows(contract,'train',arm,preloaded=preloaded[0] if preloaded else None)
    val=v2.TransportWindows(contract,'val',arm,preloaded=preloaded[1] if preloaded else None)
    sampler=OrderedSampler(len(train),seed)
    loader=DataLoader(train,batch_size=1,sampler=sampler,num_workers=0,pin_memory=device.type=='cuda',
                      generator=torch.Generator().manual_seed(2000003+seed))
    validation=DataLoader(val,batch_size=1,shuffle=False,num_workers=0)
    epochs=contract['training']['epochs']; weight=contract['training']['gamma_weight']
    velocity_weight=contract['training'].get('velocity_weight',1/3)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=contract['training']['lr'],
        total_steps=len(loader)*epochs,pct_start=contract['training']['pct_start'])
    first=global_step=0
    if (path/'last.pt').exists():
        checkpoint=torch.load(path/'last.pt',map_location=device,weights_only=False)
        if checkpoint['identity']!=identity or checkpoint.get('kind')!='transport_five_field_training':
            raise RuntimeError('Checkpoint belongs to a different experiment')
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        first,global_step=checkpoint['completed_epochs'],checkpoint['global_step']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type=='cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        del checkpoint
    for epoch in range(first,epochs):
        model.train(); sampler.epoch=epoch
        start=time.monotonic(); sums={k:0. for k in ('total','field','gamma','velocity')}
        for i,(x,y) in enumerate(loader):
            parts=v2.checked_step(model,optimizer,x.to(device),y.to(device),weight,velocity_weight)
            scheduler.step(); global_step+=1
            for key,value in parts.items():
                sums[key]+=value
            if i==0 or (i+1)%50==0 or i+1==len(loader):
                elapsed=time.monotonic()-start
                old.atomic_json(path/'status.json',dict(**identity,status='training',epoch=epoch+1,
                    epochs=epochs,step=i+1,steps_per_epoch=len(loader),global_step=global_step,loss=parts,
                    mean_step_seconds=elapsed/(i+1),updated_utc=now()))
                print(identity['job'],f'epoch {epoch+1}/{epochs} step {i+1}/{len(loader)}',parts,flush=True)
        model.eval(); vsums={k:0. for k in sums}
        with torch.no_grad():
            for x,y in validation:
                parts=v2.loss_parts(model(x.to(device)),y.to(device),weight,velocity_weight)
                for key,value in parts.items():
                    vsums[key]+=float(value)
        save_checkpoint(path/'last.pt',dict(kind='transport_five_field_training',identity=identity,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            completed_epochs=epoch+1,global_step=global_step,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            model_config=v2.model_config(contract,arm)))
        row=dict(**identity,completed_epochs=epoch+1,train_loss={k:v/len(loader) for k,v in sums.items()},
            val_loss={k:v/len(validation) for k,v in vsums.items()},updated_utc=now(),elapsed_seconds=time.monotonic()-start)
        with (path/'epochs.jsonl').open('a',encoding='utf-8') as handle:
            import json
            handle.write(json.dumps(row)+'\n')
        paused=(folder/'PAUSE_AFTER_EPOCH').exists()
        old.atomic_json(path/'status.json',dict(**row,status='paused' if paused else 'complete' if epoch+1==epochs else 'epoch_saved'))
        if paused:
            return False
    return True


def load_terminal(folder,arm,seed,device):
    contract,_=verify(folder)
    path=folder/'jobs'/(arm+str(seed))
    ck=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
    identity=dict(job=arm+str(seed),arm=arm,seed=seed,bundle_sha256=old.digest(folder/'bundle.json'))
    if (ck.get('kind')!='transport_five_field_training' or ck['identity']!=identity or
            ck['completed_epochs']!=contract['training']['epochs']):
        raise RuntimeError('Matching terminal five-field checkpoint required')
    model=v2.matched_model(contract,arm,seed).to(device)
    model.load_state_dict(ck['model'],strict=True); model.eval()
    return contract,path,model,dict(**identity,checkpoint_sha256=old.digest(path/'last.pt'))


def evaluation_cases(contract,transfer_manifest):
    if transfer_manifest is None:
        return contract['manifest']['cases']
    cases=copy.deepcopy(old.read_json(transfer_manifest)['cases'])
    seen={(c['Ez_kVm'],c['B_mT']) for c in contract['manifest']['cases']}
    if not cases or len({c['case_key'] for c in cases})!=len(cases):
        raise ValueError('Missing/duplicate transfer cases')
    for c in cases:
        if (c['Ez_kVm'],c['B_mT']) in seen or not c.get('evaluation_authorized') or not c.get('exposure') or not c.get('transfer_kind'):
            raise ValueError('Transfer requires explicitly listed unseen E/B and exposure/kind metadata')
        if c.get('spatial_stride',1)!=1 or not np.isclose(c.get('Ly_m',0),.0128) or not np.isclose(c.get('dt_frame_ns',0),15):
            raise ValueError('Transfer grid/cadence mismatch')
        c.update(channels=['electron_den','ion_den','phi'],model_height=260,model_width=256,
            normalization_low=contract['manifest']['normalization']['low'],
            normalization_high=contract['manifest']['normalization']['high'],normalization_clip=False)
    return cases


def evaluate(folder,arm,seed,split,device,transfer_manifest=None):
    contract,path,model,identity=load_terminal(folder,arm,seed,device)
    label=f'transfer_{old.digest(transfer_manifest)[:12]}_{split}' if transfer_manifest else split
    destination=path/f'evaluation_{label}.json'
    if destination.exists():
        raise RuntimeError('Evaluation already exists')
    baseline=old.read_json(contract['baseline_path'])
    cases=evaluation_cases(contract,transfer_manifest)
    start,stop=contract['splits'][split]; rows={}
    for case in cases:
        frames=v2.load_frames(case,start,stop,contract)
        truths=old.observables(frames[:,:3],case,contract)
        arrays={}; measures=[]
        with torch.no_grad():
            for s in range(len(frames)-19):
                x=v2.make_input(frames[s:s+10],case,contract,arm)
                pred=model(torch.from_numpy(x[None]).to(device)).cpu().numpy()[0]
                metrics,obs=v2.diagnose(pred,frames[s+10:s+20],case,contract,arm)
                measures.append(metrics)
                for key,value in obs.items():
                    arrays.setdefault(key,[]).append(value)
        arrays={k:np.stack(v) for k,v in arrays.items()}
        cond=old._condition_values(case,contract['manifest']['condition_channels'],
            normalization=contract['manifest']['condition_normalization'])
        scores={}
        routes={'derived_full':'full','derived_t10':'t10'}
        if arm!='none':
            routes['direct_full']='full'
        for route,observable in routes.items():
            x,y=old.forecast_windows(truths[observable])
            p=arrays[route].mean(1); pooled=baseline[observable]['pooled']
            nulls=dict(persistence=x[:,-4:],train_mean=np.broadcast_to(pooled['y_mean'],y.shape),
                pooled_AR10_EB=old.predict_ridge(pooled,np.column_stack((x,np.tile(cond,(len(x),1))))))
            if transfer_manifest is None:
                local=baseline[observable]['local'][case['case_key']]
                nulls.update(train_mean=np.broadcast_to(local['y_mean'],y.shape),local_AR10=old.predict_ridge(local,x))
            scores[route]={n:old.score_transport(p,y,v) for n,v in nulls.items()}
            for key,value in nulls.items():
                arrays[observable+'_'+key]=value
        archive=path/f'{label}_{case["case_key"]}.npz'
        np.savez_compressed(archive,**arrays)
        rows[case['case_key']]=dict(windows=len(measures),transport=scores,
            mean_diagnostics={k:np.mean([r[k] for r in measures],axis=0).tolist() for k in measures[0]},
            archive=str(archive),archive_sha256=old.digest(archive))
        print('EVALUATED',arm,case['case_key'],flush=True)
    aggregate={route:{null:float(np.median([row['transport'][route][null]['skill']
        for row in rows.values() if row['transport'][route][null]['skill'] is not None]))
        for null in next(iter(rows.values()))['transport'][route]}
        for route in next(iter(rows.values()))['transport']}
    old.atomic_json(destination,dict(**identity,created_utc=now(),split=split,frame_range=[start,stop],
        primary='derived_full' if arm=='none' else 'direct_full',per_condition=rows,
        condition_equal_weight_median_skill=aggregate,
        transfer_manifest_sha256=old.digest(transfer_manifest) if transfer_manifest else None,
        transfer_metadata=[{k:c.get(k) for k in ('case_key','exposure','transfer_kind')} for c in cases],
        caveat='Exploratory. Direct gamma gives full flux only; T10 is field-derived.'))


def rollout_job(folder,arm,seed,split,case_key,blocks,device):
    contract,path,model,identity=load_terminal(folder,arm,seed,device)
    case=next(c for c in contract['manifest']['cases'] if c['case_key']==case_key)
    start,stop=contract['splits'][split]
    if blocks<1 or start+10+10*blocks>stop:
        raise ValueError('Requested rollout crosses the registered split boundary')
    destination=path/f'rollout_{split}_{case_key}_{blocks}blocks.json'
    if destination.exists():
        raise RuntimeError('Rollout exists; preserve it')
    # Prediction is complete before reading future truth for scoring.
    past=v2.load_frames(case,start,start+10,contract)
    initial=v2.make_input(past,case,contract,arm)
    prediction=v2.autonomous_rollout(model,torch.from_numpy(initial[None]).to(device),blocks)[0].numpy()
    truth=v2.load_frames(case,start+10,start+10+10*blocks,contract)
    metrics,arrays=v2.diagnose(prediction,truth,case,contract,arm)
    archive=destination.with_suffix('.npz')
    np.savez_compressed(archive,predicted_state=prediction,**arrays)
    old.atomic_json(destination,dict(**identity,case=case_key,split=split,blocks=blocks,
        initial_frame_range=[start,start+10],forecast_end_exclusive=start+10+10*blocks,
        forecast_duration_ns=blocks*150,gamma_feedback=contract['gamma_feedback'],
        future_truth_used_for_feedback=False,metrics=metrics,archive=str(archive),
        archive_sha256=old.digest(archive),created_utc=now(),
        caveat='Single initial history: descriptive rollout, not pooled forecast skill or a stability guarantee'))


def main():
    parser=argparse.ArgumentParser(description='Five-field gamma_corr + u_ez training and autonomous prediction')
    parser.add_argument('action',choices=['prepare','smoke','seal','train','evaluate','rollout','status'])
    parser.add_argument('--out',type=Path,default=v2.OUT)
    parser.add_argument('--arm',choices=[*v2.ARMS,'all'],default='all')
    parser.add_argument('--seed',type=int,choices=[42,43])
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--case',dest='case_key',default='E10_B20')
    parser.add_argument('--blocks',type=int,default=4)
    parser.add_argument('--transfer-manifest',type=Path)
    args=parser.parse_args(); torch.set_num_threads(4)
    if args.action=='status':
        if (args.out/'runner.json').exists():
            print('RUNNER',old.read_json(args.out/'runner.json'))
        statuses=sorted((args.out/'jobs').glob('*/status.json'))
        for p in statuses:
            row=old.read_json(p)
            if row['status']=='training':
                steps=row['steps_per_epoch']; epochs=row['epochs']
                percent=100*row['global_step']/(steps*epochs)
                remaining=(steps*epochs-row['global_step'])*row['mean_step_seconds']/3600
                print(f"{p.parent.name}: epoch {row['epoch']}/{epochs}, step {row['step']}/{steps}, "
                      f"{percent:.2f}%, ETA {remaining:.1f} h (training only, early estimate)")
                print('  loss',row['loss'],'updated',row['updated_utc'])
            else:
                print(p.parent.name,row)
        if not statuses:
            print('No optimizer step recorded yet. Five-output bundle prepared:',(args.out/'bundle.json').exists())
        return
    with exclusive(args.out/'execution.lock'):
        if args.action=='prepare':
            prepare(args.out); return
        if args.action=='seal':
            seal(args.out); return
        device=torch.device(args.device)
        if device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if args.action=='smoke':
            smoke(args.out,device)
        elif args.action=='train':
            contract,_=verify(args.out)
            if (args.out/'PAUSE_AFTER_EPOCH').exists():
                raise RuntimeError('Pause requested; explicitly remove PAUSE_AFTER_EPOCH to resume')
            old.atomic_json(args.out/'runner.json',dict(pid=os.getpid(),status='loading',started_utc=now(),seeds=[args.seed] if args.seed else [42,43]))
            shared=(v2.TransportWindows(contract,'train','corr_uez').data,v2.TransportWindows(contract,'val','corr_uez').data)
            for seed in ([args.seed] if args.seed else [42,43]):
                for arm in (v2.ARMS if args.arm=='all' else [args.arm]):
                    old.atomic_json(args.out/'runner.json',dict(pid=os.getpid(),status='training',job=arm+str(seed),updated_utc=now()))
                    if not train_job(args.out,arm,seed,device,shared):
                        old.atomic_json(args.out/'runner.json',dict(pid=os.getpid(),status='paused',job=arm+str(seed),updated_utc=now()))
                        return
                    gc.collect()
                    if device.type=='cuda':
                        torch.cuda.empty_cache()
            old.atomic_json(args.out/'runner.json',dict(pid=os.getpid(),status='training_complete',updated_utc=now()))
        else:
            if args.arm=='all' or args.seed is None:
                raise ValueError('Specify one --arm and --seed')
            if args.action=='evaluate':
                evaluate(args.out,args.arm,args.seed,args.split,device,args.transfer_manifest)
            else:
                if args.transfer_manifest:
                    raise ValueError('This rollout CLI currently supports source cases only')
                rollout_job(args.out,args.arm,args.seed,args.split,args.case_key,args.blocks,device)


if __name__=='__main__':
    main()
