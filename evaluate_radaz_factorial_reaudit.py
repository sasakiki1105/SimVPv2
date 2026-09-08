"""CPU-only, matched-last exploratory factorial reanalysis; see frozen protocol."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import numpy as np
import torch

from evaluate_radaz_conditioned_factorial import (
    CELLS, MANIFEST, build_model, condition_vector, load_case_frames,
    predict_direct10, denormalize, VALID_H,
)
from openstl.models.simvp_factory import configure_translator_norm
from radaz_metrics_v3 import (VERSION, band_pool, local_observables,
    compare_observables, field_diagnostics, summarize, nrmse, skill)
from analyze_radaz_nextstep_baselines import residual_geometry, summary_error

ROOT = Path(__file__).resolve().parent
PROTO = ROOT / 'audit_protocols/factorial_reaudit_20260909.json'
PLAN = PROTO.with_suffix('.md')
OUT = ROOT / 'workdirs/2D_RadAz/radaz_factorial_reaudit_20260909'
MODELS = dict(CELLS)
MODELS['U-Pv2'] = dict(workdir=ROOT/'workdirs/2D_RadAz/radaz_axis_factorial_UPv2_powercross_direct10_bs1_60ep',
    config=ROOT/'configs/custom/pepapic/SimVP_gSTA_radaz_factorial_UPv2_60ep.py')
POLICIES = ('input', 'running')
SPLITS = {'source_validation':(1600,1800), 'source_test':(1800,2000), 'holdout_test':(1800,2000)}
KEYS = ('pn','pe','cross','gamma','flux_full','flux_retained','flux_omitted')

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2), b''): h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def freeze():
    if PROTO.exists(): raise RuntimeError('Protocol already frozen; refuse overwrite')
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    files=[Path(__file__), PLAN, MANIFEST, ROOT/'radaz_metrics_v3.py',
        ROOT/'evaluate_radaz_conditioned_factorial.py', ROOT/'evaluate_radaz_corrected_A.py',
        ROOT/'analyze_radaz_nextstep_baselines.py', *sorted((ROOT/'openstl').rglob('*.py'))]
    for cell in MODELS.values(): files.extend([cell['config'],cell['workdir']/'checkpoints/last.ckpt'])
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    fields={c['case_key']:{'path':c['path'],'size':Path(c['path']).stat().st_size,
        'mtime_ns':Path(c['path']).stat().st_mtime_ns,'role':c['role']} for c in manifest['cases']}
    write_json(PROTO, {'created_utc':now(),'status':'exploratory, previously inspected data',
        'metrics_version':VERSION,'files_sha256':hashes,'fields':fields,
        'models':list(MODELS),'policies':POLICIES,'splits':SPLITS,'device':'cpu'})
    print('Frozen', PROTO, digest(PROTO), flush=True)

def verify():
    protocol=json.loads(PROTO.read_text(encoding='utf-8'))
    for rel, expected in protocol['files_sha256'].items():
        if digest(ROOT/rel)!=expected: raise RuntimeError('Frozen asset changed: '+rel)
    for item in protocol['fields'].values():
        s=Path(item['path']).stat()
        if (s.st_size,s.st_mtime_ns)!=(item['size'],item['mtime_ns']):
            raise RuntimeError('Data metadata changed: '+item['path'])
    # Local commit is a versioned record, not an externally certified timestamp.
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(PROTO.relative_to(ROOT))],cwd=ROOT,text=True).strip()
    if not commit: raise RuntimeError('Commit the protocol before inference')
    for path in (PROTO, PLAN, Path(__file__)):
        committed=subprocess.check_output(['git','show',f'{commit}:{path.relative_to(ROOT).as_posix()}'],cwd=ROOT)
        if committed!=path.read_bytes(): raise RuntimeError('Protocol/code differs from committed bytes')
    return digest(PROTO),commit

def select_cases(manifest, split):
    source=split.startswith('source_')
    return [c for c in manifest['cases'] if (c['role']=='source')==source]

def case_inputs(case, manifest, split):
    start,stop=SPLITS[split]
    low,high=(np.asarray(manifest['normalization'][k]) for k in ('low','high'))
    frames,raw,xm,ym=load_case_frames(case,low,high,start,stop)
    del raw
    cond,_,n0=condition_vector(case,manifest)
    x=np.empty((10,10,5,*frames.shape[-2:]), dtype=np.float32)
    target=np.empty((10,10,3,*frames.shape[-2:]), dtype=np.float32)
    for j in range(10):
        x[j,:,:3]=frames[j*20:j*20+10]
        x[j,:,3:]=cond[None,:,None,None]
        target[j]=frames[j*20+10:(j+1)*20]
    pool=band_pool(xm[:VALID_H])
    return x,target,pool,float(np.median(np.diff(ym))),case['B_mT']*.001,low,high,n0

def amplitude_sums(pred, truth):
    result={}
    for key in ('pn','pe'):
        den=float(truth[key].sum())
        if den<=0: raise ValueError('Cannot calibrate zero truth power')
        result[key]={'xy':float(np.sqrt(pred[key]*truth[key]).sum()/den),
                     'xx':float(pred[key].sum()/den)}
    return result

def fit_factors(rows):
    values={}
    for key,name in (('pn','ne'),('pe','ey')):
        xy=sum(r['amplitude_fit_sums'][key]['xy'] for r in rows)
        xx=sum(r['amplitude_fit_sums'][key]['xx'] for r in rows)
        if xx<=0: raise ValueError('Cannot calibrate zero predicted power')
        values[name]=max(0.,xy/xx)
    return values

def scaled_observables(pred, factors):
    an,ae=factors['ne'],factors['ey']
    out={k:v*(an*ae) for k,v in pred.items()}
    out['pn']=pred['pn']*an**2
    out['pe']=pred['pe']*ae**2
    for key,factor in (('ne_coeff',an),('ey_coeff',ae),('phi_coeff',ae)):
        if key in pred: out[key]=pred[key]*factor
    return out

def spectral_row(pred,truth,copy,b,pool):
    row=compare_observables(pred,truth,copy,b,pool)
    for k in ('gamma','flux_full'):
        row[k+'_error_details']=summary_error(pred[k],truth[k],copy[k])
    archive={p+k:d[k] for p,d in (('pred_',pred),('truth_',truth)) for k in KEYS}
    row['residual_geometry']=residual_geometry(archive,b)
    return row

def contrasts(per_condition):
    ud,cd,up,cp,v2=(per_condition[k] for k in MODELS)
    out={name:{} for name in ('conditioning','physics','interaction','U-Pv2_minus_U-D','U-Pv2_minus_U-P')}
    for case in ud:
        for metric in ('gamma_time_nrmse','gamma_skill_vs_copy'):
            u,d,p,c,v=(r[case][metric] for r in (ud,cd,up,cp,v2))
            values=[((d-u)+(c-p))/2,((p-u)+(c-d))/2,c-d-p+u,v-u,v-p] if None not in (u,d,p,c,v) else [None]*5
            for (name,rows),value in zip(out.items(),values): rows.setdefault(case,{})[metric]=value
    return {k:{'per_condition':v,'summary':summarize(v)} for k,v in out.items()}

def evaluate():
    sha,commit=verify()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert len(select_cases(manifest,'source_test'))==6 and len(select_cases(manifest,'holdout_test'))==4
    status={'status':'running','started_utc':now(),'pid':os.getpid(),'protocol_sha256':sha,'commit':commit,'completed_rows':0}
    write_json(OUT/'status.json',status)
    result={'protocol_sha256':sha,'protocol_commit':commit,'metrics_version':VERSION,
        'interpretation':'Exploratory matched legacy windows; no independent confirmation',
        'device':'cpu','evaluations':{},'calibration':{},'input_hashes':{}}
    try:
        for cell,spec in MODELS.items():
            for policy in POLICIES:
                variant=cell+'_'+policy
                print(now(), 'Loading',variant,flush=True)
                model,epoch=build_model(spec['config'],spec['workdir']/'checkpoints/last.ckpt',torch.device('cpu'))
                if epoch!=59: raise RuntimeError('Expected last index59: '+variant)
                if policy=='input': configure_translator_norm(model,'batch_input')
                factors=None
                for split in SPLITS:
                    rows={}
                    for case in select_cases(manifest,split):
                        key=case['case_key']
                        path=OUT/(variant+'_'+split+'_'+key+'.json')
                        if path.exists():
                            record=json.loads(path.read_text(encoding='utf-8'))
                            if record['protocol_sha256']!=sha: raise RuntimeError('Stale result: '+str(path))
                            if record.get('calibration_factors')!=factors: raise RuntimeError('Calibration mismatch')
                            if digest(path.with_suffix('.npz'))!=record['spectra_sha256']: raise RuntimeError('Corrupt spectra')
                        else:
                            began=time.monotonic()
                            x,target,pool,dy,b,low,high,n0=case_inputs(case,manifest,split)
                            h=hashlib.sha256(x.tobytes());h.update(target.tobytes())
                            cp=np.repeat(x[:,-1:,:3],10,axis=1)
                            truth=local_observables(denormalize(target,low,high),pool,dy,b)
                            copy=local_observables(denormalize(cp,low,high),pool,dy,b)
                            prediction=predict_direct10(model,x,torch.device('cpu'))
                            pred=local_observables(denormalize(prediction,low,high),pool,dy,b)
                            raw=spectral_row(pred,truth,copy,b,pool)
                            raw.update(field_diagnostics(prediction[...,:VALID_H,:],target[...,:VALID_H,:],cp[...,:VALID_H,:],low,high))
                            record={'protocol_sha256':sha,'cell':cell,'policy':policy,'split':split,'case':key,
                                'epoch_index':epoch,'n0':n0,'input_target_sha256':h.hexdigest(),'raw':raw,
                                'copy':compare_observables(copy,truth,copy,b,pool),
                                'amplitude_fit_sums':amplitude_sums(pred,truth),
                                'calibration_factors':factors}
                            # Validation supplies fitting statistics only; no coefficient fit on test.
                            if factors is not None:
                                record['calibrated']=spectral_row(scaled_observables(pred,factors),truth,copy,b,pool)
                            spectra={p+k:d[k] for p,d in (('pred_',pred),('truth_',truth),('copy_',copy)) for k in KEYS}
                            np.savez_compressed(path.with_suffix('.npz'),**spectra)
                            record['spectra_sha256']=digest(path.with_suffix('.npz'))
                            record['elapsed_seconds']=time.monotonic()-began
                            write_json(path,record)
                            print(now(),variant,split,key,'Gamma',round(raw['gamma_time_nrmse'],6),'skill',round(raw['gamma_skill_vs_copy'],6),'sec',round(record['elapsed_seconds'],1),flush=True)
                            del x,target,cp,prediction,truth,copy,pred,spectra
                            gc.collect()
                        signature=split+'/'+key
                        prior=result['input_hashes'].setdefault(signature,record['input_target_sha256'])
                        if prior!=record['input_target_sha256']: raise RuntimeError('Unmatched windows')
                        rows[key]=record
                        status.update(completed_rows=status['completed_rows']+1,current=variant+'/'+signature,updated_utc=now())
                        write_json(OUT/'status.json',status)
                    if split=='source_validation':
                        factors=fit_factors(list(rows.values()))
                        result['calibration'][variant]={'factors':factors,'fit_split':split,'conditions':list(rows)}
                        # Persist calibration BEFORE test data are opened.
                        write_json(OUT/(variant+'_calibration.json'),{'protocol_sha256':sha,**result['calibration'][variant]})
                    result['evaluations'].setdefault(split,{})[variant]={'per_condition':rows,
                        'raw_summary':summarize({k:v['raw'] for k,v in rows.items()})}
                    if split!='source_validation':
                        result['evaluations'][split][variant]['calibrated_summary']=summarize({k:v['calibrated'] for k,v in rows.items()})
                    write_json(OUT/'partial_results.json',result)
                del model
                gc.collect()
        result['contrasts']={}
        for split in ('source_test','holdout_test'):
            for policy in POLICIES:
                for calibration in ('raw','calibrated'):
                    table={cell:{k:v[calibration] for k,v in result['evaluations'][split][cell+'_'+policy]['per_condition'].items()} for cell in MODELS}
                    result['contrasts']['/'.join((split,policy,calibration))]=contrasts(table)
        result['completed_utc']=now()
        write_json(OUT/'results.json',result)
        report=['# Matched-last factorial reanalysis (exploratory)','',
            'Previously inspected data; seed one; fixed phase-0 legacy windows. No p-values, new gate or causal conclusion.','',
            '| Split | Cell/policy | Median Gamma NRMSE | Median Gamma skill vs copy | Calibrated skill |',
            '|---|---|---:|---:|---:|']
        for split in ('source_test','holdout_test'):
            for variant,entry in result['evaluations'][split].items():
                raw,cal=entry['raw_summary'],entry['calibrated_summary']
                report.append(f"| {split} | {variant} | {raw['gamma_time_nrmse']['median']:.6g} | {raw['gamma_skill_vs_copy']['median']:.6g} | {cal['gamma_skill_vs_copy']['median']:.6g} |")
        report+=['','Full per-condition, signed organization, amplitude decomposition, SI errors and factorial contrasts: results.json.',
            'Amplitude calibration uses source validation only and does not repair phase. Stored/input contrasts hold weights fixed.',
            f'Protocol SHA256: {sha}; local versioned commit: {commit}.']
        (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        memo=ROOT.parent/'ICL_reserch_memo.md'
        entry='\n\n---\n\n# 2026-09-09 旧factorial再評価の完了（自動記録）\n\n'+ '\n'.join(report[2:])+'\n\nCommand: `python evaluate_radaz_factorial_reaudit.py --run` (CPU).\nOutput: `SimVPv2/workdirs/2D_RadAz/radaz_factorial_reaudit_20260909/`.\n次：モデル別・条件別の反転と補正後の残差を確認し、D/P pilotと区別して解釈する。\n'
        with memo.open('ab') as f: f.write(entry.encode('utf-8'))
        status.update(status='complete',finished_utc=now(),result=str(OUT/'results.json'))
        write_json(OUT/'status.json',status)
        print('Complete',OUT,flush=True)
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),updated_utc=now())
        write_json(OUT/'status.json',status)
        raise

def self_check():
    rng=np.random.default_rng(20260909)
    t={'pn':np.exp(rng.normal(size=(2,3,4,64))), 'pe':np.exp(rng.normal(size=(2,3,4,64)))}
    p={'pn':t['pn']*4,'pe':t['pe']*9}
    f=fit_factors([{'amplitude_fit_sums':amplitude_sums(p,t)}]*6)
    np.testing.assert_allclose([f['ne'],f['ey']],[.5,1/3])
    o=rng.uniform(-1,1,size=t['pn'].shape)
    for d,org in ((t,o),(p,-o)):
        d['cross']=np.sqrt(d['pn']*d['pe'])*org+0j
        d['gamma']=-2*d['cross'].real/.02
    scaled=scaled_observables(p,f)
    np.testing.assert_allclose(scaled['gamma'],-t['gamma'])
    archive={pre+k:d[k] for pre,d in (('pred_',p),('truth_',t)) for k in ('pn','pe','cross','gamma')}
    geom=residual_geometry(archive,.02)
    np.testing.assert_allclose(np.sum(geom['amplitude_organization_interaction_gram_over_total_sse']),1)
    table={cell:{'case':{'gamma_time_nrmse':v,'gamma_skill_vs_copy':v}} for cell,v in zip(MODELS,[1,3,4,10,12])}
    c=contrasts(table)
    np.testing.assert_allclose([c[k]['per_condition']['case']['gamma_time_nrmse'] for k in c],[4,5,4,11,8])
    print('PASS: validation-only amplitude formula, phase cannot be repaired by amplitude, exact residual reconstruction, factorial interaction')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true')
    group.add_argument('--check',action='store_true')
    group.add_argument('--self-check',action='store_true')
    group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.freeze: freeze()
    elif args.check: print(verify())
    elif args.self_check: self_check()
    else: evaluate()
