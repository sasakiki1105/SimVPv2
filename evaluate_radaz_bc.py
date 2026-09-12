"""Frozen 181-start evaluation and final direct B/C contrasts; GPU sequential.

Streaming field batches keep RAM bounded. Store spectra and all case/lead
results; compare identities to old pilot phase-0 inputs and fixed null targets.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import numpy as np
import torch
from evaluate_radaz_conditioned_factorial import (build_model, condition_vector, load_case_frames,
    predict_direct10, denormalize, VALID_H)
from radaz_metrics_v3 import band_pool, local_observables
from radaz_bc_metrics import factors, spectrum_metrics, transport_error, condition_summary
from radaz_bc_experiment import ROOT, PLAN, OUT, AUDIT, SPLITS, read_json, atomic_json, verify, PILOT
from run_radaz_paired_pilot import digest, checkpoint_epoch

KEYS = ('pn','pe','cross','gamma','flux_full','flux_omitted')


def now():
    return datetime.now(timezone.utc).isoformat()


def evaluate(job_key, one_case=False):
    bundle = verify()
    job = bundle['jobs'][job_key]
    if checkpoint_epoch(__import__('pathlib').Path(job['checkpoint'])) != 59:
        raise RuntimeError('Terminal checkpoint required')
    checkpoint_sha, bundle_sha = digest(job['checkpoint']), digest(PLAN/'bundle.json')
    folder = OUT/'evaluation'/job_key
    folder.mkdir(parents=True,exist_ok=True)
    identity = {'job':job_key,'checkpoint_sha256':checkpoint_sha,'bundle_sha256':bundle_sha}
    if (folder/'results.json').exists():
        result = read_json(folder/'results.json')
        if any(result.get(k) != v for k,v in identity.items()):
            raise RuntimeError('Completed evaluation belongs to another model/protocol')
        for split_rows in result['splits'].values():
            for row in split_rows['per_condition'].values():
                if digest(row['spectra_path']) != row['spectra_sha256']:
                    raise RuntimeError('Completed spectra changed')
        return result
    manifest = read_json(bundle['manifest'])
    low, high = (np.asarray(manifest['normalization'][k],dtype=float) for k in ('low','high'))
    old = read_json(PILOT/'evaluation/results.json')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    model, epoch = build_model(job['config'],job['checkpoint'],device)
    result = {**identity,'status':'exploratory_all_phase_source_evaluation','starts_per_condition':181,
              'splits':{},'started_utc':now()}
    for split,(start,stop) in SPLITS.items():
        rows = {}
        for case in manifest['cases']:
            key = case['case_key']
            row_path = folder/f'{split}_{key}.json'
            if row_path.exists():
                row = read_json(row_path)
                if any(row.get(k) != v for k,v in identity.items()) or digest(row['spectra_path']) != row['spectra_sha256']:
                    raise RuntimeError('Partial evaluation mismatch')
                rows[key] = row
                continue
            print(now(),job_key,split,key,'begin',flush=True)
            atomic_json(folder/'status.json',{**identity,'stage':'evaluate','split':split,'case':key,'updated_utc':now()})
            frames,raw,xm,ym = load_case_frames(case,low,high,start,stop)
            del raw
            condition,_,n0 = condition_vector(case,manifest)
            pool,dy,b = band_pool(xm[:VALID_H]),float(np.median(np.diff(ym))),case['B_mT']*.001
            collected = {prefix+k:[] for prefix in ('pred_','truth_') for k in KEYS}
            copied_flux = []
            old_x,old_y = [],[]
            input_hash = hashlib.sha256()
            field_sse,copy_sse,true_energy = np.zeros((10,3)),np.zeros((10,3)),np.zeros((10,3))
            coefficients_sse = {k:[0.,0.] for k in ('ne_coeff','ey_coeff','phi_coeff')}
            for offset in range(0,181,4):
                starts = range(offset,min(offset+4,181))
                x = np.empty((len(starts),10,5,260,256),np.float32)
                target = np.empty((len(starts),10,3,260,256),np.float32)
                for j,s in enumerate(starts):
                    x[j,:,:3] = frames[s:s+10]
                    x[j,:,3:] = condition[None,:,None,None]
                    target[j] = frames[s+10:s+20]
                    input_hash.update(x[j].tobytes())
                    input_hash.update(target[j].tobytes())
                    if s % 20 == 0:
                        old_x.append(x[j].copy())
                        old_y.append(target[j].copy())
                prediction = predict_direct10(model,x,device)
                persistence = np.repeat(x[:,-1:,:3],10,1)
                ps = local_observables(denormalize(prediction,low,high),pool,dy,b)
                ts = local_observables(denormalize(target,low,high),pool,dy,b)
                cs = local_observables(denormalize(persistence,low,high),pool,dy,b)
                for prefix,spectra in (('pred_',ps),('truth_',ts)):
                    for k in KEYS:
                        collected[prefix+k].append(spectra[k])
                copied_flux.append(cs['flux_full'])
                p,t,c = (a[...,:VALID_H,:].astype(float) for a in (prediction,target,persistence))
                field_sse += np.sum((p-t)**2,axis=(0,3,4))
                copy_sse += np.sum((c-t)**2,axis=(0,3,4))
                physical_truth = t*(high-low)[None,None,:,None,None]+low[None,None,:,None,None]
                true_energy += np.sum(physical_truth**2,axis=(0,3,4))
                radial_w = pool.mean(0)[None,None,:,None]
                for k in coefficients_sse:
                    coefficients_sse[k][0] += float(np.sum(radial_w*abs(ps[k]-ts[k])**2))
                    coefficients_sse[k][1] += float(np.sum(radial_w*abs(cs[k]-ts[k])**2))
                del x,target,prediction,persistence,p,t,c,physical_truth,ps,ts,cs
            old_hash = hashlib.sha256()
            for data in old_x+old_y:
                old_hash.update(data.tobytes())
            if old_hash.hexdigest() != old['input_hashes'][split+'/'+key]:
                raise RuntimeError('Old phase-0 input/target identity changed')
            del old_x,old_y,frames
            arrays = {k:np.concatenate(v) for k,v in collected.items()}
            del collected
            pred = {k:arrays['pred_'+k] for k in KEYS}
            truth = {k:arrays['truth_'+k] for k in KEYS}
            with np.load(AUDIT/f'{split}_{key}_nulls.npz') as n:
                for k in ('pn','pe','cross','gamma','flux_full'):
                    np.testing.assert_allclose(truth[k],n['truth_'+k],rtol=1e-4,atol=max(abs(truth[k]).max()*1e-8,1e-30))
                metrics = spectrum_metrics(pred,truth,n['copy_O'],n['AR_O'],n['copy_gamma'],b)
                for baseline in ('copy_O','input_mean_O','AR_O'):
                    from radaz_bc_metrics import organization_error
                    metrics['null_'+baseline] = organization_error(n[baseline],truth,n['copy_O'],n['AR_O'])
                phase0 = spectrum_metrics({k:v[::20] for k,v in pred.items()},
                    {k:v[::20] for k,v in truth.items()},n['copy_O'][::20],n['AR_O'][::20],n['copy_gamma'][::20],b)
            if job['cell'] == 'A':
                expected = read_json(AUDIT/'results.json')['splits'][split]['phase0']['D'+str(job['seed'])][key]['O_time_rmse']
                if abs(phase0['O_time_rmse']-expected)>1e-3:
                    raise RuntimeError('A reproduction differs from cached phase-0 O audit')
            metrics['flux_full'] = transport_error(pred['flux_full'],truth['flux_full'],np.concatenate(copied_flux))
            # Existing train-only scalar AR models; retain their original validation choice.
            from analyze_radaz_nextstep_baselines import predict_ar
            from radaz_bc_metrics import window_series
            from radaz_metrics_v3 import skill
            cache_root = ROOT/'workdirs/2D_RadAz/radaz_nextstep_baselines_v1'
            with np.load(cache_root/(key+'_spectra.npz')) as history:
                for observable in ('gamma','flux_full'):
                    with np.load(cache_root/(key+'_'+observable+'_ar.npz')) as saved_ar:
                        ar = predict_ar(dict(saved_ar),window_series(history[observable],start-1200,stop-1200)[0])
                    metrics[observable+'_skill_vs_AR'] = skill(pred[observable],truth[observable],ar)
            metrics['field_normalized_mse'] = float(field_sse.sum()/(181*10*3*VALID_H*256))
            metrics['field_skill_vs_copy'] = float(1-field_sse.sum()/copy_sse.sum()) if copy_sse.sum() else None
            metrics['field_lead_channel_skill'] = np.divide(copy_sse-field_sse,copy_sse,
                out=np.full_like(copy_sse,np.nan),where=copy_sse>0).tolist()
            if not np.isfinite(metrics['field_lead_channel_skill']).all():
                raise RuntimeError('Undefined field null comparison; requires explicit reporting update')
            metrics['field_channel_physical_nrmse'] = np.sqrt(field_sse.sum(0)*(high-low)**2/true_energy.sum(0)).tolist()
            metrics['complex_coefficient_skill_vs_copy'] = {k:1-v[0]/v[1] if v[1] else None for k,v in coefficients_sse.items()}
            archive = folder/f'{split}_{key}.npz'
            np.savez_compressed(archive,**arrays)
            row = {**identity,**metrics,'n0':n0,'phase0_secondary':phase0,
                   'all_phase_input_target_sha256':input_hash.hexdigest(),'old_phase0_input_target_sha256':old_hash.hexdigest(),
                   'spectra_path':str(archive),'spectra_sha256':digest(archive)}
            atomic_json(row_path,row)
            rows[key] = row
            print(job_key,split,key,'O RMSE',metrics['O_time_rmse'],'vs AR',metrics['O_skill_vs_AR'],flush=True)
            if one_case:
                atomic_json(folder/'smoke_case.json',{'status':'one_registered_case_verified_not_full_evaluation',
                    **identity,'row':str(row_path),'row_sha256':digest(row_path)})
                return None
        result['splits'][split] = {'per_condition':rows,'summary':{k:condition_summary(rows,k)
            for k in ('O_time_rmse','O_skill_vs_copy','O_skill_vs_AR','field_skill_vs_copy')}}
        atomic_json(folder/'partial_results.json',result)
    verify()
    result['completed_utc'] = now()
    atomic_json(folder/'results.json',result)
    atomic_json(folder/'status.json',{**identity,'stage':'complete','completed_utc':now()})
    return result


def aggregate():
    bundle = verify()
    results = {k:evaluate(k) for k in bundle['jobs']}
    final = {'bundle_sha256':digest(PLAN/'bundle.json'),'status':'complete_exploratory_BC_comparison',
             'splits':{},'completed_utc':now(),'limitations':bundle['limitations']}
    lines = ['# D-only B/C architecture comparison','','All-phase source development data; two training seeds; no significance claim.',
             'Positive B benefit means lower time-resolved signed O RMSE. A is a historical resumed reference.', '',
             '| Split | Seed | Median A−B | Median A−C | Median C−B | B better than C |', '|---|---:|---:|---:|---:|---:|']
    for split in SPLITS:
        per_seed = {}
        for seed in (42,43):
            rows = {c:results[c+str(seed)]['splits'][split]['per_condition'] for c in 'ABC'}
            contrasts = {}
            for case in rows['A']:
                a,b,c = (rows[k][case] for k in 'ABC')
                if len({r['all_phase_input_target_sha256'] for r in (a,b,c)}) != 1:
                    raise RuntimeError('Architecture cases use different inputs/targets')
                contrasts[case] = {'n0':a['n0'],'A_minus_B':a['O_time_rmse']-b['O_time_rmse'],
                    'A_minus_C':a['O_time_rmse']-c['O_time_rmse'],'C_minus_B':c['O_time_rmse']-b['O_time_rmse'],
                    'B_skill_vs_AR':b['O_skill_vs_AR'],'C_skill_vs_AR':c['O_skill_vs_AR']}
            summary = {k:condition_summary(contrasts,k)['median'] for k in ('A_minus_B','A_minus_C','C_minus_B')}
            # Rank relation is descriptive only, not a new success criterion.
            from scipy.stats import rankdata
            x,y = (rankdata([r[k] for r in contrasts.values()]) for k in ('n0','C_minus_B'))
            rho = float(np.corrcoef(x,y)[0,1]) if np.std(y)>0 else None
            per_seed[str(seed)] = {'per_condition':contrasts,'condition_medians':summary,
                                  'rho_n0_B_minus_C_improvement':rho}
            lines.append('| '+split+' | '+str(seed)+' | '+' | '.join(f'{summary[k]:.6f}' for k in ('A_minus_B','A_minus_C','C_minus_B'))+
                         ' | '+str(sum(v['C_minus_B']>0 for v in contrasts.values()))+'/6 |')
        final['splits'][split] = {'per_seed':per_seed,
            'mean_of_seed_condition_medians':{k:float(np.mean([v['condition_medians'][k] for v in per_seed.values()])) for k in ('A_minus_B','A_minus_C','C_minus_B')}}
    final['evaluation_sha256'] = {k:digest(OUT/'evaluation'/k/'results.json') for k in results}
    atomic_json(OUT/'results.json',final)
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker = '<!-- radaz-bc-donly-20260912-complete -->'
    memo = ROOT.parent/'ICL_reserch_memo.md'
    if marker not in memo.read_text(encoding='utf-8'):
        entry = '\n\n'+marker+'\n## '+now()+'：D-only B/C 2seed比較完了（自動記録・探索的）\n\n'
        entry += '\n'.join(lines[2:])+'\n\n全181開始窓、source 6条件、GN、D-only、60 epoch終端。'
        entry += '\nB/Cはseed・epoch別の学習窓順序を固定。Aは中断履歴のある既存D42/D43で、完全paired対照ではない。'
        entry += '\n独立PICの確認・統計的有意性・literal Nyquist原因の同定は主張しない。'
        entry += '\n出力：`SimVPv2/workdirs/2D_RadAz/radaz_bc_donly_20260912/`。詳細はresults.json、REPORT.mdと凍結protocol。\n'
        with memo.open('ab') as f:
            f.write(entry.encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',choices=['A42','A43','B42','B43','C42','C43'])
    parser.add_argument('--aggregate',action='store_true')
    parser.add_argument('--one-case',action='store_true',help='Verify one registered A case; preserve partial output, no completed evaluation')
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.job:
        evaluate(args.job,one_case=args.one_case)
    else:
        parser.error('Choose --job or --aggregate')
