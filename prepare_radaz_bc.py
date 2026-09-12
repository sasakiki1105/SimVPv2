"""Freeze the separate post-pilot B/C bundle after CPU audit and GPU smoke."""
import json
from pathlib import Path
import numpy as np
from radaz_bc_experiment import ROOT, PLAN, AUDIT, PILOT, atomic_json
from run_radaz_paired_pilot import digest, verify_bundle


def main():
    parent = verify_bundle()
    PLAN.mkdir(parents=True,exist_ok=True)
    if (PLAN/'bundle.json').exists():
        raise RuntimeError('B/C already frozen; create a dated amendment instead of overwriting')
    smoke = json.loads((AUDIT/'training_smoke.json').read_text())
    audit = json.loads((AUDIT/'results.json').read_text())
    assert all(smoke['cells'][k]['finite_gradients'] for k in 'ABC')
    jobs = {}
    for seed in (42,43):
        a_checkpoint = ROOT/f'workdirs/2D_RadAz/radaz_pilot_D_GN_A_seed{seed}_60ep/checkpoints/last.ckpt'
        jobs['A'+str(seed)] = {'cell':'A','seed':seed,'config':str(ROOT/parent['configs']['D']),
                              'checkpoint':str(a_checkpoint),'checkpoint_sha256':digest(a_checkpoint),'historical_resumed_reference':True}
        for cell in ('B','C'):
            cfg = ROOT/f'configs/custom/pepapic/SimVP_gSTA_radaz_bc_Donly_{cell}_60ep.py'
            name = f'radaz_bc_Donly_{cell}_GN_seed{seed}_60ep_20260912'
            arguments = list(parent['commands']['D'][1:])
            for flag,value in (('--config_file',str(cfg)),('--ex_name',name),('--seed',str(seed))):
                arguments[arguments.index(flag)+1] = value
            jobs[cell+str(seed)] = {'cell':cell,'seed':seed,'config':str(cfg),'training_args':arguments,
                'checkpoint':str(ROOT/f'workdirs/2D_RadAz/{name}/checkpoints/last.ckpt')}
    assets = [PLAN/'PROTOCOL.md']+[ROOT/name for name in (
        'radaz_bc_metrics.py','audit_radaz_bc_temporal.py','radaz_bc_experiment.py','train_radaz_bc.py',
        'evaluate_radaz_bc.py','run_radaz_bc.py','prepare_radaz_bc.py','smoke_radaz_bc.py','tests/test_radaz_bc.py')]
    assets += [Path(v['config']) for v in jobs.values()]
    assets += [AUDIT/'results.json',AUDIT/'protocol.json',AUDIT/'training_smoke.json']+list(AUDIT.glob('*.npz'))
    assets += [ROOT/'workdirs/2D_RadAz/radaz_nextstep_baselines_v1'/f'{c}_{obs}_ar.npz'
               for c in parent['source_fields'] for obs in ('gamma','flux_full')]
    assets += [ROOT/'workdirs/2D_RadAz/radaz_nextstep_baselines_v1'/f'{c}_spectra.npz' for c in parent['source_fields']]
    # 1581 valid starts per 1600-frame source train segment, 6 fixed cases.
    samples = np.asarray([(case,start) for case in range(6) for start in range(1581)],dtype='<i8')
    import hashlib
    bundle = {'status':'frozen_before_BC_training','version':1,'parent_bundle_sha256':digest(PILOT/'bundle.json'),
        'manifest':parent['manifest'],'jobs':jobs,'training_order':['B42','C42','B43','C43'],
        'samples_per_epoch':9486,'train_sample_map_sha256':hashlib.sha256(samples.tobytes()).hexdigest(),
        'selected_O_AR_ridge':audit['selected_O_ridge'],'evaluation_windows_per_case':181,
        'assets_sha256':{str(p.resolve()):digest(p) for p in assets},
        'limitations':['Already inspected source cases, one PIC realization per condition; exploratory',
                       'A42/A43 are historical resumed references, not matched sampler controls',
                       'B/C share isolated epoch permutations, not identical initialization tensors',
                       'No confidence intervals, significance claims, or threshold calibrated from current endpoint',
                       'B changes sampling in encoder/decoder; C does not expand every hidden dimension',
                       'Local git commit is versioning, not an independent external timestamp; no remote push'],
        'smoke_step_seconds':{k:smoke['cells'][k]['median_step_seconds'] for k in 'ABC'}}
    atomic_json(PLAN/'bundle.json',bundle)
    print('Frozen',PLAN/'bundle.json','SHA256',digest(PLAN/'bundle.json'))


if __name__ == '__main__':
    main()
