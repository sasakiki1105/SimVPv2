"""Calibrate source-TRAIN gamma scales and direct ARs; seal after verification.

No TEST fields or condition holdouts are opened. Outputs are additive and a
completed calibration is never silently overwritten.
"""
import argparse
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from radaz_transport_channels import (
    ROOT, OUT, PLAN, SOURCE, BASE_CONFIG, SPLITS, read_json, digest, atomic_json,
    validate_manifest, load_frames, physical_fields, transport_image, observables,
    forecast_windows, fit_ridge, _condition_values,
)


def calibrate(folder):
    folder = Path(folder)
    if (folder/'contract.json').exists() or (folder/'bundle.json').exists():
        raise RuntimeError('Calibration exists; use another version/output folder')
    manifest = read_json(SOURCE)
    validate_manifest(manifest)
    selection = ROOT.parent/'research_results/fewmode_stepA_20260927/results.json'
    t10 = read_json(selection)['sets_fixed_on_source_validation_common']['T10']
    for b in range(4):
        modes = t10[str(b)]
        if len(modes) != 10 or len(set(modes)) != 10 or min(modes) < 1 or max(modes) > 64:
            raise ValueError('Unexpected fixed T10 set')
    contract = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        status='implementation_preparation_not_confirmatory', manifest=manifest,
        pre=10, aft=10, splits=SPLITS, t10=t10,
        t10_source=str(selection), t10_source_sha256=digest(selection),
        gamma_rms={}, gamma_normalization='zero offset, source TRAIN per-arm global RMS',
        input_definition='product derived from physical reconstruction of the SAME float32 input fields',
        primary='future leads 1..10 mean of Gamma_full, four equally weighted radial bands',
        secondary='same future-window mean of fixed-source-validation Gamma_T10',
        training=dict(epochs=60, seeds=[42,43], batch_size=1, optimizer='Adam', lr=.001,
                      scheduler='OneCycleLR', pct_start=.1, weight_decay=0., precision='float32',
                      field_loss='normalized three-field MSE including replicated radial padding',
                      checkpoint='terminal epoch 60', condition_film=True),
        exposures='All source cases previously examined; exploratory, not fresh blind evidence')
    series, sums, count, identities = {}, {'raw': 0., 'corr': 0.}, 0, {}
    folder.mkdir(parents=True, exist_ok=True)
    for case in manifest['cases']:
        key = case['case_key']
        path = Path(case['path'])
        before = [path.stat().st_size, path.stat().st_mtime_ns]
        identities[str(path)] = before
        with h5py.File(path, 'r') as handle:
            for axis in ('x_m', 'y_m'):
                np.testing.assert_allclose(handle['axes/'+axis][:], np.arange(257)*.00005, rtol=1e-6, atol=1e-10)
            np.testing.assert_allclose(np.diff(handle['axes/time_s'][:1800]), 15e-9, rtol=1e-5, atol=1e-14)
        rows = {k: [] for k in ('full', 't10')}
        for start in range(0, 1800, 16):
            stop = min(start+16, 1800)
            frames = load_frames(case, start, stop)
            if start < 1600:
                physical = physical_fields(frames[:min(stop,1600)-start], manifest)
                for arm in ('raw','corr'):
                    g = transport_image(physical, .00005, case['B_mT']*.001, arm)
                    sums[arm] += float(np.square(g).sum())
                count += g.size
            obs = observables(frames, case, contract)
            for name in rows:
                rows[name].append(obs[name])
            if start % 400 == 0:
                print(key, 'source frames', stop, '/ 1800', flush=True)
        if [path.stat().st_size, path.stat().st_mtime_ns] != before:
            raise RuntimeError('Source changed during calibration')
        for name in rows:
            series[key+'/'+name] = np.concatenate(rows[name])
    contract['gamma_rms'] = {a: float(np.sqrt(v/count)) for a,v in sums.items()}
    contract['gamma_train_elements_per_arm'] = count
    if any(not np.isfinite(v) or v <= 0 for v in contract['gamma_rms'].values()):
        raise ValueError('Nonfinite/zero calibration RMS')
    baselines = {}
    for name in ('full', 't10'):
        local, pooled_x, pooled_y, pooled_xv, pooled_yv = {}, [], [], [], []
        for c in manifest['cases']:
            key = c['case_key']
            x,y = forecast_windows(series[key+'/'+name][:1600])
            xv,yv = forecast_windows(series[key+'/'+name][1600:1800])
            local[key] = fit_ridge(x,y,xv,yv)
            cond = _condition_values(c, manifest['condition_channels'],
                                     normalization=manifest['condition_normalization'])
            pooled_x.append(np.column_stack((x, np.tile(cond, (len(x),1)))))
            pooled_xv.append(np.column_stack((xv, np.tile(cond, (len(xv),1)))))
            pooled_y.append(y); pooled_yv.append(yv)
        baselines[name] = dict(local=local, pooled=fit_ridge(np.concatenate(pooled_x),
            np.concatenate(pooled_y), np.concatenate(pooled_xv), np.concatenate(pooled_yv)))
    np.savez_compressed(folder/'source_train_val_transport.npz', **series)
    atomic_json(folder/'baselines.json', baselines)
    atomic_json(folder/'data_identity.json', identities)
    atomic_json(folder/'contract.json', contract)
    print('CALIBRATED', contract['gamma_rms'], flush=True)


def seal(folder):
    folder = Path(folder)
    if (folder/'bundle.json').exists():
        raise RuntimeError('Already sealed; create a new dated version for changes')
    smoke = read_json(folder/'smoke.json')
    if smoke['status'] != 'passed' or set(smoke['arms']) != {'none','raw','corr'}:
        raise RuntimeError('Complete three-arm real-data smoke required')
    sources = [ROOT/name for name in (
        'radaz_transport_channels.py', 'prepare_radaz_transport_channels.py',
        'run_radaz_transport_channels.py', 'tests/test_radaz_transport_channels.py',
        'radaz_metrics_v3.py', 'openstl/models/simvp_model.py',
        'openstl/models/simvp_factory.py', 'openstl/datasets/dataloader_pepapic_h5.py')]
    sources += [BASE_CONFIG, PLAN/'PROTOCOL.md', SOURCE]
    if smoke['code_sha256'] != {str(p):digest(p) for p in sources[:3]}:
        raise RuntimeError('Implementation changed after smoke')
    assets = sources + [folder/name for name in ('contract.json','baselines.json',
        'source_train_val_transport.npz','data_identity.json','smoke.json')]
    contract = read_json(folder/'contract.json')
    assets.append(Path(contract['t10_source']))
    atomic_json(folder/'bundle.json', dict(status='sealed_implementation_before_long_training',
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        training_order=[a+str(s) for s in (42,43) for a in ('none','raw','corr')],
        assets_sha256={str(p):digest(p) for p in assets},
        data_identity=read_json(folder/'data_identity.json')))
    print('SEALED', folder/'bundle.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['calibrate','seal'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    (calibrate if args.action == 'calibrate' else seal)(args.out)
