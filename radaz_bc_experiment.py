"""Frozen assets and paths for the D-only B/C intervention (no implicit jobs)."""
import json
from pathlib import Path
from run_radaz_paired_pilot import ROOT, OUT as PILOT, digest, verify_bundle as verify_pilot

PLAN = ROOT/'protocols/radaz_bc_donly_20260912'
OUT = ROOT/'workdirs/2D_RadAz/radaz_bc_donly_20260912'
AUDIT = ROOT.parent/'research_results/audits/bc_temporal_preflight_20260912'
SPLITS = {'source_validation': (1600,1800), 'source_test': (1800,2000)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def verify():
    bundle = read_json(PLAN/'bundle.json')
    verify_pilot()
    if digest(PILOT/'bundle.json') != bundle['parent_bundle_sha256']:
        raise RuntimeError('Parent pilot changed')
    for path, expected in bundle['assets_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Frozen B/C asset changed: '+path)
    for job in bundle['jobs'].values():
        if job['cell'] == 'A' and digest(job['checkpoint']) != job['checkpoint_sha256']:
            raise RuntimeError('A reference checkpoint changed')
    if bundle['training_order'] != ['B42','C42','B43','C43']:
        raise RuntimeError('Unexpected execution plan')
    return bundle
