# VENDORED VERBATIM from axial-layout-scoping@a6f3e6f4bbd6acfeda70ce15d43db978d43de779 scoping/evidence_io.py (sha256 67bade5eee32be7e8f8d3ca01c4087a3fbc70ad744cce6ddefecde63d9a1da0a)
# Dependency of the vendored kernel_map.py; AXIAL_MANIFEST is unset here so manifest() is {}.
"""JSON artifacts belong to one immutable baseline generation.

AXIAL_MANIFEST is set by the remote orchestrator. Keeping this module free
of torch lets offline analysis and regression tests use the same checks.
"""
import json
import os
from pathlib import Path


def manifest():
    path = os.environ.get('AXIAL_MANIFEST')
    return json.loads(Path(path).read_text()) if path else {}


def write_json(path, obj):
    record = dict(obj)
    run = manifest()
    if run:
        record['generation_id'] = run['generation_id']
        if record.get('use_amp') is None:
            record['use_amp'] = run['use_amp']
    with open(path, 'w') as f:
        json.dump(record, f, indent=1, allow_nan=False)
    print(f'-> {path}')


def read_json(path):
    record = json.loads(Path(path).read_text())
    run = manifest()
    if run and record.get('generation_id') != run['generation_id']:
        raise ValueError(f'{path}: evidence belongs to a different or missing generation')
    return record
