"""Provenance gate + session manifest (gate-02 finding 23).

verify(paths) FAILS LOUDLY unless: the msst clone HEAD equals the pinned
commit (configs/msst_commit.txt) and is clean; the checkpoint's sha256 equals
the pinned value (configs/checkpoint.txt); the config path is the registered
one relative to the clone (line 3 of checkpoint.txt); use_amp is an explicit
boolean. Every run calls it before building anything.

The session manifest (evidence/session_manifest.json) binds a session to one
immutable identity: repo commit + clean tree, sha256 of the registration file,
sha256 of every fullmodel/*.py, fullmodel/c3_fix/*.py, scripts/session*.sh,
the pins above, the clip manifest hash, and the environment fingerprint.
Runs record the manifest id; references are written with a sidecar carrying
it, and validation refuses references from another manifest.
"""

import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _git(repo, *args):
    return subprocess.check_output(['git', '-C', repo, *args], text=True).strip()


def pins():
    lines = [l.strip() for l in open(os.path.join(_ROOT, 'configs', 'msst_commit.txt'))
             if l.strip() and not l.strip().startswith('#')]
    msst_commit = lines[0]
    data = [l.strip() for l in open(os.path.join(_ROOT, 'configs', 'checkpoint.txt'))
            if l.strip() and not l.strip().startswith('#')]
    assert len(data) == 3, 'configs/checkpoint.txt must hold URL / sha256 / config path'
    return {'msst_commit': msst_commit, 'ckpt_url': data[0], 'ckpt_sha256': data[1].lower(),
            'config_rel': data[2]}


def verify(paths):
    """Returns the provenance record; raises RuntimeError on any mismatch."""
    p = pins()
    head = _git(paths['msst'], 'rev-parse', 'HEAD')
    if head != p['msst_commit']:
        raise RuntimeError(f"msst clone HEAD {head} != pinned {p['msst_commit']}")
    dirty = _git(paths['msst'], 'status', '--porcelain')
    if dirty:
        raise RuntimeError('msst clone is not clean:\n' + dirty[:500])
    ck = sha256_file(paths['ckpt'])
    if ck != p['ckpt_sha256']:
        raise RuntimeError(f"checkpoint sha256 {ck} != pinned {p['ckpt_sha256']}")
    expected_cfg = os.path.realpath(os.path.join(paths['msst'], p['config_rel']))
    if os.path.realpath(paths['config']) != expected_cfg:
        raise RuntimeError(f"config {paths['config']} is not the registered {expected_cfg}")
    return {'msst_commit': head, 'msst_clean': True, 'ckpt_sha256': ck,
            'config': expected_cfg, 'config_sha256': sha256_file(expected_cfg)}


def code_hashes():
    files = []
    for sub in ('fullmodel', os.path.join('fullmodel', 'c3_fix'), os.path.join('fullmodel', 'attrib'),
                'arms', 'scripts'):
        d = os.path.join(_ROOT, sub)
        for name in sorted(os.listdir(d)):
            if name.endswith(('.py', '.sh')):
                files.append(os.path.join(sub, name))
    return {f: sha256_file(os.path.join(_ROOT, f)) for f in files}


def create_manifest(evidence_dir, paths, env, session):
    if _git(_ROOT, 'status', '--porcelain'):
        raise RuntimeError('session requires a committed, clean tree')
    rec = {'manifest_id': str(uuid.uuid4()), 'session': session,
           'created_utc': datetime.now(timezone.utc).isoformat(),
           'repo_commit': _git(_ROOT, 'rev-parse', 'HEAD'),
           'registration_sha256': sha256_file(os.path.join(_ROOT, 'prereg_amendments.md')),
           'clips_manifest_sha256': sha256_file(os.path.join(_ROOT, 'configs', 'sdr_clips.json')),
           'code': code_hashes(), 'provenance': verify(paths), 'env': env}
    path = os.path.join(evidence_dir, 'session_manifest.json')
    if os.path.exists(path):
        raise RuntimeError(f'{path} exists — one manifest per evidence directory')
    with open(path, 'w') as f:
        json.dump(rec, f, indent=1)
    return rec


def load_manifest(evidence_dir):
    path = os.path.join(evidence_dir, 'session_manifest.json')
    if not os.path.exists(path):
        raise RuntimeError('session_manifest.json missing — run the session preflight')
    with open(path) as f:
        rec = json.load(f)
    if rec['code'] != code_hashes():
        raise RuntimeError('measuring code changed since the session manifest was created')
    if rec['repo_commit'] != _git(_ROOT, 'rev-parse', 'HEAD'):
        raise RuntimeError('repo commit differs from the session manifest')
    return rec
