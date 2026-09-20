"""Task-quality gate (prereg_amendments.md §2): 8 fixed MUSDB18-HQ test clips,
evaluated at EVERY compared operating point, paired against eager arm A at
that same operating point.

Reconstruction policy (registered): each 8.000-s clip (352800 samples) is
processed as non-overlapping consecutive chunks of the operating point's size
(1 x 352800, 4 x 88200, 16 x 22050, 80 x 4410) and the chunk outputs are
concatenated (no overlap, no crossfade). Short-chunk SDR_ref values are
therefore NOT quality claims about the model; the gate is the paired
regression vs A under the identical policy.

Metrics (stereo signals flattened to one vector; float64 accumulation):
  SDR(ref, est)   = 10 log10( ||ref||^2 / ||ref - est||^2 )         (plain)
  SI-SDR(ref, est): both zero-mean; s = <est,ref>/||ref||^2 * ref;
                    10 log10( ||s||^2 / ||est - s||^2 )
computed (a) vs the reference vocals stem and (b) vs the saved eager-A output
for the same clip at the same operating point. A perfect match yields +inf
(reported as the string "inf"). Gate: mean over the 8 clips of
ΔSDR_ref = SDR_ref(arm) - SDR_ref(A) >= -0.1 dB; per-clip minimum, SI-SDR and
the vs-A numbers are reported alongside (the gate is the mean).
"""

import hashlib
import json
import math
import os

import numpy as np
import soundfile as sf
import torch

CLIP_SAMPLES = 352800
THRESHOLD_DB = -0.1


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_clips(clips_dir, manifest_path):
    """Verifies every wav's sha256 against the manifest (loud failure)."""
    with open(manifest_path) as f:
        man = json.load(f)
    clips = []
    for c in man['clips']:
        rec = {'index': c['index'], 'track': c['track'], 'start_sample': c['start_sample']}
        for kind in ('mixture', 'vocals'):
            entry = c['files'][kind]
            path = os.path.join(clips_dir, entry['name'])
            if not os.path.exists(path):
                raise FileNotFoundError(f'clip missing: {path}')
            got = _sha256(path)
            if got != entry['sha256']:
                raise ValueError(f'clip sha256 mismatch: {path} {got} != {entry["sha256"]}')
            audio, sr = sf.read(path, dtype='float32', always_2d=True)   # (n, 2)
            assert sr == 44100 and audio.shape == (CLIP_SAMPLES, 2), (path, sr, audio.shape)
            rec[kind] = torch.from_numpy(audio.T.copy())                  # (2, n)
        clips.append(rec)
    assert len(clips) == 8, f'expected 8 clips, got {len(clips)}'
    assert len({c['track'] for c in clips}) == 8, 'clips must come from 8 distinct tracks'
    return man, clips


def _db(num, den):
    """Registered semantics: num>0, den==0 -> 'inf' (perfect match);
    num==0 -> 'undefined' (zero reference/projection); non-finite inputs ->
    'undefined'; else a float."""
    if not (math.isfinite(num) and math.isfinite(den)):
        return 'undefined'
    if num == 0:
        return 'undefined'
    if den == 0:
        return 'inf'
    return float(10 * math.log10(num / den))


def sdr_db(ref, est):
    ref = ref.reshape(-1).double()
    est = est.reshape(-1).double()
    return _db(float((ref ** 2).sum()), float(((ref - est) ** 2).sum()))


def si_sdr_db(ref, est):
    ref = ref.reshape(-1).double()
    est = est.reshape(-1).double()
    ref = ref - ref.mean()
    est = est - est.mean()
    rr = float((ref ** 2).sum())
    if rr == 0:
        return 'undefined'
    s = (est * ref).sum() / rr * ref
    return _db(float((s ** 2).sum()), float(((est - s) ** 2).sum()))


def _mean_with_inf(values):
    finite = [v for v in values if isinstance(v, float)]
    n_inf = sum(1 for v in values if v == 'inf')
    n_undef = sum(1 for v in values if v == 'undefined')
    return {'mean_finite_db': float(np.mean(finite)) if finite else None,
            'n_finite': len(finite), 'n_inf': n_inf, 'n_undefined': n_undef}


def a_output_path(ref_dir, chunk, index):
    return os.path.join(ref_dir, f'sdr_A_c{chunk}_clip{index:02d}.pt')


def separate_clip(step, ctx, mixture, chunk):
    """Non-overlapping chunked processing + concatenation (registered policy)."""
    assert CLIP_SAMPLES % chunk == 0, (CLIP_SAMPLES, chunk)
    outs = []
    with torch.no_grad(), ctx():
        for i in range(CLIP_SAMPLES // chunk):
            x = mixture[:, i * chunk:(i + 1) * chunk].unsqueeze(0).cuda()
            out = step(x)
            torch.cuda.synchronize()
            out = out.float().clone()            # borrowed replay buffer -> owned
            if out.dim() == 3:
                out = out[0]
            assert out.shape == (2, chunk), tuple(out.shape)
            outs.append(out.cpu())
    return torch.cat(outs, dim=-1)


def run_gate(step, ctx, clips, ref_dir, chunk, save_reference=False, manifest_id=None):
    import provenance
    rows = []
    for c in clips:
        est = separate_clip(step, ctx, c['mixture'], chunk)
        vocals = c['vocals']
        row = {'index': c['index'], 'track': c['track'],
               'sdr_ref_db': sdr_db(vocals, est),
               'si_sdr_ref_db': si_sdr_db(vocals, est),
               'finite': bool(torch.isfinite(est).all())}
        path = a_output_path(ref_dir, chunk, c['index'])
        if save_reference:
            os.makedirs(ref_dir, exist_ok=True)
            torch.save(est, path)
            if manifest_id is not None:
                provenance.bind_artifact(path, manifest_id)
                row['a_output_sha256'] = provenance.sha256_file(path)
            row['saved'] = path
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f'eager-A clip output missing: {path}')
            if manifest_id is not None:
                provenance.check_artifact(path, manifest_id)
                row['a_output_sha256'] = provenance.sha256_file(path)
            a_out = torch.load(path, weights_only=True)
            row['sdr_ref_A_db'] = sdr_db(vocals, a_out)
            d = (row['sdr_ref_db'] - row['sdr_ref_A_db']
                 if isinstance(row['sdr_ref_db'], float) and isinstance(row['sdr_ref_A_db'], float)
                 else None)
            row['delta_sdr_ref_db'] = d
            row['sdr_vs_A_db'] = sdr_db(a_out, est)
            row['si_sdr_vs_A_db'] = si_sdr_db(a_out, est)
            row['max_abs_diff_vs_A'] = float((a_out - est).abs().max())
        rows.append(row)
    rec = {'clips': rows, 'operating_point': chunk,
           'chunks_per_clip': CLIP_SAMPLES // chunk,
           'reconstruction': 'non-overlapping concatenation',
           'threshold_mean_delta_sdr_db': THRESHOLD_DB, 'n_clips': len(rows)}
    rec['sdr_ref_summary'] = _mean_with_inf([r['sdr_ref_db'] for r in rows])
    rec['mean_sdr_ref_db'] = rec['sdr_ref_summary']['mean_finite_db']
    if save_reference:
        rec['status'] = 'REFERENCE_SAVED'
        return rec
    # the gate uses the SDR_ref terms only: both arm and A must be finite
    deltas = [r['delta_sdr_ref_db'] for r in rows]
    if any(d is None for d in deltas):
        rec['status'] = 'FAIL'
        rec['reason'] = ('a paired SDR_ref term is non-finite ("inf" = perfect match to the '
                         'reference stem, "undefined" = zero reference) — not gateable')
        return rec
    rec['mean_delta_sdr_ref_db'] = float(np.mean(deltas))
    rec['min_delta_sdr_ref_db'] = float(np.min(deltas))
    # output fidelity vs A: infinities are expected for A itself and are kept
    rec['sdr_vs_A_summary'] = _mean_with_inf([r['sdr_vs_A_db'] for r in rows])
    rec['all_finite'] = all(r['finite'] for r in rows)
    rec['status'] = ('PASS' if rec['all_finite'] and
                     rec['mean_delta_sdr_ref_db'] >= THRESHOLD_DB else 'FAIL')
    return rec
