"""Versioned, shared qualification validator (gate-04 f47/f55/f56).

Every consumer that publishes or audits a number re-derives a run's
eligibility from the RECORD ITSELF (protocol, validation, lineage,
recording stability, gate, manifest) — never from stored booleans.
"""

QUALIFY_VERSION = 3
REGISTERED_ITERS = 50
REGISTERED_REPEATS = 5
COMPILE_ARMS = ('C1', 'C2', 'C3', 'C1A', 'C1D', 'H')
ZERO_BREAK_ARMS = ('C3', 'H')
VARIANT_FIELDS = ('capture_scope', 'capture_adapter_active', 'compile_mode', 'stream_count')
TIERS = {'loose': (1e-2, 1e-3), 'tight': (1e-4, 1e-5)}
TIER_OF = {a: ('loose' if a in COMPILE_ARMS else 'tight')
           for a in ('A', 'ANVTX', 'B', 'C1', 'C2', 'C3', 'C1A', 'C1D', 'D', 'E', 'F', 'G', 'H')}
VALIDATION_SEEDS = (1001, 2002)
SDR_CLIPS = 8
SDR_THRESHOLD_DB = -0.1
STABILITY_KEYS = ('graph_break', 'unimplemented')


def compile_stable(before, after):
    """Recording/compilation stability (gate-02 f21): no NEW graph breaks or
    unimplemented ops, the same number of unique Dynamo graphs, and EVERY
    Inductor cudagraph counter unchanged. Per-call counters are ignored."""
    for k in STABILITY_KEYS:
        if (before or {}).get(k, {}) != (after or {}).get(k, {}):
            return False
    if (before or {}).get('stats', {}).get('unique_graphs') != (after or {}).get('stats', {}).get('unique_graphs'):
        return False
    b_ind = {k: v for k, v in (before or {}).get('inductor', {}).items() if 'cudagraph' in k}
    a_ind = {k: v for k, v in (after or {}).get('inductor', {}).items() if 'cudagraph' in k}
    return b_ind == a_ind


def recording_stable(a, b):
    """None when the observer is missing (unresolved); else identity-set equality."""
    if not a or not b:
        return None
    ta, tb = a.get('tree') or {}, b.get('tree') or {}
    if not (ta.get('available') and tb.get('available')):
        return None
    if ta.get('identity_mode') != 'registry' or tb.get('identity_mode') != 'registry':
        return None                      # address-based identities are not lifetime-safe
    return (compile_stable(a.get('counters'), b.get('counters')) and
            ta.get('manager_id') == tb.get('manager_id') and
            ta.get('node_ids') == tb.get('node_ids') and
            ta.get('function_ids') == tb.get('function_ids') and
            ta.get('roots') == tb.get('roots'))


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float('inf')


def _close(a, b, rel=1e-9, ab=1e-9):
    return _finite(a) and _finite(b) and abs(a - b) <= max(ab, rel * max(abs(a), abs(b)))


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def recomputed_timing(timing, iters, repeats):
    """Statistics recomputed from the recorded samples (gate-05 f64); None if
    the samples are not a complete, finite, positive registered payload."""
    blocks = (timing or {}).get('blocks') or []
    if len(blocks) != repeats:
        return None
    meds, thr = [], []
    for b in blocks:
        lat = b.get('latency_ms_per_iter') or []
        if len(lat) != iters or not all(_finite(x) and x > 0 for x in lat):
            return None
        if not (_finite(b.get('throughput_ms_per_pass')) and b['throughput_ms_per_pass'] > 0):
            return None
        meds.append(_median(lat))
        thr.append(b['throughput_ms_per_pass'])
    return {'block_medians_ms': meds, 'min_of_medians_ms': min(meds),
            'median_spread_ms': max(meds) - min(meds), 'throughput_best_ms': min(thr)}


def variant(rec):
    """Canonical effective-variant string, identical for single-arm and
    matched records (all four fields, fixed order, explicit None)."""
    return '|'.join(f'{k}={rec.get(k)}' for k in VARIANT_FIELDS)


def _blocks_ok(timing, iters, repeats):
    if not timing:
        return 'no timed region'
    if timing.get('repeats') != repeats or timing.get('iters_per_block') != iters:
        return (f"protocol {timing.get('repeats')}x{timing.get('iters_per_block')} "
                f"is not the registered {repeats}x{iters}")
    blocks = timing.get('blocks') or []
    if len(blocks) != repeats:
        return f'{len(blocks)} blocks recorded, {repeats} required'
    for b in blocks:
        if len(b.get('latency_ms_per_iter') or []) != iters:
            return 'a block did not record exactly the registered pass count'
        if not isinstance(b.get('throughput_ms_per_pass'), (int, float)):
            return 'a block lacks a throughput record'
    return None


def reference_hashes(rec, include_clips=False):
    """Checked reference hashes persisted by the run: the validation
    references every restart checks (the cross-restart join key) and, only
    when asked, the eager-A clip outputs the r1 SDR gate checked."""
    out = {}
    for p in (rec.get('validation') or {}).get('passes') or []:
        if p.get('ref_sha256'):
            out[f"seed{p.get('seed')}"] = p['ref_sha256']
    if include_clips:
        for row in (rec.get('sdr_gate') or {}).get('clips') or []:
            if row.get('a_output_sha256'):
                out[f"clip{row.get('index')}"] = row['a_output_sha256']
    return out


def validation_problems(rec):
    """The validation payload itself (gate-05 f61): both registered seeds,
    every pass ok with a finite max_abs_diff under the arm's registered tier
    tolerances, lineage hash persisted, and the stored status equal to the
    recomputed one."""
    out = []
    v = rec.get('validation') or {}
    arm = rec.get('arm')
    tier = TIER_OF.get(arm)
    if rec.get('forced_tier'):
        tier = rec['forced_tier']
    passes = v.get('passes') or []
    seeds = sorted(p.get('seed') for p in passes)
    if seeds != sorted(VALIDATION_SEEDS):
        out.append(f'validation seeds {seeds} are not the registered {list(VALIDATION_SEEDS)}')
    if tier and v.get('tier') != tier:
        out.append(f"validation tier {v.get('tier')} is not the registered {tier}")
    if tier and (v.get('rtol'), v.get('atol')) != TIERS[tier]:
        out.append('validation tolerances are not the registered tier values')
    ok_all = bool(passes)
    for p in passes:
        if not (p.get('ok') is True and _finite(p.get('max_abs_diff')) and p.get('ref_sha256')):
            ok_all = False
    recomputed = 'PASS' if ok_all else 'FAIL'
    if v.get('status') != 'PASS' or recomputed != 'PASS':
        out.append(f"validation {v.get('status', 'NONE')} (recomputed {recomputed})")
    return out


def sdr_recomputed(gate):
    """Recompute the SDR gate from its rows: exactly eight distinct clips,
    every delta finite, mean vs the registered threshold."""
    rows = (gate or {}).get('clips') or []
    idx = sorted(r.get('index') for r in rows)
    if len(rows) != SDR_CLIPS or len(set(idx)) != SDR_CLIPS:
        return None, f'{len(rows)} clip rows ({len(set(idx))} distinct); registered {SDR_CLIPS}'
    deltas = [r.get('delta_sdr_ref_db') for r in rows]
    if not all(_finite(d) for d in deltas):
        return None, 'a clip delta is missing or non-finite'
    if not all(r.get('a_output_sha256') for r in rows):
        return None, 'eager-A clip outputs not lineage-hashed'
    mean = sum(deltas) / len(deltas)
    status = 'PASS' if mean >= SDR_THRESHOLD_DB else 'FAIL'
    if gate.get('mean_delta_sdr_ref_db') is None or not _close(gate['mean_delta_sdr_ref_db'], mean, 1e-9, 1e-9):
        return None, 'stored mean delta differs from the recomputed one'
    if gate.get('status') != status:
        return None, f"stored SDR status {gate.get('status')} differs from the recomputed {status}"
    return status, None


def stability_recomputed(rec):
    """Recording stability recomputed from the stored phase snapshots."""
    snaps = rec.get('counter_snapshots') or {}
    seq = [snaps.get(k) for k in ('after_settle', 'after_validation', 'before_timing', 'after_timing', 'after_sdr')]
    seq = [x for x in seq if x]
    if len(seq) < 3:
        return None
    verdicts = [recording_stable(a, b) for a, b in zip(seq, seq[1:])]
    if any(v is None for v in verdicts):
        return None
    return all(verdicts)


def run_problems(rec, iters=REGISTERED_ITERS, repeats=REGISTERED_REPEATS,
                 require_manifest=True, require_timing=True):
    """All reasons a single-arm record is NOT a qualified measurement."""
    problems = []
    if rec is None:
        return ['record missing']
    if rec.get('status') != 'VALID':
        problems.append(f"status {rec.get('status')}")
    if rec.get('arm') == 'ANVTX':
        problems.append('attribution arm')
    if rec.get('nvtx_pass'):
        problems.append('traced/attribution run')
    if rec.get('planted_fault') or rec.get('planted_fault_const') is not None:
        problems.append('planted fault run')
    problems += validation_problems(rec)
    if require_manifest and not rec.get('manifest_id'):
        problems.append('not bound to a session manifest')
    if require_timing:
        b = _blocks_ok(rec.get('timing'), iters, repeats)
        if b:
            problems.append(b)
    if not rec.get('effective_variant') and any(k in rec for k in VARIANT_FIELDS):
        problems.append('effective variant not recorded')
    if rec.get('arm') in COMPILE_ARMS:
        st = stability_recomputed(rec)
        if rec.get('counters_stable') is not True or st is not True:
            problems.append('recording stability not affirmatively established from the stored snapshots')
        s = rec.get('settle') or {}
        if s.get('settle_capped') or s.get('stable') is False:
            problems.append('compile settle capped or unstable')
        g = rec.get('graph_gate') or {}
        if g.get('status') != 'PASS':
            problems.append(f"graph gate {g.get('status', 'NONE')}")
        if rec.get('arm') in ZERO_BREAK_ARMS and g.get('graph_break_count') not in (0,):
            problems.append('zero-break gate: graph_break_count is not 0')
    if require_timing and rec.get('timing'):
        rt = recomputed_timing(rec['timing'], iters, repeats)
        if rt is None:
            problems.append('timing samples incomplete or non-finite')
        else:
            for k_rec, k_rt in (('min_of_medians_ms', 'min_of_medians_ms'), ('median_spread_ms', 'median_spread_ms'),
                                ('throughput_ms_per_pass_best', 'throughput_best_ms')):
                if not _close(rec.get(k_rec), rt[k_rt], 1e-9, 1e-9):
                    problems.append(f'stored {k_rec} differs from the value recomputed from samples')
    if rec.get('sdr_gate') and rec['sdr_gate'].get('status') != 'REFERENCE_SAVED':
        st, why = sdr_recomputed(rec['sdr_gate'])
        if st != 'PASS':
            problems.append(f"SDR gate {why or st}")
    # stored-flag contradictions in BOTH directions (gate-05 f62)
    if rec.get('headline_eligible') is False and not problems:
        problems.append(f"stored eligibility flag says ineligible: {rec.get('ineligibility_reasons')}")
    return problems


def sdr_problems(rec, chunk, manifest_id, var):
    """The registered SDR qualification record for a cell: r1, same manifest,
    same variant, gate PASS at that operating point, clip outputs lineage-hashed."""
    if not rec or not rec.get('sdr_gate'):
        return ['no SDR record in r1']
    out = []
    g = rec['sdr_gate']
    if manifest_id and rec.get('manifest_id') != manifest_id:
        out.append('r1 SDR record belongs to another manifest')
    if var and variant(rec) != var:
        out.append('r1 SDR record has another effective variant')
    if rec.get('chunk') != chunk:
        out.append('r1 SDR record operating point mismatch')
    st, why = sdr_recomputed(g)
    if st != 'PASS':
        out.append(f"SDR gate {why or st}")
    if g.get('operating_point') != chunk:
        out.append('SDR record operating point mismatch')
    return out


def pair_problems(m, iters=REGISTERED_ITERS, repeats=REGISTERED_REPEATS):
    problems = []
    if m is None:
        return ['record missing']
    if m.get('status') != 'VALID':
        problems.append(f"status {m.get('status')}")
    if not m.get('manifest_id'):
        problems.append('not bound to a session manifest')
    if (m.get('iters'), m.get('repeats')) != (iters, repeats):
        problems.append(f"protocol {m.get('repeats')}x{m.get('iters')} is not the registered {repeats}x{iters}")
    js = m.get('joint_settle') or {}
    if js.get('stable') is not True:
        problems.append('joint settle not affirmatively stable')
    if js.get('new_breaks_during_joint_settle'):
        problems.append('new graph breaks during the joint settle')
    # every required phase transition recomputed from the stored snapshots (gate-05 f62)
    sn = m.get('snapshots') or {}
    chain = list(sn.get('joint_settle') or []) + [sn.get('before_validation'), sn.get('after_validation')] + list(sn.get('blocks') or [])
    chain = [x for x in chain if x]
    if len(chain) < 4 + repeats * 2:
        problems.append('matched recording snapshots incomplete')
    else:
        verdicts = [recording_stable(a, b) for a, b in zip(chain, chain[1:])]
        if any(v is None for v in verdicts) or not all(verdicts):
            problems.append('recording state changed (or unresolved) across a matched phase transition')
    if m.get('pair_eligible') is False and not problems:
        problems.append(f"stored pair flag says ineligible: {m.get('ineligibility_reasons')}")
    for arm, a in (m.get('arms') or {}).items():
        if a.get('build_status') != 'OK':
            problems.append(f'{arm}: build {a.get("build_status")}')
            continue
        v = a.get('validation') or {}
        if v.get('status') != 'PASS':
            problems.append(f"{arm}: validation {v.get('status', 'NONE')}")
        elif not all(p.get('ref_sha256') for p in v.get('passes') or []):
            problems.append(f'{arm}: references not lineage-checked')
        if (a.get('graph_gate') or {}).get('status') != 'PASS':
            problems.append(f'{arm}: graph gate not PASS')
        if (a.get('settle') or {}).get('settle_capped'):
            problems.append(f'{arm}: settle capped')
        blocks = [b for b in (m.get('blocks') or []) if b.get('arm') == arm]
        if len(blocks) != repeats or any(len(b.get('latency_ms_per_iter') or []) != iters for b in blocks):
            problems.append(f'{arm}: blocks do not hold the registered {repeats}x{iters}')
        elif not all(all(_finite(x) and x > 0 for x in b['latency_ms_per_iter']) for b in blocks):
            problems.append(f'{arm}: a latency sample is non-finite or non-positive')
        elif not all(_close(b.get('latency_median_ms'), _median(b['latency_ms_per_iter']), 1e-9, 1e-9) for b in blocks):
            problems.append(f'{arm}: a stored block median differs from its samples')
        if not all(b.get('counters_stable') is True for b in blocks):
            problems.append(f'{arm}: recording not affirmatively stable during a block')
        problems += [f'{arm}: {x}' for x in validation_problems({'arm': arm, 'validation': a.get('validation')})]
    return problems


def recomputed_matched_delta(m, iters=REGISTERED_ITERS, repeats=REGISTERED_REPEATS):
    """Matched delta (a - b per block, median, % of a) recomputed from samples."""
    a, b = m.get('pair') or (None, None)
    per = {}
    for arm in (a, b):
        blocks = sorted((x for x in (m.get('blocks') or []) if x.get('arm') == arm), key=lambda x: x.get('repeat', 0))
        if len(blocks) != repeats:
            return None
        per[arm] = [_median(x['latency_ms_per_iter']) for x in blocks if len(x.get('latency_ms_per_iter') or []) == iters]
        if len(per[arm]) != repeats:
            return None
    diffs = [pa - pb for pa, pb in zip(per[a], per[b])]
    return {'a_minus_b_per_block': diffs, 'median': _median(diffs), 'min': min(diffs), 'max': max(diffs),
            'median_pct_of_a': _median(diffs) / _median(per[a]) * 100.0}
