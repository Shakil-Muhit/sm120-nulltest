"""Versioned, shared qualification validator (gate-04 f47/f55/f56).

Every consumer that publishes or audits a number re-derives a run's
eligibility from the RECORD ITSELF (protocol, validation, lineage,
recording stability, gate, manifest) — never from stored booleans.
"""

QUALIFY_VERSION = 5
REGISTERED_ITERS = 50
REGISTERED_REPEATS = 5
COMPILE_ARMS = ('C1', 'C2', 'C3', 'C1A', 'C1D', 'H', 'C3F', 'C3P', 'C3M', 'C1P', 'C3N')
ZERO_BREAK_ARMS = ('C3', 'H', 'C3F', 'C3P', 'C3M', 'C3N')
VARIANT_FIELDS = ('capture_scope', 'capture_adapter_active', 'compile_mode', 'stream_count')
TIERS = {'loose': (1e-2, 1e-3), 'tight': (1e-4, 1e-5)}
TIER_OF = {a: ('loose' if a in COMPILE_ARMS else 'tight')
           for a in ('A', 'ANVTX', 'B', 'C1', 'C2', 'C3', 'C1A', 'C1D', 'D', 'E', 'F', 'G', 'H',
                     'C3F', 'C3P', 'C3M', 'C1P', 'C3N')}


def base(arm):
    """'C3#2' (second instance in an A/A pair) -> 'C3'."""
    return str(arm).split('#')[0]
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


def snapshot_complete(x):
    """A recording snapshot must carry complete counters and observer fields."""
    if not isinstance(x, dict):
        return False
    c, t = x.get('counters'), x.get('tree')
    if not isinstance(c, dict) or not isinstance(t, dict):
        return False
    # dynamo counters only carry keys that occurred: graph_break / unimplemented
    # are absent when zero; stats and inductor are always present
    if not (isinstance(c.get('stats'), dict) and isinstance(c.get('inductor'), dict)):
        return False
    ug = c['stats'].get('unique_graphs')
    if not (isinstance(ug, int) and not isinstance(ug, bool) and ug >= 0):
        return False
    for k in ('graph_break', 'unimplemented'):
        if k in c and not isinstance(c[k], dict):
            return False
    if not t.get('available') or t.get('identity_mode') != 'registry':
        return False
    ids, fids = t.get('node_ids'), t.get('function_ids')
    if not (isinstance(ids, list) and all(isinstance(i, int) and not isinstance(i, bool) for i in ids)
            and len(set(ids)) == len(ids)):
        return False
    if not (isinstance(t.get('nodes'), int) and isinstance(t.get('roots'), int)
            and t['nodes'] == len(ids) and 0 <= t['roots'] <= t['nodes']):
        return False
    if not (isinstance(fids, list) and all(isinstance(f, str) for f in fids)):
        return False
    return isinstance(t.get('manager_id'), int) and not isinstance(t.get('manager_id'), bool)


def recording_stable(a, b):
    """None when the observer is missing or a snapshot is incomplete
    (unresolved); else identity-set equality."""
    if not (snapshot_complete(a) and snapshot_complete(b)):
        return None
    ta, tb = a['tree'], b['tree']
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
    the samples are not a complete, finite, positive registered payload or if
    any NESTED stored summary contradicts its own samples (gate-06 f88)."""
    blocks = (timing or {}).get('blocks') or []
    if len(blocks) != repeats:
        return None
    if [b.get('repeat') for b in blocks] != list(range(repeats)):
        return None
    meds, thr = [], []
    for b in blocks:
        lat = b.get('latency_ms_per_iter') or []
        if len(lat) != iters or not all(_finite(x) and x > 0 for x in lat):
            return None
        if not (_finite(b.get('throughput_ms_per_pass')) and b['throughput_ms_per_pass'] > 0):
            return None
        srt = sorted(lat)
        if not (_close(b.get('latency_median_ms'), _median(lat), 1e-9, 1e-9)
                and _close(b.get('latency_min_ms'), srt[0], 1e-9, 1e-9)
                and _close(b.get('latency_max_ms'), srt[-1], 1e-9, 1e-9)):
            return None
        if not (_finite(b.get('unix_start')) and _finite(b.get('unix_end')) and b['unix_end'] >= b['unix_start']):
            return None
        meds.append(_median(lat))
        thr.append(b['throughput_ms_per_pass'])
    t = timing
    rt = {'block_medians_ms': meds, 'min_of_medians_ms': min(meds),
          'median_spread_ms': max(meds) - min(meds), 'throughput_best_ms': min(thr)}
    lm = t.get('latency_medians_ms')
    if lm is not None and not (isinstance(lm, list) and len(lm) == repeats
                               and all(_close(a, b, 1e-9, 1e-9) for a, b in zip(lm, meds))):
        return None
    sd = t.get('serialization_delta_pct')
    if sd is not None and not (isinstance(sd, list) and len(sd) == repeats and all(_finite(x) for x in sd)):
        return None
    for k_t, k_rt in (('min_of_medians_ms', 'min_of_medians_ms'), ('median_spread_ms', 'median_spread_ms'),
                      ('throughput_ms_per_pass_best', 'throughput_best_ms')):
        if k_t in t and not _close(t.get(k_t), rt[k_rt], 1e-9, 1e-9):
            return None
    if t.get('repeats') != repeats or t.get('iters_per_block') != iters:
        return None
    return rt


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
    if v.get('status') == 'REFERENCE_SAVED':
        return ['validation REFERENCE_SAVED (a reference-writing run is never a measurement)']
    arm = rec.get('arm')
    tier = TIER_OF.get(base(arm))
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
    if rec.get('forced_tier') and rec.get('planted_fault_const') is None:
        out.append('tier override on a non-planted record')
    ok_all = bool(passes)
    for p in passes:
        # every per-pass invariant (gate-06 f80): shapes equal, both operands finite,
        # the pass's own tolerances equal the registered tier, zero violating
        # elements, and the stored ok flag consistent with all of that
        shape = p.get('shape_out')
        shape_ok = (isinstance(shape, list) and shape and all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in shape))
        numel = 1
        for d in (shape or []):
            numel *= d if isinstance(d, int) else 0
        good = (p.get('ok') is True and _finite(p.get('max_abs_diff')) and p['max_abs_diff'] >= 0
                and p.get('ref_sha256') and shape_ok and p.get('shape_ref') == shape
                and p.get('n_elements') == numel and _finite(p.get('max_abs_ref')) and p['max_abs_ref'] >= 0
                and p.get('finite_out') is True and p.get('finite_ref') is True
                and (tier is None or (p.get('rtol'), p.get('atol')) == TIERS[tier])
                and p.get('violations') == 0
                and p['max_abs_diff'] <= p['atol'] + p['rtol'] * p['max_abs_ref'] + 1e-12)
        if not good:
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
    if idx != list(range(SDR_CLIPS)):
        return None, f'clip indices {idx} are not the registered 0..{SDR_CLIPS - 1}'
    if gate.get('n_clips') != SDR_CLIPS:
        return None, 'stored n_clips differs from the registered clip count'
    # paired terms recomputed (gate-06 f81): both SDR_ref terms finite, the
    # stored delta equal to their difference
    deltas = []
    for r in rows:
        a, b = r.get('sdr_ref_db'), r.get('sdr_ref_A_db')
        if not (_finite(a) and _finite(b)):
            return None, 'a paired SDR_ref term is missing or non-finite'
        if not _close(r.get('delta_sdr_ref_db'), a - b, 1e-9, 1e-9):
            return None, 'a stored clip delta differs from its paired terms'
        deltas.append(a - b)
    if not all(r.get('a_output_sha256') for r in rows):
        return None, 'eager-A clip outputs not lineage-hashed'
    mean = sum(deltas) / len(deltas)
    status = 'PASS' if mean >= SDR_THRESHOLD_DB else 'FAIL'
    if gate.get('mean_delta_sdr_ref_db') is None or not _close(gate['mean_delta_sdr_ref_db'], mean, 1e-9, 1e-9):
        return None, 'stored mean delta differs from the recomputed one'
    if gate.get('status') != status:
        return None, f"stored SDR status {gate.get('status')} differs from the recomputed {status}"
    return status, None


def required_phases(rec):
    """The exact snapshot schema of a compiled run (gate-06 f82)."""
    phases = ['after_settle', 'after_validation']
    if rec.get('timing'):
        phases += ['before_timing', 'after_timing']
    if rec.get('sdr_gate'):
        phases.append('after_sdr')
    return phases


def stability_recomputed(rec):
    """Recording stability recomputed from the stored phase snapshots; every
    required phase must be present and complete, else None (unresolved)."""
    snaps = rec.get('counter_snapshots') or {}
    seq = [snaps.get(k) for k in required_phases(rec)]
    if any(not snapshot_complete(x) for x in seq) or len(seq) < 2:
        return None
    verdicts = [recording_stable(a, b) for a, b in zip(seq, seq[1:])]
    if any(v is None for v in verdicts):
        return None
    return all(verdicts)


def gate_problems(arm, gate, snapshot_after_settle):
    """The complete graph-gate predicate, shared by every consumer (gate-06 f83):
    counters, suppress_errors, zero-break rule for C3/H, and reconciliation with
    the after-settle snapshot's counters."""
    out = []
    g = gate or {}
    if g.get('status') != 'PASS':
        out.append(f"graph gate {g.get('status', 'NONE')}")
    if g.get('suppress_errors') is not False:
        out.append('graph gate: suppress_errors not affirmatively False')
    if not isinstance(g.get('graph_break_count'), int) or not isinstance(g.get('unimplemented_count'), int):
        out.append('graph gate counters missing')
        return out
    if base(arm) in ZERO_BREAK_ARMS and (g['graph_break_count'] != 0 or g['unimplemented_count'] != 0
                                  or g.get('require_zero_breaks') is not True):
        out.append('zero-break gate: graph breaks / unimplemented ops not 0 (or rule not applied)')
    c = (snapshot_after_settle or {}).get('counters') if isinstance(snapshot_after_settle, dict) else None
    if not isinstance(c, dict):
        out.append('graph gate: no after-settle snapshot to reconcile against')
    else:
        if sum(c.get('graph_break', {}).values()) != g['graph_break_count'] or \
                sum(c.get('unimplemented', {}).values()) != g['unimplemented_count']:
            out.append('graph gate counters disagree with the after-settle snapshot')
    return out


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
    if base(rec.get('arm')) in COMPILE_ARMS:
        st = stability_recomputed(rec)
        if rec.get('counters_stable') is not True or st is not True:
            problems.append('recording stability not affirmatively established from the stored snapshots')
        s = rec.get('settle') or {}
        if s.get('settle_capped') or s.get('stable') is False:
            problems.append('compile settle capped or unstable')
        problems += gate_problems(rec.get('arm'), rec.get('graph_gate'),
                                  (rec.get('counter_snapshots') or {}).get('after_settle'))
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


def trace_problems(rec, iters=REGISTERED_ITERS):
    """A one-block traced or perturbation run (never a headline): everything a
    measurement needs except the multi-block protocol; nvtx is allowed."""
    if rec is None:
        return ['record missing']
    problems = []
    if rec.get('status') != 'VALID':
        problems.append(f"status {rec.get('status')}")
    if rec.get('planted_fault') or rec.get('planted_fault_const') is not None:
        problems.append('planted fault run')
    problems += validation_problems(rec)
    if not rec.get('manifest_id'):
        problems.append('not bound to a session manifest')
    if recomputed_timing(rec.get('timing'), iters, 1) is None:
        problems.append('one-block timing samples incomplete, non-finite, or contradicting stored summaries')
    if not rec.get('effective_variant'):
        problems.append('effective variant not recorded')
    if base(rec.get('arm')) in COMPILE_ARMS:
        st = stability_recomputed(rec)
        if rec.get('counters_stable') is not True or st is not True:
            problems.append('recording stability not affirmatively established from the stored snapshots')
        if (rec.get('settle') or {}).get('settle_capped'):
            problems.append('compile settle capped')
        problems += gate_problems(rec.get('arm'), rec.get('graph_gate'),
                                  (rec.get('counter_snapshots') or {}).get('after_settle'))
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
    rounds = int(js.get('rounds') or 0)
    jsn = list(sn.get('joint_settle') or [])
    chain = jsn + [sn.get('before_validation'), sn.get('after_validation')] + list(sn.get('blocks') or [])
    # exact schema (gate-06 f82): rounds+2 joint-settle snapshots, the two validation
    # snapshots, and one snapshot per timed block — every one complete
    if rounds < 1 or len(jsn) != rounds + 2 or len(list(sn.get('blocks') or [])) != repeats * 2 \
            or any(not snapshot_complete(x) for x in chain):
        problems.append('matched recording snapshots incomplete or malformed')
    else:
        verdicts = [recording_stable(a, b) for a, b in zip(chain, chain[1:])]
        if any(v is None for v in verdicts) or not all(verdicts):
            problems.append('recording state changed (or unresolved) across a matched phase transition')
    if m.get('pair_eligible') is False and not problems:
        problems.append(f"stored pair flag says ineligible: {m.get('ineligibility_reasons')}")
    # exactly the two declared arms, each with its full numerical validation
    # payload and stable blocks (gate-08 f112)
    if sorted((m.get('arms') or {}).keys()) != sorted(x for x in (m.get('pair') or []) if x):
        problems.append('matched record does not carry exactly the two declared arms')
    for arm, a in (m.get('arms') or {}).items():
        if a.get('build_status') != 'OK':
            problems.append(f'{arm}: build {a.get("build_status")}')
            continue
        problems += [f'{arm}: {x}' for x in validation_problems({'arm': arm, 'validation': a.get('validation')})]
        arm_blocks = [b for b in (m.get('blocks') or []) if b.get('arm') == arm]
        if not arm_blocks or not all(b.get('counters_stable') is True for b in arm_blocks):
            problems.append(f'{arm}: recording not affirmatively stable during every block')
        # each arm reconciles its gate against the snapshot taken right after its own
        # gate (the next arm's build clears the global counters)
        problems += [f'{arm}: {x}' for x in gate_problems(arm, a.get('graph_gate'), a.get('gate_snapshot'))]
        if (a.get('settle') or {}).get('settle_capped'):
            problems.append(f'{arm}: settle capped')
        blocks = sorted((b for b in (m.get('blocks') or []) if b.get('arm') == arm), key=lambda b: b.get('repeat', -1))
        if len(blocks) != repeats or any(len(b.get('latency_ms_per_iter') or []) != iters for b in blocks) \
                or [b.get('repeat') for b in blocks] != list(range(repeats)):
            problems.append(f'{arm}: blocks do not hold the registered {repeats}x{iters} with unique repeat ids')
        elif not all(_finite(b.get('throughput_ms_per_pass')) and b['throughput_ms_per_pass'] > 0 for b in blocks):
            problems.append(f'{arm}: a block throughput is non-finite')
        else:
            meds = [_median(b['latency_ms_per_iter']) for b in blocks]
            if not (all(_close(b.get('latency_median_ms'), mm, 1e-9, 1e-9) for b, mm in zip(blocks, meds))
                    and isinstance(a.get('latency_medians_ms'), list) and len(a['latency_medians_ms']) == repeats
                    and all(_close(x, y, 1e-9, 1e-9) for x, y in zip(a['latency_medians_ms'], meds))
                    and _close(a.get('min_of_medians_ms'), min(meds), 1e-9, 1e-9)
                    and _close(a.get('median_spread_ms'), max(meds) - min(meds), 1e-9, 1e-9)):
                problems.append(f'{arm}: a stored matched summary differs from its samples')
    # chronological alternation (gate-07 f103): finite, strictly increasing block
    # timestamps whose (repeat, arm) order is exactly (0,a),(0,b),(1,a),(1,b),...
    a_, b_ = (m.get('pair') or [None, None])[:2]
    allb = list(m.get('blocks') or [])
    if allb and all(_finite(x.get('unix_start')) and _finite(x.get('unix_end')) and x['unix_end'] >= x['unix_start'] for x in allb):
        order = sorted(allb, key=lambda x: x['unix_start'])
        starts = [x['unix_start'] for x in order]
        ends = [x['unix_end'] for x in order]
        expected = [(r, arm) for r in range(repeats) for arm in (a_, b_)]
        if [(x.get('repeat'), x.get('arm')) for x in order] != expected:
            problems.append('matched blocks are not in the registered alternating order')
        if any(starts[i + 1] < ends[i] for i in range(len(order) - 1)):
            problems.append('matched block timestamps overlap')
    elif allb:
        problems.append('a matched block lacks finite timestamps')
    d = m.get('matched_delta_ms') or {}
    rd = recomputed_matched_delta(m, iters, repeats)
    if rd is None:
        problems.append('matched delta cannot be recomputed from samples')
    else:
        ok = (isinstance(d.get('a_minus_b_per_block'), list) and len(d['a_minus_b_per_block']) == repeats
              and all(_close(x, y, 1e-9, 1e-9) for x, y in zip(d['a_minus_b_per_block'], rd['a_minus_b_per_block']))
              and all(_close(d.get(k), rd[k], 1e-9, 1e-9) for k in ('median', 'min', 'max', 'median_pct_of_a')))
        if not ok:
            problems.append('a stored matched delta summary differs from the value recomputed from samples')
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
        if len(blocks) != repeats or [x.get('repeat') for x in blocks] != list(range(repeats)):
            return None
        if any(len(x.get('latency_ms_per_iter') or []) != iters or
               not all(_finite(v) and v > 0 for v in x['latency_ms_per_iter']) for x in blocks):
            return None
        per[arm] = [_median(x['latency_ms_per_iter']) for x in blocks]
    diffs = [pa - pb for pa, pb in zip(per[a], per[b])]
    return {'a_minus_b_per_block': diffs, 'median': _median(diffs), 'min': min(diffs), 'max': max(diffs),
            'median_pct_of_a': _median(diffs) / _median(per[a]) * 100.0}
