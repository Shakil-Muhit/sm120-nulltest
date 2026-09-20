"""Versioned, shared qualification validator (gate-04 f47/f55/f56).

Every consumer that publishes or audits a number re-derives a run's
eligibility from the RECORD ITSELF (protocol, validation, lineage,
recording stability, gate, manifest) — never from stored booleans.
"""

QUALIFY_VERSION = 2
REGISTERED_ITERS = 50
REGISTERED_REPEATS = 5
COMPILE_ARMS = ('C1', 'C2', 'C3', 'C1A', 'C1D', 'H')
ZERO_BREAK_ARMS = ('C3', 'H')
VARIANT_FIELDS = ('capture_scope', 'capture_adapter_active', 'compile_mode', 'stream_count')


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
    v = rec.get('validation') or {}
    if v.get('status') != 'PASS':
        problems.append(f"validation {v.get('status', 'NONE')}")
    else:
        if not all(p.get('ref_sha256') for p in v.get('passes') or []):
            problems.append('validation references not lineage-checked (no persisted hash)')
    if require_manifest and not rec.get('manifest_id'):
        problems.append('not bound to a session manifest')
    if require_timing:
        b = _blocks_ok(rec.get('timing'), iters, repeats)
        if b:
            problems.append(b)
    if not rec.get('effective_variant') and any(k in rec for k in VARIANT_FIELDS):
        problems.append('effective variant not recorded')
    if rec.get('arm') in COMPILE_ARMS:
        if rec.get('counters_stable') is not True:
            problems.append('recording stability not affirmatively established')
        s = rec.get('settle') or {}
        if s.get('settle_capped'):
            problems.append('compile settle capped')
        g = rec.get('graph_gate') or {}
        if g.get('status') != 'PASS':
            problems.append(f"graph gate {g.get('status', 'NONE')}")
    if rec.get('sdr_gate') and rec['sdr_gate'].get('status') not in ('PASS', 'REFERENCE_SAVED'):
        problems.append(f"SDR gate {rec['sdr_gate'].get('status')}")
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
    if g.get('status') != 'PASS':
        out.append(f"SDR gate {g.get('status')}")
    if not all(row.get('a_output_sha256') for row in g.get('clips') or []):
        out.append('eager-A clip outputs not lineage-hashed')
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
        if not all(b.get('counters_stable') is True for b in blocks):
            problems.append(f'{arm}: recording not affirmatively stable during a block')
    return problems
