#!/bin/bash
# SESSION 3 (prereg_amendments.md v0.2 — Phase 3 experiment registrations E1–E9).
# Order (handoff §4): E6 → E3 → E2 → E1 → E9 → E4a → E5 → E7, then post-processing,
# collection and the completeness verdict. Same deadline / reserve / terminal-record
# rules as sessions 1–2 (scripts/session_lib.sh). E5 and E7 are optional under the
# deadline (registered: skipping them is not a failure). SHAKEDOWN=1 (engineering
# host): the E9 held-out lengths are recorded UNTESTED with the registered reason and
# are never run; the audit verdict then carries shakedown=true and is never evidence.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/session_lib.sh"
HEADLINE_ARMS="${HEADLINE_ARMS:-C3 H A C1}"
HEADLINE_RESTARTS="${HEADLINE_RESTARTS:-5}"
E5_BUDGET_S="${E5_BUDGET_S:-3600}"
E5_VARIANTS="${E5_VARIANTS:-d1_mfull d1_mquarter d6_mfull d6_mquarter}"
E5_RESTARTS="${E5_RESTARTS:-2}"
SHAKEDOWN="${SHAKEDOWN:-0}"
HELD_OUT="176400 264600"
export REQUIRED_RESTARTS="$HEADLINE_RESTARTS"
REG="$REPO_DIR/docs/phase3/e9/e9_cost_model.json"
mkdir -p "$EVID" "$REF" "$EVID/traces"
cd "$REPO_DIR"
exec > >(tee -a "$EVID/session3_console.log") 2>&1
echo "SESSION3 START $(date -Is) iters=$ITERS repeats=$REPEATS restarts=$RESTARTS headline_restarts=$HEADLINE_RESTARTS budget=${SESSION_BUDGET_S}s reserve=${EXTRACT_RESERVE_S}s shakedown=$SHAKEDOWN pod=${SESSION_POD:-0}"
[ -z "${TMUX:-}" ] && echo "WARNING: not inside tmux"
[ "${SESSION_POD:-0}" = "1" ] || SHAKEDOWN=1      # gate-09 f136: any off-pod run is engineering data
python3 -c "import json;json.dump({'shakedown': '$SHAKEDOWN'=='1', 'session_pod': '${SESSION_POD:-0}'=='1', 'headline_restarts': $HEADLINE_RESTARTS, 'pair_restarts': $RESTARTS, 'e5_variants': '$E5_VARIANTS'.split(), 'e5_restarts': $E5_RESTARTS, 'e9_registration': '$REG', 'e9_registration_sha256': __import__('hashlib').sha256(open('$REG','rb').read()).hexdigest()}, open('$EVID/session3_config.json','w'), indent=1)"

echo "=== [0] preflight ==="
command -v nsys >/dev/null || { echo "FATAL: nsys missing"; exit 1; }
nsys profile --help 2>/dev/null | grep -q 'cuda-graph-trace' || { echo "FATAL: nsys lacks --cuda-graph-trace"; exit 1; }
nsys --version | tee "$EVID/nsys_version_session3.txt"
python - <<'PYEOF' || { echo "FATAL: preflight imports failed"; exit 1; }
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'fullmodel'))
import torch, triton, fullarms, foldnorm, precast, variants, e_tools, boundary_bench
import torch._inductor.config as icfg
assert hasattr(icfg, 'freezing') and hasattr(icfg, 'graph_partition'), 'Inductor knobs missing'
print('torch', torch.__version__, 'triton', triton.__version__, 'arms', len(fullarms.ARMS), 'chunks', fullarms.CHUNKS)
PYEOF
if [ ! -s "$EVID/session_manifest.json" ]; then
    python - "$EVID" session3 <<'PYEOF' || { echo "FATAL: provenance / session manifest failed"; exit 1; }
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'fullmodel'))
import loader, provenance, timing
rec = provenance.create_manifest(sys.argv[1], loader.paths_from_env(), timing.env_fingerprint(), sys.argv[2])
print('manifest', rec['manifest_id'])
PYEOF
else
    python - "$EVID" <<'PYEOF' || { echo "FATAL: session manifest does not match this tree"; exit 1; }
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'fullmodel'))
import loader, provenance, timing
print('manifest ok', provenance.load_manifest(sys.argv[1], loader.paths_from_env(), timing.env_fingerprint())['manifest_id'])
PYEOF
fi
clocks_start "$EVID/clocks_session3.csv"
MID=$(python3 -c "import json;print(json.load(open('$EVID/session_manifest.json'))['manifest_id'])")

echo "=== [0a] eager-A references (validation + SDR clip outputs) at 352800 and 4410 ==="
for chunk in 352800 4410; do
    run_one A "$chunk" ref --save-reference --sdr --skip-timing
done
if [ "${SESSION_POD:-0}" = "1" ] && [ "$SHAKEDOWN" != "1" ]; then
    for chunk in $HELD_OUT; do
        run_one A "$chunk" ref --save-reference --skip-timing      # E9: validation references only, on the pod only
    done
fi
PREREQ_OK=1; PREREQ_REASON=""
for f in ref_A_c352800_s1001.pt ref_A_c352800_s2002.pt ref_A_c4410_s1001.pt ref_A_c4410_s2002.pt \
         sdr_A_c352800_clip00.pt sdr_A_c352800_clip07.pt sdr_A_c4410_clip00.pt sdr_A_c4410_clip07.pt manifest_id.txt; do
    [ -s "$REF/$f" ] || { echo "reference $f missing -> every validated measurement is UNTESTED (session INCOMPLETE)"; PREREQ_OK=0; PREREQ_REASON="references unavailable"; }
done

echo "=== [0b] planted-fault validator checks (both must FAIL) ==="
run_one A 4410 planted_tight --planted-fault --skip-timing
run_one A 4410 planted_loose --planted-fault --planted-const 0.05 --force-tier loose --skip-timing
python3 - "$EVID/full_A_c4410_planted_tight.json" 0.001 "$EVID/full_A_c4410_planted_loose.json" 0.05 <<'PYEOF' || { echo "a validator did not catch its planted fault -> every validated measurement is UNTESTED"; PREREQ_OK=0; PREREQ_REASON="planted fault not caught"; }
import json, sys
ok_all = True
for path, const in ((sys.argv[1], float(sys.argv[2])), (sys.argv[3], float(sys.argv[4]))):
    r = json.load(open(path)); v = r.get('validation') or {}; passes = v.get('passes') or []
    diffs = [p.get('max_abs_diff') for p in passes]
    ok = (v.get('status') == 'FAIL' and len(passes) == 2 and all(not p['ok'] for p in passes)
          and all(d is not None and abs(d - const) < 1e-4 * const + 1e-6 for d in diffs))
    print(f"planted {const} tier={v.get('tier')}: status={v.get('status')} max_abs_diff={diffs} -> {'CAUGHT' if ok else 'NOT CAUGHT'}")
    ok_all &= ok
sys.exit(0 if ok_all else 1)
PYEOF

echo "=== [0c] C3 operator qualification per registered frame count ==="
C3_QUAL_OK=1
C3_QUAL_FRAMES="11 51 201 801"; [ "$SHAKEDOWN" = "1" ] || C3_QUAL_FRAMES="11 51 201 801 401 601"   # held-out frame counts qualified on the pod only
run_aux c3_operator_qualification 600 -- python - "$EVID/c3_operator_qualification.json" $C3_QUAL_FRAMES <<'PYEOF' || { echo "C3 operator qualification FAILED/UNTESTED -> C3-derived work is UNTESTED"; C3_QUAL_OK=0; }
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'fullmodel'))
import fullarms
rec = fullarms.verify_c3_operator([int(x) for x in sys.argv[2:]])
json.dump(rec, open(sys.argv[1], 'w'), indent=1)
assert all(v.get('status') == 'PASS' for v in rec.values()), rec
print('C3 operator qualification PASS for frames', sorted(rec))
PYEOF
python3 -c "import json,sys; r=json.load(open('$EVID/c3_operator_qualification.json')); assert set(r)==set('$C3_QUAL_FRAMES'.split()) and all(v['status']=='PASS' for v in r.values())" 2>/dev/null \
    || { echo "C3 qualification JSON invalid -> C3-derived work is UNTESTED"; C3_QUAL_OK=0; }
needs_c3() { case "$1" in C3|H|C3F|C3P|C3M|C3N) return 0;; *) return 1;; esac; }
run_one_q() {
    if [ "$PREREQ_OK" != "1" ]; then terminal_json "$EVID/full_$1_c$2_$3.json" "$1" "$2" "$3" UNTESTED "$PREREQ_REASON"; return; fi
    if [ "$C3_QUAL_OK" != "1" ] && needs_c3 "$1"; then terminal_json "$EVID/full_$1_c$2_$3.json" "$1" "$2" "$3" UNTESTED "C3 operator qualification failed"; return; fi
    run_one "$@"
}
run_matched_q() {
    if [ "$PREREQ_OK" != "1" ] || { [ "$C3_QUAL_OK" != "1" ] && { needs_c3 "$1" || needs_c3 "$2"; }; }; then
        local why="C3 operator qualification failed"; [ "$PREREQ_OK" = "1" ] || why="$PREREQ_REASON"
        python3 -c "import json;json.dump({'pair':['$1','$2'],'chunk':$3,'tag':'$4','status':'UNTESTED','error':'$why','pair_eligible':False},open('$EVID/matched_$1_$2_c$3_$4.json','w'),indent=1)"; return
    fi
    run_matched "$@"
}
run_traced_q() {
    if [ "$PREREQ_OK" != "1" ] || { [ "$C3_QUAL_OK" != "1" ] && needs_c3 "$1"; }; then
        local why="C3 operator qualification failed"; [ "$PREREQ_OK" = "1" ] || why="$PREREQ_REASON"
        terminal_json "$EVID/full_$1_c$2_$3.json" "$1" "$2" "$3" UNTESTED "$why"; return 1
    fi
    run_traced "$@"
}
# trace_arm ARM CHUNK TAG [extra...]: node-level trace + graph verification + timing views + cast count
trace_arm() {
    local arm="$1" chunk="$2" tag="$3"; shift 3
    run_traced_q "$arm" "$chunk" "$tag" node "$@" || return 1
    local base="$EVID/traces/trace_${arm}_${chunk}_${tag}"
    [ -s "${base}_cuda_gpu_trace.csv" ] || return 1
    local sfx=""; [ "$tag" != "trace" ] && sfx="_${tag}"
    rm -f "$EVID/traces/graph_verify_${arm}_${chunk}${sfx}.json" "$EVID/traces/timing_views_${arm}_${chunk}${sfx}.json" \
          "$EVID/traces/cast_count_${arm}_${chunk}${sfx}.json"
    run_aux "graph_verify_${arm}_${chunk}${sfx}" 300 -- python "$REPO_DIR/fullmodel/graph_verify.py" --base "$base" --arm "$arm" \
        --expected-passes "$ITERS" --out "$EVID/traces/graph_verify_${arm}_${chunk}${sfx}.json" || true
    run_aux "timing_views_${arm}_${chunk}${sfx}" 300 -- python "$REPO_DIR/fullmodel/timing_views.py" --base "$base" --arm "$arm" \
        --out "$EVID/traces/timing_views_${arm}_${chunk}${sfx}.json" || true
    local dump="$EVID/traces/dump_${arm}_${chunk}_${tag}"; [ -d "$dump" ] || dump=""
    rm -f "$EVID/traces/ideal_costs_${arm}_${chunk}${sfx}.json" "$EVID/traces/ideal_costs_${arm}_${chunk}${sfx}.csv" "$EVID/traces/template_count_${arm}_${chunk}${sfx}.json"
    run_aux "ideal_costs_${arm}_${chunk}${sfx}" 600 -- python "$REPO_DIR/fullmodel/ideal_costs.py" --base "$base" --arm "$arm" --chunk "$chunk" \
        ${dump:+--dump-dir "$dump"} --passes "$ITERS" --out "$EVID/traces/ideal_costs_${arm}_${chunk}${sfx}.json" || true
    run_aux "cast_count_${arm}_${chunk}${sfx}" 120 -- python "$REPO_DIR/fullmodel/e_tools.py" cast_count \
        --ideal-costs "$EVID/traces/ideal_costs_${arm}_${chunk}${sfx}.json" --dump-dir "${dump:-/nonexistent}" \
        --out "$EVID/traces/cast_count_${arm}_${chunk}${sfx}.json" || true
    run_aux "template_count_${arm}_${chunk}${sfx}" 120 -- python "$REPO_DIR/fullmodel/e_tools.py" template_count \
        --ideal-costs "$EVID/traces/ideal_costs_${arm}_${chunk}${sfx}.json" --out "$EVID/traces/template_count_${arm}_${chunk}${sfx}.json" || true
}
untested_pair() {   # A B CHUNK TAG REASON
    python3 -c "import json;json.dump({'pair':['$1','$2'],'chunk':$3,'tag':'$4','status':'UNTESTED','error':'$5','pair_eligible':False},open('$EVID/matched_$1_$2_c$3_$4.json','w'),indent=1)"
}

echo "=== [1] E6: A/A control (C3 vs C3, block-alternating) + $HEADLINE_RESTARTS restarts of $HEADLINE_ARMS at both operating points ==="
for chunk in 352800 4410; do
    run_matched_q C3 C3 "$chunk" m1
done
for r in $(seq 1 "$HEADLINE_RESTARTS"); do
    for chunk in 352800 4410; do
        for arm in $HEADLINE_ARMS; do
            extra=""; [ "$r" = "1" ] && extra="--sdr"
            run_one_q "$arm" "$chunk" "r$r" $extra
        done
    done
done
rm -f "$EVID/aa_stats.json"
run_aux aa_stats 120 -- python "$REPO_DIR/fullmodel/e_tools.py" aa_stats --evidence-dir "$EVID" --out "$EVID/aa_stats.json" || true

echo "=== [2] E3: C1 vs C1P (Inductor graph partition) @352800, $RESTARTS matched restarts + node trace ==="
for r in $(seq 1 "$RESTARTS"); do
    run_matched_q C1 C1P 352800 "m$r"
done
trace_arm C1P 352800 trace || true

echo "=== [3] E2: C3 vs C3M (repair x max-autotune), both operating points + traces + SDR ==="
for chunk in 352800 4410; do
    for r in $(seq 1 "$RESTARTS"); do
        run_matched_q C3 C3M "$chunk" "m$r"
    done
done
T_TRACE_AUTOTUNE="${T_TRACE_AUTOTUNE:-3000}"   # max-autotune traces recompile with caches disabled under nsys (autotuning from scratch)
for chunk in 352800 4410; do
    run_one_q C3M "$chunk" r1 --sdr
    T_COMPILE_SAVED="$T_COMPILE"; T_COMPILE="$T_TRACE_AUTOTUNE"
    trace_arm C3M "$chunk" trace || true
    T_COMPILE="$T_COMPILE_SAVED"
done

echo "=== [4] E1: C3 vs C3F (freezing), both operating points + traces (C3 and C3F) + SDR; C3P contingency ==="
for chunk in 352800 4410; do
    for r in $(seq 1 "$RESTARTS"); do
        run_matched_q C3 C3F "$chunk" "m$r"
    done
done
for chunk in 352800 4410; do
    run_one_q C3F "$chunk" r1 --sdr
    trace_arm C3 "$chunk" trace || true
    trace_arm C3F "$chunk" trace || true
done
rm -f "$EVID/e1_branch.json"
python3 - "$EVID" <<'PYEOF'
import glob, json, os, sys
ev = sys.argv[1]
def load(p):
    try: return json.load(open(p))
    except Exception: return None
need, why = False, []
sdr_fail = []
for c in (352800, 4410):
    r1 = load(os.path.join(ev, f'full_C3F_c{c}_r1.json')) or {}
    if ((r1.get('sdr_gate') or {}).get('status')) == 'FAIL':
        sdr_fail.append(c)
for c in (352800, 4410):
    c3 = load(os.path.join(ev, 'traces', f'cast_count_C3_{c}.json')); f = load(os.path.join(ev, 'traces', f'cast_count_C3F_{c}.json'))
    if not c3 or not f:
        need = True; why.append(f'{c}: cast counts unavailable'); continue
    if f['cast_kernels_per_pass'] >= 0.5 * c3['cast_kernels_per_pass']:
        need = True; why.append(f"{c}: C3F still runs {f['cast_kernels_per_pass']:.1f} of C3's {c3['cast_kernels_per_pass']:.1f} cast kernels per pass")
    ms = [load(p) for p in sorted(glob.glob(os.path.join(ev, f'matched_C3_C3F_c{c}_m*.json')))]
    elig = [m for m in ms if m and m.get('pair_eligible') and m.get('matched_delta_ms')]
    if not elig:
        need = True; why.append(f'{c}: no eligible C3/C3F matched pair (freezing did not compose)')
    else:
        import statistics
        gain = statistics.median([m['matched_delta_ms']['median'] for m in elig])     # C3 - C3F, + = C3F faster
        if gain <= 0:
            need = True; why.append(f'{c}: C3F realized no gain over C3 ({gain:+.3f} ms): freezing composes but does not pay; C3P isolates the pre-cast mechanism')
if sdr_fail:
    need = False
    why.append(f'E1 lane STOPPED: C3F SDR gate FAIL at {sdr_fail} (registered kill); C3P not run')
rec = {'run_C3P': need, 'reasons': why, 'e1_lane_stopped_sdr': sdr_fail,
       'rule': 'registered E1: C3P runs iff C3F is not an eligible pair at an operating point, leaves >= 50% of the C3 cast kernels, or realizes no gain (C3 - C3F <= 0) at an operating point; the lane stops if the C3F SDR gate fails'}
json.dump(rec, open(os.path.join(ev, 'e1_branch.json'), 'w'), indent=1)
print('[E1 branch]', rec)
PYEOF
if [ "$(python3 -c "import json;print(json.load(open('$EVID/e1_branch.json'))['run_C3P'])")" = "True" ]; then
    for chunk in 352800 4410; do
        for r in $(seq 1 "$RESTARTS"); do
            run_matched_q C3 C3P "$chunk" "m$r"
        done
        run_one_q C3P "$chunk" r1 --sdr
        trace_arm C3P "$chunk" trace || true
    done
fi

echo "=== [5] E9: held-out crossover — matched C3 vs H at 176400 and 264600 (+ A/A block), pod only ==="
if [ "${SESSION_POD:-0}" = "1" ] && [ "$SHAKEDOWN" != "1" ]; then
    for chunk in $HELD_OUT; do
        run_matched_q C3 C3 "$chunk" m1
        for r in $(seq 1 "$RESTARTS"); do
            run_matched_q C3 H "$chunk" "m$r"
        done
    done
    rm -f "$EVID/aa_stats.json" "$EVID/aux_aa_stats.json"
    run_aux aa_stats 120 -- python "$REPO_DIR/fullmodel/e_tools.py" aa_stats --evidence-dir "$EVID" --out "$EVID/aa_stats.json" || true
    rm -f "$EVID/e9_check.json"
    run_aux e9_check 120 -- python "$REPO_DIR/fullmodel/e_tools.py" e9_check --evidence-dir "$EVID" --registration "$REG" --out "$EVID/e9_check.json" || true
else
    for chunk in $HELD_OUT; do
        untested_pair C3 C3 "$chunk" m1 "E9 held-out length: never run outside the registered pod session (shakedown)"
        for r in $(seq 1 "$RESTARTS"); do
            untested_pair C3 H "$chunk" "m$r" "E9 held-out length: never run outside the registered pod session (shakedown)"
        done
    done
fi

echo "=== [6] E4a: norm-scale folding (C3N) vs the control E2 selects; SDR gate; traces ==="
rm -f "$EVID/e4a_decision.json"
python3 - "$EVID" <<'PYEOF'
import glob, json, os, statistics, sys
ev = sys.argv[1]
meds = []
for p in sorted(glob.glob(os.path.join(ev, 'matched_C3_C3M_c352800_m*.json'))):
    try:
        m = json.load(open(p))
    except Exception:
        continue
    if m.get('pair_eligible') and m.get('matched_delta_ms'):
        meds.append(m['matched_delta_ms']['median'])          # C3 - C3M, ms (+ = C3M faster)
gain = statistics.median(meds) if meds else None
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
qual = {}
for c in (352800, 4410):
    r1 = load(os.path.join(ev, f'full_C3M_c{c}_r1.json')) or {}
    gv = load(os.path.join(ev, 'traces', f'graph_verify_C3M_{c}.json')) or {}
    qual[c] = {'sdr': (r1.get('sdr_gate') or {}).get('status'), 'run_status': r1.get('status'),
               'validation': (r1.get('validation') or {}).get('status'), 'graph_verify': gv.get('verdict')}
quality_ok = all(q['sdr'] == 'PASS' and q['run_status'] == 'VALID' and q['validation'] == 'PASS' and q['graph_verify'] == 'FULLY_GRAPHED' for q in qual.values())
use_m = bool(gain is not None and gain >= 1.0 and quality_ok)
rec = {'C3_minus_C3M_8s_median_ms': gain, 'eligible_restarts': len(meds), 'C3M_qualification': qual, 'quality_ok': quality_ok,
       'rule': 'registered E4a (gate-09 f131): control = C3M and mode = max-autotune iff C3M beat C3 at 8 s by >= 1.0 ms (E2 prediction floor) AND C3M is quality-eligible at both operating points (r1 VALID, validation PASS, SDR PASS, FULLY_GRAPHED); else control = C3, mode = reduce-overhead',
       'control': 'C3M' if use_m else 'C3', 'mode': 'max-autotune' if use_m else 'reduce-overhead'}
json.dump(rec, open(os.path.join(ev, 'e4a_decision.json'), 'w'), indent=1)
print('[E4a decision]', rec)
PYEOF
E4A_CONTROL=$(python3 -c "import json;print(json.load(open('$EVID/e4a_decision.json'))['control'])")
export E4A_MODE=$(python3 -c "import json;print(json.load(open('$EVID/e4a_decision.json'))['mode'])")
for chunk in 352800 4410; do
    for r in $(seq 1 "$RESTARTS"); do
        run_matched_q "$E4A_CONTROL" C3N "$chunk" "m$r"
    done
done
for chunk in 352800 4410; do
    run_one_q C3N "$chunk" r1 --sdr
    T_COMPILE_SAVED="$T_COMPILE"; [ "$E4A_MODE" = "max-autotune" ] && T_COMPILE="$T_TRACE_AUTOTUNE"
    trace_arm C3N "$chunk" trace || true
    T_COMPILE="$T_COMPILE_SAVED"
done

echo "=== [7] E5: structural variants (depth x mask width), timing only, allowance ${E5_BUDGET_S}s ==="
E5_T0=$(date +%s)
e5_left() { echo $(( E5_BUDGET_S - ($(date +%s) - E5_T0) )); }
for v in $E5_VARIANTS; do
    if [ "$(e5_left)" -lt "$T_MIN" ] || [ "$(remaining)" -lt "$T_MIN" ]; then
        for arm in C3 H; do for chunk in 352800 4410; do for r in $(seq 1 "$E5_RESTARTS"); do
            terminal_json "$EVID/full_${arm}_c${chunk}_v_${v}_r${r}.json" "$arm" "$chunk" "v_${v}_r${r}" UNTESTED "E5 allowance exhausted (optional experiment)"
        done; done; done
        continue
    fi
    for chunk in 352800 4410; do
        run_one_q A "$chunk" "v_${v}_ref" --variant "$v" --save-reference --skip-timing
    done
    for arm in C3 H; do
        for chunk in 352800 4410; do
            for r in $(seq 1 "$E5_RESTARTS"); do
                run_one_q "$arm" "$chunk" "v_${v}_r${r}" --variant "$v"
            done
        done
        trace_arm "$arm" 4410 "v_${v}_trace" --variant "$v" || true
    done
done
rm -f "$EVID/e5_summary.json"
run_aux e5_summary 120 -- python "$REPO_DIR/fullmodel/e_tools.py" e5_summary --evidence-dir "$EVID" --out "$EVID/e5_summary.json" || true

echo "=== [8] E7: boundary-cost microbench with PDL evidence (attributes, PTX, SASS, prologue chains) + its node-level trace ==="
rm -f "$EVID/boundary_bench.json"
run_aux boundary_bench 900 -- python "$REPO_DIR/fullmodel/boundary_bench.py" --out "$EVID/boundary_bench.json" || true
rm -f "$EVID/traces/trace_boundary_bench.nsys-rep" "$EVID/traces/trace_boundary_bench"_*.csv \
      "$EVID/boundary_bench_traced.json" "$EVID/boundary_bench_trace_analysis.json"
run_aux boundary_bench_trace 900 -- nsys profile --cuda-graph-trace=node --force-overwrite=true \
    -o "$EVID/traces/trace_boundary_bench" \
    python "$REPO_DIR/fullmodel/boundary_bench.py" --out "$EVID/boundary_bench_traced.json" || true
run_aux boundary_bench_export 600 -- nsys stats --force-export=true --force-overwrite=true --format csv \
    --report cuda_gpu_trace --report cuda_gpu_kern_sum --report nvtx_gpu_proj_trace \
    --output "$EVID/traces/trace_boundary_bench" "$EVID/traces/trace_boundary_bench.nsys-rep" > /dev/null 2>&1 || true
run_aux boundary_bench_analysis 300 -- python "$REPO_DIR/fullmodel/boundary_bench.py" --analyze-trace "$EVID/traces/trace_boundary_bench" \
    --traced-json "$EVID/boundary_bench_traced.json" --out "$EVID/boundary_bench_trace_analysis.json" || true

echo "=== [9] collection + completeness verdict ==="
finalize_session session3
echo "SESSION3 DONE $(date -Is) elapsed=$(( $(date +%s) - T0 ))s remaining=$(remaining)s"
echo ">>> EXTRACT evidence/ OFF-POD BEFORE TERMINATE"
