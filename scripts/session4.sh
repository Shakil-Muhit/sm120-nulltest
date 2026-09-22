#!/bin/bash
# SESSION 4 (docs/phase4/PLAN.md — Phase-4 follow-up, prospective plan). Order: P0 A/A → PA C3–H ranking
# at 1/4/6/8 s → PB C1P runner discrepancy → PC1–PC4 configuration interactions → PQ quality/trace checks
# for the new arms → collection and audit. Same deadline / reserve / terminal-record rules as sessions 1–3.
# SHAKEDOWN=1 (engineering host): everything runs (no held-out rule applies in Phase 4); the audit grades it engineering.
set -u
export SESSION_BUDGET_S="${SESSION_BUDGET_S:-16200}" EXTRACT_RESERVE_S="${EXTRACT_RESERVE_S:-900}"   # registered defaults (docs/phase4/PLAN.md; gate-11 f160)
export PHASE4=1                                                                                     # the Phase-3 held-out rule no longer applies (gate-11 f159)
. "$(dirname "${BASH_SOURCE[0]}")/session_lib.sh"
SHAKEDOWN="${SHAKEDOWN:-0}"; [ "${SESSION_POD:-0}" = "1" ] || SHAKEDOWN=1
PA_LENGTHS="${PA_LENGTHS:-44100 176400 264600 352800}"
AA_LENGTHS="${AA_LENGTHS:-44100 176400 264600 352800 4410}"
PC_OPS="${PC_OPS:-4410 352800}"
T_TRACE_AUTOTUNE="${T_TRACE_AUTOTUNE:-3000}"
export CLOCK_LMS="${CLOCK_LMS:-200}"
export REQUIRED_RESTARTS=3
mkdir -p "$EVID" "$REF" "$EVID/traces"
cd "$REPO_DIR"
exec > >(tee -a "$EVID/session4_console.log") 2>&1
echo "SESSION4 START $(date -Is) iters=$ITERS repeats=$REPEATS restarts=$RESTARTS budget=${SESSION_BUDGET_S}s reserve=${EXTRACT_RESERVE_S}s shakedown=$SHAKEDOWN pod=${SESSION_POD:-0}"
[ -z "${TMUX:-}" ] && echo "WARNING: not inside tmux"
python3 -c "import json;json.dump({'shakedown': '$SHAKEDOWN'=='1', 'session_pod': '${SESSION_POD:-0}'=='1', 'pair_restarts': $RESTARTS, 'pa_lengths': '$PA_LENGTHS'.split(), 'aa_lengths': '$AA_LENGTHS'.split(), 'pc_ops': '$PC_OPS'.split(), 'plan': 'docs/phase4/PLAN.md', 'plan_sha256': __import__('hashlib').sha256(open('docs/phase4/PLAN.md','rb').read()).hexdigest(), 'thresholds_ms': {'352800': 0.30, '264600': 0.30, '176400': 0.30, '44100': 0.10, '4410': 0.10}}, open('$EVID/session4_config.json','w'), indent=1)"

echo "=== [0] preflight ==="
command -v nsys >/dev/null || { echo "FATAL: nsys missing"; exit 1; }
nsys --version | tee "$EVID/nsys_version_session4.txt"
python - <<'PYEOF' || { echo "FATAL: preflight imports failed"; exit 1; }
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'fullmodel'))
import torch, triton, fullarms, e_tools, stats_v2
assert all(a in fullarms.ARMS for a in ('C3I', 'C3MG', 'HG')) and 44100 in fullarms.CHUNKS
print('torch', torch.__version__, 'triton', triton.__version__, 'chunks', fullarms.CHUNKS)
PYEOF
if [ ! -s "$EVID/session_manifest.json" ]; then
    python - "$EVID" session4 <<'PYEOF' || { echo "FATAL: provenance / session manifest failed"; exit 1; }
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
clocks_start "$EVID/clocks_session4.csv"
echo "=== [0a] eager-A references: validation at every Phase-4 length; SDR clip outputs at 352800 and 4410 ==="
for chunk in 352800 4410; do run_one A "$chunk" ref --save-reference --sdr --skip-timing; done
for chunk in 44100 176400 264600; do run_one A "$chunk" ref --save-reference --skip-timing; done
PREREQ_OK=1; PREREQ_REASON=""
for f in ref_A_c352800_s1001.pt ref_A_c4410_s1001.pt ref_A_c44100_s1001.pt ref_A_c176400_s1001.pt ref_A_c264600_s1001.pt sdr_A_c352800_clip07.pt sdr_A_c4410_clip07.pt manifest_id.txt; do
    [ -s "$REF/$f" ] || { echo "reference $f missing -> validated measurements UNTESTED (session INCOMPLETE)"; PREREQ_OK=0; PREREQ_REASON="references unavailable"; }
done
echo "=== [0b] planted-fault validator checks (both must FAIL) ==="
run_one A 4410 planted_tight --planted-fault --skip-timing
run_one A 4410 planted_loose --planted-fault --planted-const 0.05 --force-tier loose --skip-timing
python3 - "$EVID/full_A_c4410_planted_tight.json" 0.001 "$EVID/full_A_c4410_planted_loose.json" 0.05 <<'PYEOF' || { echo "a validator did not catch its planted fault"; PREREQ_OK=0; PREREQ_REASON="planted fault not caught"; }
import json, sys
ok_all = True
for path, const in ((sys.argv[1], float(sys.argv[2])), (sys.argv[3], float(sys.argv[4]))):
    r = json.load(open(path)); v = r.get('validation') or {}; passes = v.get('passes') or []
    diffs = [p.get('max_abs_diff') for p in passes]
    ok = (v.get('status') == 'FAIL' and len(passes) == 2 and all(not p['ok'] for p in passes) and all(d is not None and abs(d - const) < 1e-4 * const + 1e-6 for d in diffs))
    print(f"planted {const}: {'CAUGHT' if ok else 'NOT CAUGHT'}"); ok_all &= ok
sys.exit(0 if ok_all else 1)
PYEOF
echo "=== [0c] C3 operator qualification per frame count (11 51 201 801 401 601 101) ==="
C3_QUAL_OK=1
run_aux c3_operator_qualification 600 -- python - "$EVID/c3_operator_qualification.json" 11 51 201 801 401 601 101 <<'PYEOF' || { echo "C3 operator qualification FAILED -> C3-derived work UNTESTED"; C3_QUAL_OK=0; }
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'fullmodel'))
import fullarms
rec = fullarms.verify_c3_operator([int(x) for x in sys.argv[2:]])
json.dump(rec, open(sys.argv[1], 'w'), indent=1)
assert all(v.get('status') == 'PASS' for v in rec.values()), rec
print('C3 operator qualification PASS for frames', sorted(rec))
PYEOF
needs_c3() { case "$1" in C3|H|C3F|C3P|C3M|C3N|C3G|C3I|C3MG|HG) return 0;; *) return 1;; esac; }
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
        terminal_json "$EVID/full_$1_c$2_$3.json" "$1" "$2" "$3" UNTESTED "${PREREQ_REASON:-C3 operator qualification failed}"; return 1
    fi
    run_traced "$@"
}
trace_arm() {
    local arm="$1" chunk="$2" tag="$3"; shift 3
    run_traced_q "$arm" "$chunk" "$tag" node "$@" || return 1
    local base="$EVID/traces/trace_${arm}_${chunk}_${tag}"
    [ -s "${base}_cuda_gpu_trace.csv" ] || return 1
    local sfx=""; [ "$tag" != "trace" ] && sfx="_${tag}"
    rm -f "$EVID/traces/graph_verify_${arm}_${chunk}${sfx}.json" "$EVID/traces/timing_views_${arm}_${chunk}${sfx}.json" "$EVID/traces/cast_count_${arm}_${chunk}${sfx}.json"
    run_aux "graph_verify_${arm}_${chunk}${sfx}" 300 -- python "$REPO_DIR/fullmodel/graph_verify.py" --base "$base" --arm "$arm" --expected-passes "$ITERS" --out "$EVID/traces/graph_verify_${arm}_${chunk}${sfx}.json" || true
    run_aux "timing_views_${arm}_${chunk}${sfx}" 300 -- python "$REPO_DIR/fullmodel/timing_views.py" --base "$base" --arm "$arm" --out "$EVID/traces/timing_views_${arm}_${chunk}${sfx}.json" || true
    local dump="$EVID/traces/dump_${arm}_${chunk}_${tag}"; [ -d "$dump" ] || dump=""
    rm -f "$EVID/traces/ideal_costs_${arm}_${chunk}${sfx}.json" "$EVID/traces/ideal_costs_${arm}_${chunk}${sfx}.csv" "$EVID/traces/template_count_${arm}_${chunk}${sfx}.json"
    run_aux "ideal_costs_${arm}_${chunk}${sfx}" 600 -- python "$REPO_DIR/fullmodel/ideal_costs.py" --base "$base" --arm "$arm" --chunk "$chunk" ${dump:+--dump-dir "$dump"} --passes "$ITERS" --out "$EVID/traces/ideal_costs_${arm}_${chunk}${sfx}.json" || true
    run_aux "cast_count_${arm}_${chunk}${sfx}" 120 -- python "$REPO_DIR/fullmodel/e_tools.py" cast_count --ideal-costs "$EVID/traces/ideal_costs_${arm}_${chunk}${sfx}.json" --dump-dir "${dump:-/nonexistent}" --out "$EVID/traces/cast_count_${arm}_${chunk}${sfx}.json" || true
    run_aux "template_count_${arm}_${chunk}${sfx}" 120 -- python "$REPO_DIR/fullmodel/e_tools.py" template_count --ideal-costs "$EVID/traces/ideal_costs_${arm}_${chunk}${sfx}.json" --out "$EVID/traces/template_count_${arm}_${chunk}${sfx}.json" || true
}
# run_matched_profiled A B CHUNK TAG: one repeat of the matched process under nsys (node trace), exports incl. host NVTX ranges, aligned analysis
run_matched_profiled() {
    local a="$1" b="$2" chunk="$3" tag="$4"
    local base="$EVID/traces/trace_matched_${a}_${b}_${chunk}_${tag}"
    local path="$EVID/matched_${a}_${b}_c${chunk}_${tag}.json"
    local cap; cap=$(cap_for "$T_COMPILE")
    rm -f "$path" "$base.nsys-rep" "${base}"_*.csv "${base}_exports.json" "$EVID/traces/matched_trace_${a}_${b}_${chunk}_${tag}.json"
    if [ "$cap" -lt "$T_MIN" ]; then
        python3 -c "import json;json.dump({'pair':['$a','$b'],'chunk':$chunk,'tag':'$tag','status':'UNTESTED','error':'session deadline','pair_eligible':False},open('$path','w'),indent=1)"; return 1
    fi
    echo "--- PROFILED matched pair $a vs $b chunk=$chunk tag=$tag $(date -Is) cap=${cap}s ---"
    env TORCH_LOGS="graph_breaks,recompiles,cudagraphs" timeout -k 30 "$cap" nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        --cuda-graph-trace=node --force-overwrite=true -o "$base" python "$MATCHED" --pair "$a" "$b" --chunk "$chunk" --tag "$tag" \
        --iters "$ITERS" --repeats "$REPEATS" --profile-repeat 0 --evidence-dir "$EVID" 2> "$EVID/torchlogs_matched_${a}_${b}_${chunk}_${tag}.log"
    local rc=$?; [ "$rc" = "0" ] || echo "    profiled matched exit $rc"
    [ -s "$path" ] || python3 -c "import json;json.dump({'pair':['$a','$b'],'chunk':$chunk,'tag':'$tag','status':'UNTESTED','error':'exit $rc','pair_eligible':False},open('$path','w'),indent=1)"
    [ -s "$base.nsys-rep" ] || { echo "    no .nsys-rep"; return 1; }
    local ecap; ecap=$(cap_for 900); local erc=1
    if [ "$ecap" -ge 60 ]; then
        timeout -k 30 "$ecap" nsys stats --force-export=true --force-overwrite=true --format csv \
            --report cuda_gpu_trace --report cuda_api_trace --report nvtx_pushpop_trace --report nvtx_gpu_proj_trace \
            --output "$base" "$base.nsys-rep" > /dev/null 2>&1; erc=$?
    fi
    python3 -c "import json,os;json.dump({'complete': $erc == 0 and all(os.path.exists('${base}_'+r+'.csv') for r in ('cuda_gpu_trace','cuda_api_trace','nvtx_pushpop_trace')), 'export_exit': $erc, 'graph_trace_mode': 'node'}, open('${base}_exports.json','w'))"
    run_aux "matched_trace_${a}_${b}_${chunk}_${tag}" 300 -- python "$REPO_DIR/fullmodel/e_tools.py" matched_trace --base "$base" --record "$path" --out "$EVID/traces/matched_trace_${a}_${b}_${chunk}_${tag}.json" || true
}

echo "=== [1] P0: A/A control (C3 vs C3) at $AA_LENGTHS ==="
for chunk in $AA_LENGTHS; do run_matched_q C3 C3 "$chunk" m1; done

echo "=== [2] PA: matched C3 vs H at $PA_LENGTHS, $RESTARTS restarts each ==="
for r in $(seq 1 "$RESTARTS"); do for chunk in $PA_LENGTHS; do run_matched_q C3 H "$chunk" "m$r"; done; done

echo "=== [3] PB: C1P runner discrepancy @352800 ==="
for r in $(seq 1 "$RESTARTS"); do extra=""; [ "$r" = "1" ] && extra="--sdr"; run_one_q C1P 352800 "r$r" $extra; done   # standalone UNPROFILED C1P (never run before); SDR in r1
for r in $(seq 1 "$RESTARTS"); do run_matched_q C1 C1P 352800 "m$r"; done         # matched, control first (Phase-3 order)
for r in 1 2; do run_matched_q C1P C1 352800 "m$r"; done                            # matched, build order swapped
run_matched_profiled C1 C1P 352800 p1                                                # the matched context itself under the profiler (repeat 0)
trace_arm C1P 352800 trace || true                                                   # standalone profiled C1P (same session, for the aligned comparison)
trace_arm C1 352800 trace || true

echo "=== [4a] PQ quality gates FIRST (registered stop rule): r1 with the SDR gate for every arm used in PC pairs, both operating points ==="
declare -A ARM_OK
for chunk in $PC_OPS; do
    for arm in C3 C3I C3G C3M C3MG H HG; do
        run_one_q "$arm" "$chunk" r1 --sdr
        st=$(python3 - "$EVID/full_${arm}_c${chunk}_r1.json" <<'PYEOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
    print('OK' if r.get('status') == 'VALID' and (r.get('validation') or {}).get('status') == 'PASS' and (r.get('sdr_gate') or {}).get('status') == 'PASS' else 'STOP')
except Exception:
    print('STOP')
PYEOF
)
        ARM_OK["$arm@$chunk"]="$st"
        [ "$st" = "OK" ] || echo "ARM STOP: $arm@$chunk ($st) — its matched pairs at this operating point are recorded UNTESTED"
    done
done
pair_or_stop() {   # A B CHUNK TAG: run the pair only when both arms passed their gates at this operating point
    if [ "${ARM_OK[$1@$3]:-STOP}" = "OK" ] && [ "${ARM_OK[$2@$3]:-STOP}" = "OK" ]; then run_matched_q "$@"; else
        python3 -c "import json;json.dump({'pair':['$1','$2'],'chunk':$3,'tag':'$4','status':'UNTESTED','error':'registered stop rule: an arm failed its correctness gate at this operating point','pair_eligible':False},open('$EVID/matched_$1_$2_c$3_$4.json','w'),indent=1)"
    fi
}

echo "=== [4b] PC1–PC4: configuration interactions at $PC_OPS ==="
for chunk in $PC_OPS; do
    for r in $(seq 1 "$RESTARTS"); do
        pair_or_stop C3 C3I "$chunk" "m$r"
        pair_or_stop C3I C3G "$chunk" "m$r"
        pair_or_stop C3M C3MG "$chunk" "m$r"
        pair_or_stop H HG "$chunk" "m$r"
    done
done

echo "=== [5] PQ: node traces for the new arms and their controls ==="
for arm in C3I HG C3G C3; do for chunk in 4410 352800; do trace_arm "$arm" "$chunk" trace || true; done; done
T_COMPILE_SAVED="$T_COMPILE"; T_COMPILE="$T_TRACE_AUTOTUNE"
trace_arm C3MG 4410 trace || true                    # mandatory (deployment evidence at 100 ms)
trace_arm C3MG 352800 trace || true                  # optional under the deadline (recompiles with caches off, ~25 min)
T_COMPILE="$T_COMPILE_SAVED"

echo "=== [6] statistics, collection + completeness verdict ==="
rm -f "$EVID/aa_stats.json" "$EVID/stats_v2.json"
run_aux aa_stats 120 -- python "$REPO_DIR/fullmodel/e_tools.py" aa_stats --evidence-dir "$EVID" --out "$EVID/aa_stats.json" || true
run_aux stats_v2 300 -- python "$REPO_DIR/fullmodel/stats_v2.py" --evidence-dir "$EVID" --out "$EVID/stats_v2.json" \
    --pairs C3:H:44100 C3:H:176400 C3:H:264600 C3:H:352800 C1:C1P:352800 C1P:C1:352800 C3:C3I:4410 C3:C3I:352800 C3I:C3G:4410 C3I:C3G:352800 \
            C3M:C3MG:4410 C3M:C3MG:352800 H:HG:4410 H:HG:352800 C3:C3:44100 C3:C3:176400 C3:C3:264600 C3:C3:352800 C3:C3:4410 || true
finalize_session session4
echo "SESSION4 DONE $(date -Is) elapsed=$(( $(date +%s) - T0 ))s remaining=$(remaining)s"
