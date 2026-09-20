# Shared session machinery (sourced by scripts/session1.sh and session2.sh).
# Fixes from gate 02: never export an EMPTY TORCH_LOGS (it breaks `import
# torch` on the pinned build); environment assignments go through `env` with
# an array (never expanded in command position); every launch is bounded by
# min(per-process cap, remaining usable time) and always leaves a terminal
# JSON; a hard session deadline with an extraction reserve applies to every
# stage (qualification and microbenchmarks included).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVID="${EVID:-$REPO_DIR/evidence}"
REF="$EVID/ref"
ITERS="${ITERS:-50}"
REPEATS="${REPEATS:-5}"
RESTARTS="${RESTARTS:-3}"
CLIPS_DIR="${CLIPS_DIR:-$HOME/work/sdr_clips}"
T_COMPILE="${T_COMPILE:-1500}"
T_OTHER="${T_OTHER:-900}"
T_MIN="${T_MIN:-240}"                            # below this, a cell is not started
SESSION_BUDGET_S="${SESSION_BUDGET_S:-10800}"
EXTRACT_RESERVE_S="${EXTRACT_RESERVE_S:-900}"
RUN="$REPO_DIR/fullmodel/run_full_arm.py"
MATCHED="$REPO_DIR/fullmodel/matched_run.py"
T0=$(date +%s)

remaining() { echo $(( SESSION_BUDGET_S - EXTRACT_RESERVE_S - ($(date +%s) - T0) )); }
cap_for()   { local want="$1" rem; rem=$(remaining); [ "$rem" -lt "$want" ] && echo "$rem" || echo "$want"; }

logs_for() { case "$1" in C1|C2|C3|C1A|C1D|H) echo "graph_breaks,recompiles,perf_hints,cudagraphs" ;; *) echo "" ;; esac; }
tmo_for()  { case "$1" in C1|C2|C3|C1A|C1D|H) echo "$T_COMPILE" ;; *) echo "$T_OTHER" ;; esac; }

terminal_json() {   # path arm chunk tag status reason
    python3 - "$@" <<'PYEOF'
import json, sys
path, arm, chunk, tag, status, reason = sys.argv[1:]
json.dump({'arm': arm, 'chunk': int(chunk), 'tag': tag, 'status': status,
           'error': reason, 'headline_eligible': False,
           'ineligibility_reasons': [reason]}, open(path, 'w'), indent=1)
PYEOF
}

# run_one ARM CHUNK TAG [extra args...]  — one fresh process, env via array
run_one() {
    local arm="$1" chunk="$2" tag="$3"; shift 3
    local path="$EVID/full_${arm}_c${chunk}_${tag}.json"
    local cap; cap=$(cap_for "$(tmo_for "$arm")")
    if [ "$cap" -lt "$T_MIN" ]; then
        echo "--- arm=$arm chunk=$chunk tag=$tag: DEADLINE (usable $cap s) -> UNTESTED ---"
        terminal_json "$path" "$arm" "$chunk" "$tag" UNTESTED "session deadline: not started"
        return
    fi
    local envv=() logs; logs="$(logs_for "$arm")"
    [ -n "$logs" ] && envv+=("TORCH_LOGS=$logs")
    echo "--- arm=$arm chunk=$chunk tag=$tag $(date -Is) cap=${cap}s remaining=$(remaining)s ---"
    env "${envv[@]}" timeout -k 30 "$cap" python "$RUN" --arm "$arm" --chunk "$chunk" \
        --tag "$tag" --iters "$ITERS" --repeats "$REPEATS" --clips-dir "$CLIPS_DIR" \
        --evidence-dir "$EVID" "$@" 2> "$EVID/torchlogs_${arm}_${chunk}_${tag}.log"
    local rc=$?
    if [ "$rc" = "124" ] || [ "$rc" = "137" ]; then
        terminal_json "$path" "$arm" "$chunk" "$tag" UNTESTED "timeout after ${cap}s"
        echo "    TIMEOUT -> UNTESTED recorded"
    elif [ "$rc" != "0" ]; then
        echo "    process exit $rc"
        [ -s "$path" ] || terminal_json "$path" "$arm" "$chunk" "$tag" INVALID "process exit $rc without JSON"
    fi
}

# run_matched ARM_A ARM_B CHUNK TAG
run_matched() {
    local a="$1" b="$2" chunk="$3" tag="$4"
    local path="$EVID/matched_${a}_${b}_c${chunk}_${tag}.json"
    local cap; cap=$(cap_for "$T_COMPILE")
    if [ "$cap" -lt "$T_MIN" ]; then
        echo "--- matched $a/$b $tag: DEADLINE -> UNTESTED ---"
        python3 -c "import json;json.dump({'pair':['$a','$b'],'chunk':$chunk,'tag':'$tag','status':'UNTESTED','error':'session deadline','pair_eligible':False},open('$path','w'),indent=1)"
        return
    fi
    echo "--- matched pair $a vs $b chunk=$chunk tag=$tag $(date -Is) cap=${cap}s ---"
    env TORCH_LOGS="graph_breaks,recompiles,cudagraphs" timeout -k 30 "$cap" python "$MATCHED" \
        --pair "$a" "$b" --chunk "$chunk" --tag "$tag" --iters "$ITERS" --repeats "$REPEATS" \
        --evidence-dir "$EVID" 2> "$EVID/torchlogs_matched_${a}_${b}_${chunk}_${tag}.log"
    local rc=$?
    [ "$rc" = "0" ] || echo "    matched exit $rc"
    [ -s "$path" ] || python3 -c "import json;json.dump({'pair':['$a','$b'],'chunk':$chunk,'tag':'$tag','status':'UNTESTED','error':'exit $rc','pair_eligible':False},open('$path','w'),indent=1)"
}

# run_traced ARM CHUNK TAG GRAPHMODE(node|graph) [extra...]  — one 5a block under nsys
run_traced() {
    local arm="$1" chunk="$2" tag="$3" gmode="$4"; shift 4
    local base="$EVID/traces/trace_${arm}_${chunk}_${tag}"
    local path="$EVID/full_${arm}_c${chunk}_${tag}.json"
    local cap; cap=$(cap_for "$(tmo_for "$arm")")
    mkdir -p "$EVID/traces"
    if [ "$cap" -lt "$T_MIN" ]; then
        echo "--- TRACE $arm/$chunk/$tag: DEADLINE -> UNTESTED ---"
        terminal_json "$path" "$arm" "$chunk" "$tag" UNTESTED "session deadline: not started"
        return 1
    fi
    local envv=("TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1") logs; logs="$(logs_for "$arm")"
    [ -n "$logs" ] && envv+=("TORCH_LOGS=$logs")
    if [ -n "$logs" ] && [ "$gmode" = "node" ]; then
        local dump="$EVID/traces/dump_${arm}_${chunk}_${tag}"; rm -rf "$dump"; mkdir -p "$dump"
        envv+=("TORCH_COMPILE_DEBUG=1" "TORCH_COMPILE_DEBUG_DIR=$dump" "TORCHINDUCTOR_FORCE_DISABLE_CACHES=1")
    fi
    echo "--- TRACE arm=$arm chunk=$chunk tag=$tag graph-trace=$gmode $(date -Is) cap=${cap}s ---"
    env "${envv[@]}" timeout -k 30 "$cap" nsys profile --capture-range=cudaProfilerApi \
        --capture-range-end=stop --cuda-graph-trace="$gmode" --force-overwrite=true \
        -o "$base" python "$RUN" --arm "$arm" --chunk "$chunk" --tag "$tag" \
        --iters "$ITERS" --repeats 1 --nvtx-pass --clips-dir "$CLIPS_DIR" --evidence-dir "$EVID" "$@" \
        2> "$EVID/torchlogs_${arm}_${chunk}_${tag}.log"
    local rc=$?
    if [ "$rc" != "0" ]; then
        echo "    trace run exit $rc"
        [ -s "$path" ] || terminal_json "$path" "$arm" "$chunk" "$tag" UNTESTED "trace run exit $rc"
        return 1
    fi
    [ -s "$base.nsys-rep" ] || { echo "    no .nsys-rep"; return 1; }
    if [ "$gmode" = "node" ]; then
        # stale outputs invalidated BEFORE the attempt; the export is bounded and
        # its exit code decides completeness (gate-04 f57)
        rm -f "${base}"_*.csv "${base}_exports.json" "${base}.sqlite"
        local ecap; ecap=$(cap_for 900)
        local erc=1
        if [ "$ecap" -ge 60 ]; then
            timeout -k 30 "$ecap" nsys stats --force-export=true --force-overwrite=true --format csv \
                --report cuda_gpu_trace --report cuda_gpu_kern_sum --report cuda_api_trace \
                --report nvtx_gpu_proj_trace --report nvtx_gpu_proj_sum \
                --output "$base" "$base.nsys-rep" > /dev/null 2>&1; erc=$?
            [ "$erc" = "0" ] || echo "    WARN: export exit $erc for $base"
        else
            echo "    DEADLINE: exports not attempted for $base"
        fi
        local missing=0
        for rep in cuda_gpu_trace cuda_gpu_kern_sum cuda_api_trace nvtx_gpu_proj_trace; do
            [ -s "${base}_${rep}.csv" ] || { echo "    MISSING export: ${base}_${rep}.csv"; missing=1; }
        done
        python3 -c "import json;json.dump({'complete': ($(( 1 - missing )) == 1) and ($erc == 0), 'export_exit': $erc, 'graph_trace_mode': '$gmode'}, open('${base}_exports.json','w'))"
    fi
    return 0
}

# finalize_session SESSION: collection and audit are supervised; a failure of
# either leaves an explicit INCOMPLETE session record (gate-04 f57)
finalize_session() {
    local sess="$1" verdict
    run_aux collect 600 -- python "$REPO_DIR/fullmodel/collect_full.py" --evidence-dir "$EVID" \
        || python3 -c "import json;json.dump({'session':'$sess','verdict':'INCOMPLETE','reason':'collection failed'},open('$EVID/session_status.json','w'))"
    run_aux completeness 300 -- python "$REPO_DIR/fullmodel/completeness.py" --evidence-dir "$EVID" --session "$sess"
    [ -s "$EVID/session_status.json" ] || python3 -c "import json;json.dump({'session':'$sess','verdict':'INCOMPLETE','reason':'completeness audit failed'},open('$EVID/session_status.json','w'))"
    verdict=$(python3 -c "import json;print(json.load(open('$EVID/session_status.json')).get('verdict','INCOMPLETE'))" 2>/dev/null || echo INCOMPLETE)
    echo "SESSION VERDICT ($sess): $verdict"
}

# bounded auxiliary command: run_aux NAME CAP -- cmd...   (supervisor messages
# go to STDERR so a redirected stdout stays clean; a terminal JSON records
# the outcome either way)
aux_json() {   # name status reason
    python3 -c "import json,sys;json.dump({'item':sys.argv[1],'status':sys.argv[2],'reason':sys.argv[3]},open('$EVID/aux_'+sys.argv[1]+'.json','w'))" "$1" "$2" "$3"
}
run_aux() {
    local name="$1" want="$2"; shift 2; [ "$1" = "--" ] && shift
    local cap; cap=$(cap_for "$want")
    if [ "$cap" -lt 60 ]; then
        echo "--- $name: DEADLINE -> skipped (UNTESTED) ---" >&2
        aux_json "$name" UNTESTED "session deadline"; return 1
    fi
    echo "--- $name $(date -Is) cap=${cap}s ---" >&2
    timeout -k 30 "$cap" "$@"; local rc=$?
    if [ "$rc" = "0" ]; then aux_json "$name" OK ""; else
        echo "    $name exit $rc" >&2
        aux_json "$name" $([ "$rc" = "124" ] && echo UNTESTED || echo FAILED) "exit $rc"
    fi
    return $rc
}

clocks_start() {   # file
    nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
        --format=csv,noheader -l 1 > "$1" 2>/dev/null &
    CLK_PID=$!
    trap 'kill $CLK_PID 2>/dev/null' EXIT
}
