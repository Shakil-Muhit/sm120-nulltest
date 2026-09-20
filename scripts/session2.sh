#!/bin/bash
# SESSION 2 (prereg_amendments.md §5): sweep, node-level traces, profiling
# perturbation, microbenchmarks, verification exports and post-processing.
# Same deadline/reserve/terminal-record rules as session 1 (session_lib.sh).
# Requires session-1 evidence/ (manifest + refs); on a fresh pod the
# preflight creates a session-2 manifest and re-creates the references.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/session_lib.sh"
TRACE_ARMS="${TRACE_ARMS:-A ANVTX C3 G H C1 B E F}"
SWEEP_ARMS="${SWEEP_ARMS:-A C3 G H}"
mkdir -p "$EVID" "$REF" "$EVID/traces"
cd "$REPO_DIR"
exec > >(tee -a "$EVID/session2_console.log") 2>&1
echo "SESSION2 START $(date -Is) budget=${SESSION_BUDGET_S}s reserve=${EXTRACT_RESERVE_S}s"
[ -z "${TMUX:-}" ] && echo "WARNING: not inside tmux"

echo "=== [0] preflight ==="
command -v nsys >/dev/null || { echo "FATAL: nsys missing"; exit 1; }
nsys profile --help 2>/dev/null | grep -q 'cuda-graph-trace' || { echo "FATAL: nsys lacks --cuda-graph-trace"; exit 1; }
nsys --version | tee "$EVID/nsys_version_session2.txt"
if [ ! -s "$EVID/session_manifest.json" ]; then
    python - "$EVID" session2 <<'PYEOF' || { echo "FATAL: provenance / session manifest failed"; exit 1; }
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
import provenance
print('manifest ok', provenance.load_manifest(sys.argv[1])['manifest_id'])
PYEOF
fi
clocks_start "$EVID/clocks_session2.csv"
for chunk in 352800 4410 22050 88200; do
    if [ ! -s "$REF/ref_A_c${chunk}_s2002.pt" ] || [ ! -s "$REF/sdr_A_c${chunk}_clip07.pt" ]; then
        run_one A "$chunk" ref --save-reference --sdr --skip-timing
    fi
done

echo "=== [1] chunk sweep: $SWEEP_ARMS x {22050, 88200} x $RESTARTS restarts (round-major) ==="
for r in $(seq 1 "$RESTARTS"); do
    for chunk in 22050 88200; do
        for arm in $SWEEP_ARMS; do
            extra=""; [ "$r" = "1" ] && extra="--sdr"
            run_one "$arm" "$chunk" "r$r" $extra
        done
    done
done

echo "=== [2] node-level traces (+ graph verification, timing views) ==="
for chunk in 352800 4410; do
    for arm in $TRACE_ARMS; do
        run_traced "$arm" "$chunk" trace node || continue
        base="$EVID/traces/trace_${arm}_${chunk}_trace"
        [ -s "${base}_cuda_gpu_trace.csv" ] || continue
        python "$REPO_DIR/fullmodel/graph_verify.py" --base "$base" --arm "$arm" --expected-passes "$ITERS" \
            --out "$EVID/traces/graph_verify_${arm}_${chunk}.json" || echo "    graph_verify failed for $arm/$chunk"
        python "$REPO_DIR/fullmodel/timing_views.py" --base "$base" --arm "$arm" \
            --out "$EVID/traces/timing_views_${arm}_${chunk}.json" || echo "    timing_views failed"
    done
done
# C2 replay verification at 4410 only (registered: the §6 comparator point)
if run_traced C2 4410 trace node; then
    base="$EVID/traces/trace_C2_4410_trace"
    python "$REPO_DIR/fullmodel/graph_verify.py" --base "$base" --arm C2 --expected-passes "$ITERS" \
        --out "$EVID/traces/graph_verify_C2_4410.json" || echo "    graph_verify failed for C2"
fi

echo "=== [3] profiling perturbation: A, C3, G — unprofiled vs graph-level vs node-level (one block each) ==="
for chunk in 352800 4410; do
    for arm in A C3 G; do
        run_one "$arm" "$chunk" pert_unprofiled --repeats 1
        run_traced "$arm" "$chunk" pert_graph graph
        run_traced "$arm" "$chunk" pert_node node
    done
done

echo "=== [4] attention-backend microbench (model backend from the C3 trace) ==="
KS="$EVID/traces/trace_C3_352800_trace_cuda_gpu_kern_sum.csv"
run_aux attn_backend_bench 900 -- python "$REPO_DIR/fullmodel/attn_backend_bench.py" \
    --out "$EVID/attn_backend_bench.json" $( [ -s "$KS" ] && echo "--model-kern-sum $KS" )

echo "=== [5] boundary-cost microbench + its node-level trace ==="
run_aux boundary_bench 900 -- python "$REPO_DIR/fullmodel/boundary_bench.py" --out "$EVID/boundary_bench.json"
run_aux boundary_bench_trace 900 -- nsys profile --cuda-graph-trace=node --force-overwrite=true \
    -o "$EVID/traces/trace_boundary_bench" \
    python "$REPO_DIR/fullmodel/boundary_bench.py" --out "$EVID/boundary_bench_traced.json" \
    && nsys stats --force-export=true --force-overwrite=true --format csv \
        --report cuda_gpu_trace --report cuda_gpu_kern_sum --report nvtx_gpu_proj_trace \
        --output "$EVID/traces/trace_boundary_bench" "$EVID/traces/trace_boundary_bench.nsys-rep" > /dev/null 2>&1 \
    && python "$REPO_DIR/fullmodel/boundary_bench.py" --analyze-trace "$EVID/traces/trace_boundary_bench" \
        --out "$EVID/boundary_bench_trace_analysis.json" \
    || echo "WARN: boundary trace/analysis incomplete"

echo "=== [6] per-kernel ideal costs (plan 1.4; measured L2 rate from [5]) ==="
BW_L2=$(python3 -c "import json;print(json.load(open('$EVID/boundary_bench.json')).get('l2_resident_bandwidth',{}).get('tbps',''))" 2>/dev/null)
L2B=$(python3 -c "import json;print(json.load(open('$EVID/boundary_bench.json')).get('l2_resident_bandwidth',{}).get('l2_cache_bytes',''))" 2>/dev/null)
for chunk in 352800 4410; do
    for arm in $TRACE_ARMS C2; do
        base="$EVID/traces/trace_${arm}_${chunk}_trace"
        [ -s "${base}_cuda_gpu_kern_sum.csv" ] || continue
        dump="$EVID/traces/dump_${arm}_${chunk}_trace"; [ -d "$dump" ] || dump=""
        python "$REPO_DIR/fullmodel/ideal_costs.py" --base "$base" --arm "$arm" --chunk "$chunk" \
            ${dump:+--dump-dir "$dump"} --passes "$ITERS" \
            ${BW_L2:+--bw-l2-tbps "$BW_L2"} ${L2B:+--l2-bytes "$L2B"} \
            --out "$EVID/traces/ideal_costs_${arm}_${chunk}.json" || echo "    ideal_costs failed for $arm/$chunk"
    done
done

echo "=== [7] post-processing: perturbation ratios, boundary perturbation, source map ==="
python "$REPO_DIR/fullmodel/perturbation.py" --evidence-dir "$EVID" --iters "$ITERS" \
    --out "$EVID/perturbation_and_source_map.json" || echo "WARN: post-processing failed"

echo "=== [8] collection ==="
python "$REPO_DIR/fullmodel/collect_full.py" --evidence-dir "$EVID" || true
echo "SESSION2 DONE $(date -Is) elapsed=$(( $(date +%s) - T0 ))s remaining=$(remaining)s"
echo ">>> EXTRACT evidence/ OFF-POD BEFORE TERMINATE"
