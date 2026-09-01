#!/bin/bash
# nsys traces + kernel-count extraction (prereg structural evidence + GPU-busy).
#
# PREREG anchors:
#   - "nsys timeline of (best generic arm) and (G): kernel count + gap
#     structure. These two traces are the structural evidence either way."
#   - "GPU-busy per arm: extract from the nsys kernel summary of that arm's
#     trace where a trace exists; otherwise from a dedicated short nsys pass."
#
# Implementation choice (documented in README): EVERY (arm x regime) gets its
# own nsys pass with an identical invocation, so GPU-busy always comes from
# that arm's own trace and the "otherwise" branch is never exercised. The
# best-generic and G traces are the two structural timelines; the rest exist
# for the per-arm busy numbers.
#
# Each pass is a full arms/run_arm.py run with --tag trace. The cudaProfilerApi
# capture range covers ONLY the prereg-5a latency block (common.timed_run), so
#     busy_ms_per_pass = total kernel time / iters
#     kernels_per_pass = total instances   / iters
# and the decision rule's R pairs THIS run's latency median with THIS trace's
# busy time — "busy from nsys kernel sum, same run".
#
# Usage:  bash analysis/traces.sh [BEST_GENERIC_ARM]
#         (no argument: asks analysis/collect.py, which needs the r1-r3 arm
#          runs to exist already — run_all.sh runs arms first)
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVID="$REPO_DIR/evidence"
ITERS="${ITERS:-50}"
ARMS="A B C1 C2 D E F G"

BEST="${1:-}"
if [ -z "$BEST" ]; then
    BEST="$(python "$REPO_DIR/analysis/collect.py" --best-generic)" || {
        echo "FATAL: best generic arm not determinable (arm runs missing?)."
        echo "       Run the arms first, or pass it explicitly: traces.sh C1"
        exit 1
    }
fi
echo "[traces] structural timelines: best generic arm = $BEST, and G"

# CRITICAL correctness gate: arms B/E/G replay manual CUDA graphs and C1/C2
# replay inductor cudagraphs. Modern nsys defaults to whole-graph tracing, in
# which replayed kernels DO NOT appear in the kernel summary — GPU-busy would
# collapse to ~0 for those arms and the decision-rule R would inflate toward
# the thesis-favoring verdict. --cuda-graph-trace=node traces kernels inside
# graph replays individually. Refuse to produce busy numbers without it.
if ! nsys profile --help 2>/dev/null | grep -q 'cuda-graph-trace'; then
    echo "FATAL: this nsys does not support --cuda-graph-trace=node; busy"
    echo "       numbers for graph-replaying arms would be silently wrong."
    echo "       Install a newer nsight-systems (see setup.sh [2/4])."
    exit 1
fi
nsys --version | tee "$EVID/nsys_version.txt"

for chunk in 352800 4410; do
    for arm in $ARMS; do
        base="$EVID/trace_${arm}_${chunk}"
        echo "[traces] nsys pass: arm=$arm chunk=$chunk -> $base.nsys-rep"
        # never leave a stale busy file paired with a newer trace run
        rm -f "$EVID/busy_${arm}_${chunk}.json"

        # TORCH_LOGS on compile arms (prereg: captured on every compile arm).
        # fd 9 carries the pass's stderr: a torchlogs FILE for C1/C2, else a
        # DUP of the current stderr — never a /dev/stderr re-open, which
        # would O_TRUNC the log when stderr is a regular file (nohup runs)
        logs_env=""
        case "$arm" in C1|C2) logs_env="recompiles,perf_hints" ;; esac
        if [ -n "$logs_env" ]; then
            exec 9> "$EVID/torchlogs_${arm}_${chunk}_trace.log"
        else
            exec 9>&2
        fi

        TORCH_LOGS="$logs_env" nsys profile \
            --capture-range=cudaProfilerApi --capture-range-end=stop \
            --cuda-graph-trace=node \
            --force-overwrite=true -o "$base" \
            python "$REPO_DIR/arms/run_arm.py" \
                --arm "$arm" --chunk "$chunk" --iters "$ITERS" --tag trace \
            2>&9 || { exec 9>&-; echo "[traces] WARN: nsys pass failed for $arm/$chunk — skipping"; continue; }
        exec 9>&-

        # kernel summary CSV (new report name first, legacy alias fallback)
        nsys stats --force-export=true --force-overwrite=true \
                -r cuda_gpu_kern_sum --format csv \
                --output "$base" "$base.nsys-rep" >/dev/null 2>&1 \
            || nsys stats --force-export=true --force-overwrite=true \
                -r gpukernsum --format csv \
                --output "$base" "$base.nsys-rep" >/dev/null \
            || { echo "[traces] WARN: nsys stats failed for $arm/$chunk"; continue; }

        python3 - "$arm" "$chunk" "$EVID" <<'PYEOF' \
            || echo "[traces] WARN: busy extraction failed for $arm/$chunk — no busy_${arm}_${chunk}.json"
import csv, glob, json, os, sys
arm, chunk, evid = sys.argv[1], int(sys.argv[2]), sys.argv[3]
base = os.path.join(evid, f'trace_{arm}_{chunk}')
paths = sorted(glob.glob(base + '*kern*sum*.csv'))
if not paths:
    sys.exit(f'no kernel-summary csv found for {base}')
total_ns, instances = 0, 0
with open(paths[-1], newline='') as f:
    rows = list(csv.reader(f))
head = rows[0]
def col(*names):
    # exact header match first (any candidate), else guarded substring
    for name in names:
        for i, h in enumerate(head):
            if h.strip().strip('"') == name:
                return i
    key = names[0].split(' ')[0]
    return next(i for i, h in enumerate(head)
                if key in h and 'Range' not in h and 'Proj' not in h
                and 'NVTX' not in h)
t_col = col('Total Time (ns)')
i_col = col('Instances', 'Kern Inst')
for row in rows[1:]:
    if len(row) <= max(t_col, i_col) or not row[i_col].strip():
        continue
    total_ns += int(float(row[t_col].replace(',', '')))
    instances += int(float(row[i_col].replace(',', '')))
run_json = os.path.join(evid, f'arm_{arm}_c{chunk}_trace.json')
with open(run_json) as f:
    run = json.load(f)
if run.get('status') != 'VALID':
    sys.exit(f'trace run for {arm}/{chunk} is {run.get("status")} — refusing '
             f'to write busy numbers from an invalid run')
# divide by the number of passes ACTUALLY completed inside the capture range
# (= the latency block; a partial run must not skew busy-per-pass)
iters = len(run.get('latency_ms_per_iter') or [])
busy_ms = total_ns / 1e6 / iters if iters else None
out = {
    'arm': arm, 'chunk': chunk,
    'busy_ms_per_pass': busy_ms,
    'kernels_per_pass': instances / iters if iters else None,
    'total_kernel_time_ms': total_ns / 1e6,
    'total_instances': instances,
    'iters': iters,
    'busy_source': 'own trace (capture range = latency block only)',
    'source_trace': os.path.basename(paths[-1]),
    'trace_latency_median_ms': run.get('latency_median_ms'),
    'wall_over_busy_same_run': (run['latency_median_ms'] / busy_ms
                                if busy_ms and run.get('latency_median_ms')
                                else None),
}
# GRAPH-TRACE-FAILURE GUARD (fix-pass 2026-08-18): arms whose timed region
# replays a CUDA graph (manual capture: B/E/G; inductor cudagraph trees under
# reduce-overhead / max-autotune in torch 2.8: C1/C2) must show their kernels
# individually. A graph arm reporting ~1 kernel/pass means
# --cuda-graph-trace=node did not take effect; such busy numbers are wrong in
# the thesis-favoring direction and must never be recorded. Floor = 100
# (session-owner spec). Note C2's P2-predicted ~120 kernels/pass sits only
# ~20% above the floor — a legitimately better-than-predicted Inductor fusion
# would trip this guard, which fails toward human review, the safe direction.
GRAPH_ARMS = ('B', 'E', 'G', 'C1', 'C2')
KPP_FLOOR = 100
kpp = out['kernels_per_pass']
if arm in GRAPH_ARMS and kpp is not None and kpp < KPP_FLOOR:
    sys.exit(f'GRAPH-TRACE GUARD: {arm}/{chunk} shows {kpp:.1f} kernels/pass '
             f'(< {KPP_FLOOR}) on a graph-replay arm — INVALID; refusing to '
             f'write busy numbers. Verify --cuda-graph-trace=node took effect '
             f'(docs/profiling_crib.md section 4.1).')
dest = os.path.join(evid, f'busy_{arm}_{chunk}.json')
with open(dest, 'w') as f:
    json.dump(out, f, indent=1)
print(f"[traces] {arm}/{chunk}: busy {busy_ms and round(busy_ms, 4)} ms/pass, "
      f"{out['kernels_per_pass'] and round(out['kernels_per_pass'], 1)} kernels/pass "
      f"-> {dest}")
PYEOF
    done
done

echo ""
echo "[traces] done. Structural timelines to inspect for gap structure and the"
echo "[traces] ~60-separate-GEMMs criterion (prereg structural criterion):"
echo "[traces]     ${EVID}/trace_${BEST}_<chunk>.nsys-rep   (best generic arm)"
echo "[traces]     ${EVID}/trace_G_<chunk>.nsys-rep         (arm G)"

