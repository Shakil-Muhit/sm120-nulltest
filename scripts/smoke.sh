#!/bin/bash
# SHAKEDOWN, NOT THE SESSION. Engineering smoke run (README: "Pod smoke test").
#
# Purpose: prove, in ~15-30 min of pod time, the CUDA-only behavior that no
# off-GPU verification can reach — BEFORE committing to the full pre-registered
# session:
#   [S1] frontend.py --sanity on the GPU (bitwise vs the pinned public repo)
#   [S2] arms A, B, G at the 4410 regime, ONE run each, default tag 'adhoc'
#        (adhoc runs are excluded from headline statistics by design):
#        A = baseline sanity; B = capture-fallback protocol (P4);
#        G = decision arm: fused Triton kernel + multi-stream + capture,
#        validation must PASS (first real launch of rms_fused on SM120)
#   [S3] one nsys pass of G@4410 with --cuda-graph-trace=node: the kernel
#        summary MUST show kernels inside the capture range (graph replay
#        visible) — the decision-rule R depends on exactly this
#   [S4] p8_full_trace.sh once (model load, NVTX hook boundaries, extraction)
#
# HYGIENE (pre-registered-session rules):
#   - outputs land in evidence_smoke/, NEVER in evidence/; smoke numbers are
#     engineering data and are never cited or extracted as session evidence
#   - the real session runs afterwards with a clean evidence/ and one
#     uninterrupted run_all.sh (fresh pod is equivalent — pods are ephemeral)
#   - mechanical fixes prompted by the smoke are committed BEFORE the session;
#     changes to arms/thresholds/predictions would be prereg amendments and
#     must be documented explicitly, never silently applied
#
# Usage: bash setup.sh   (first, once)   then   bash scripts/smoke.sh
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE="$REPO_DIR/evidence_smoke"
MSST_DIR="${MSST_DIR:-$HOME/work/msst}"
mkdir -p "$SMOKE"
cd "$REPO_DIR"

PASS=0; FAIL=0; VERDICTS=""
verdict() {  # $1 phase  $2 0/1 (0=pass)
    if [ "$2" -eq 0 ]; then PASS=$((PASS+1)); VERDICTS="$VERDICTS
  $1 : PASS"
    else FAIL=$((FAIL+1)); VERDICTS="$VERDICTS
  $1 : FAIL"
    fi
}

echo "================ SMOKE (shakedown, not the session) $(date -Is) ================"
command -v nsys >/dev/null || { echo "FATAL: nsys missing — run setup.sh first"; exit 1; }

echo "--- [S1] GPU bitwise sanity vs pinned public repo ---"
PYTHONPATH="$MSST_DIR" python frontend.py --sanity --chunk 4410 \
    2>&1 | tee "$SMOKE/s1_sanity.log" | tail -2
grep -q "bitwise exact" "$SMOKE/s1_sanity.log"; verdict "S1 sanity-bitwise" $?

echo "--- [S2] arms A, B, G @ 4410 (single runs, tag=adhoc) ---"
for arm in A B G; do
    python arms/run_arm.py --arm "$arm" --chunk 4410 --out-dir "$SMOKE" \
        > "$SMOKE/s2_arm_${arm}.log" 2>&1
    tail -2 "$SMOKE/s2_arm_${arm}.log"
    python3 - "$SMOKE/arm_${arm}_c4410_adhoc.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
ok = (r['status'] == 'VALID'
      and (r.get('validation') or {}).get('status') == 'PASS'
      and len(r.get('latency_ms_per_iter') or []) >= 50)
print(f"  {r['arm']}: status={r['status']} validation="
      f"{(r.get('validation') or {}).get('status')} "
      f"median={r.get('latency_median_ms')} scope={r.get('capture_scope')}")
sys.exit(0 if ok else 1)
PYEOF
    verdict "S2 arm-$arm" $?
done

echo "--- [S3] nsys graph-node visibility on arm G @ 4410 ---"
if ! nsys profile --help 2>/dev/null | grep -q 'cuda-graph-trace'; then
    echo "  FATAL: nsys lacks --cuda-graph-trace"; verdict "S3 graph-node-trace" 1
else
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        --cuda-graph-trace=node --force-overwrite=true -o "$SMOKE/trace_G_4410" \
        python arms/run_arm.py --arm G --chunk 4410 --tag smoketrace \
            --out-dir "$SMOKE" > "$SMOKE/s3_nsys.log" 2>&1
    nsys stats --force-export=true --force-overwrite=true \
            -r cuda_gpu_kern_sum --format csv \
            --output "$SMOKE/trace_G_4410" "$SMOKE/trace_G_4410.nsys-rep" \
            >/dev/null 2>&1 \
        || nsys stats --force-export=true --force-overwrite=true \
            -r gpukernsum --format csv \
            --output "$SMOKE/trace_G_4410" "$SMOKE/trace_G_4410.nsys-rep" >/dev/null
    python3 - "$SMOKE" <<'PYEOF'
import csv, glob, json, sys
smoke = sys.argv[1]
paths = sorted(glob.glob(smoke + '/trace_G_4410*kern*sum*.csv'))
if not paths:
    sys.exit('  no kernel-summary csv produced')
with open(paths[-1], newline='') as f:
    rows = list(csv.reader(f))
head = rows[0]
i_col = next(i for i, h in enumerate(head)
             if h.strip().strip('"') == 'Instances' or
             ('Instances' in h and 'Range' not in h))
inst = sum(int(float(r[i_col].replace(',', ''))) for r in rows[1:]
           if len(r) > i_col and r[i_col].strip())
run = json.load(open(smoke + '/arm_G_c4410_smoketrace.json'))
iters = len(run.get('latency_ms_per_iter') or [])
kpp = inst / iters if iters else 0
print(f"  kernels in capture range: {inst} ({kpp:.1f}/pass over {iters} iters), "
      f"scope={run.get('capture_scope')}")
# graph replay must be VISIBLE: a whole-forward or band_split graph holds
# ~60 band GEMMs minimum — near-zero kernels/pass means graph tracing failed
sys.exit(0 if iters and kpp > 30 else 1)
PYEOF
    verdict "S3 graph-node-trace" $?
fi

echo "--- [S4] P8 full-forward trace (model load + hooks + extraction) ---"
EVID_DIR="$SMOKE" bash scripts/p8_full_trace.sh > "$SMOKE/s4_p8.log" 2>&1
s4=$?
tail -5 "$SMOKE/s4_p8.log"
[ $s4 -eq 0 ] && [ -s "$SMOKE/p8_result.json" ]; verdict "S4 p8-trace" $?

echo ""
echo "================ SMOKE VERDICT ================$VERDICTS"
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "ALL $PASS PHASES PASS — pod is fit for the real session."
    echo "Next: wipe/ignore evidence_smoke/, then 'bash scripts/run_all.sh'"
    echo "inside tmux for the pre-registered session (clean evidence/)."
else
    echo "$FAIL FAILURE(S) — fix before the session; commit mechanical fixes"
    echo "so the audited hash is what runs. evidence_smoke/ holds the logs."
    exit 1
fi
