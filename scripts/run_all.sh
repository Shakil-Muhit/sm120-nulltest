#!/bin/bash
# Full null-test session, in prereg order. AUDIT-MAP[4] — the ordering:
#
#   [0] preflight (imports, GPU, nsys; environment snapshot)
#   [1] P8 full-forward trace FIRST         (prereg P8: session opener,
#                                            before any arm)
#   [2] C2 output_code reads, both chunks   (prereg: "first action ... before
#                                            any timing" — runs before any
#                                            TIMED arm; P8 is a trace, not an
#                                            arm timing. Flagged in README.)
#   [3] arms A-G x two regimes x 3 restarts (PREREG: separate OS process per
#                                            run — fresh `python` every time;
#                                            >= 3 restarts for headlines)
#   [4] nsys traces + busy extraction       (analysis/traces.sh)
#   [5] collection table                    (analysis/collect.py)
#   [6] version-matrix venv check LAST      (prereg environment rule: torch
#                                            never upgraded before/among arms)
#
# Everything lands in evidence/ — extract it off-pod BEFORE terminate (README).
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVID="$REPO_DIR/evidence"
mkdir -p "$EVID"
RESTARTS="${RESTARTS:-3}"   # PREREG: >= 3 process restarts for headline numbers
ITERS="${ITERS:-50}"        # PREREG: >= 50 timed iters after warmup
# refuse quietly-degraded sessions (mirrors run_arm.py's --iters guard)
[ "$RESTARTS" -ge 3 ] || { echo "FATAL: prereg requires >= 3 restarts (RESTARTS=$RESTARTS)"; exit 1; }
[ "$ITERS" -ge 50 ]   || { echo "FATAL: prereg requires >= 50 timed iters (ITERS=$ITERS)"; exit 1; }
cd "$REPO_DIR"

# one master console log; per-run TORCH_LOGS still get dedicated files
exec > >(tee -a "$EVID/session_console.log") 2>&1
echo "SESSION START $(date -Is)  restarts=$RESTARTS iters=$ITERS"
[ -z "${TMUX:-}" ] && echo "WARNING: not inside tmux — a dropped connection kills the session"

echo "=== [0] preflight ==="
command -v nsys >/dev/null || { echo "FATAL: nsys missing — run setup.sh"; exit 1; }
python - <<'PYEOF' || { echo "FATAL: preflight failed"; exit 1; }
import torch, triton, numpy, librosa
assert torch.cuda.is_available(), 'no CUDA device'
print('torch', torch.__version__, '| triton', triton.__version__,
      '| cap', torch.cuda.get_device_capability(),
      '|', torch.cuda.get_device_name(0))
PYEOF
{ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, 'cuda', torch.version.cuda)"
} > "$EVID/env_snapshot.txt" 2>&1

echo "=== [1] P8 full-forward trace (prereg P8 — session opener, before any arm) ==="
if ! bash "$REPO_DIR/scripts/p8_full_trace.sh"; then
    if [ "${SKIP_P8:-0}" = "1" ]; then
        echo "WARNING: P8 opener FAILED but SKIP_P8=1 set — continuing."
        echo "         This deviates from prereg order; document it."
    else
        echo "FATAL: P8 opener failed (prereg: it runs before any arm)."
        echo "       Fix it (README troubleshooting) or accept the deviation"
        echo "       explicitly by re-running with SKIP_P8=1."
        exit 1
    fi
fi

echo "=== [2] C2 output_code reads at both chunk sizes (kernel inventory) ==="
for chunk in 352800 4410; do
    echo "--- output_code: C2 chunk=$chunk ---"
    # PREREG: TORCH_LOGS (perf_hints + recompile logs) on EVERY compile-arm
    # run — the output_code inventory run included
    TORCH_LOGS="output_code,recompiles,perf_hints" python "$REPO_DIR/arms/run_arm.py" \
        --arm C2 --chunk "$chunk" --tag outputcode --skip-timing \
        2> "$EVID/output_code_C2_${chunk}.log" || true
    echo "    -> evidence/output_code_C2_${chunk}.log"
done

echo "=== [3] arms: {A..G} x {352800,4410} x $RESTARTS restarts, fresh process each ==="
for chunk in 352800 4410; do
    for arm in A B C1 C2 D E F G; do
        for r in $(seq 1 "$RESTARTS"); do
            echo "--- arm=$arm chunk=$chunk restart=r$r $(date -Is) ---"
            case "$arm" in
                C1|C2)  # PREREG: TORCH_LOGS captured on every compile arm
                    TORCH_LOGS="recompiles,perf_hints" \
                    python "$REPO_DIR/arms/run_arm.py" --arm "$arm" \
                        --chunk "$chunk" --iters "$ITERS" --tag "r$r" \
                        2> "$EVID/torchlogs_${arm}_${chunk}_r${r}.log" || true
                    ;;
                *)
                    python "$REPO_DIR/arms/run_arm.py" --arm "$arm" \
                        --chunk "$chunk" --iters "$ITERS" --tag "r$r" || true
                    ;;
            esac
        done
    done
done

echo "=== [4] nsys traces + GPU-busy / kernel-count extraction ==="
bash "$REPO_DIR/analysis/traces.sh" \
    || echo "WARNING: traces incomplete — busy numbers and R will be missing"

echo "=== [5] collection table ==="
python "$REPO_DIR/analysis/collect.py" || true

echo "=== [6] version-matrix venv check (LAST — prereg environment rule) ==="
bash "$REPO_DIR/scripts/version_matrix_check.sh" \
    || echo "WARNING: version-matrix check failed — re-run standalone or use a micro-pod"

echo ""
echo "SESSION DONE $(date -Is)"
echo ">>> EXTRACT evidence/ OFF-POD BEFORE TERMINATE, e.g.:"
echo ">>>   tar czf /root/evidence_\$(date +%Y%m%d_%H%M).tgz -C $REPO_DIR evidence"
echo ">>>   then scp/runpodctl it down. Nothing in evidence/ is committed."
