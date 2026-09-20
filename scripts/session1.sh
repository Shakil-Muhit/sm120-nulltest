#!/bin/bash
# SESSION 1 (prereg_amendments.md §5) — the full-model arm matrix, unprofiled.
#   [0] preflight: nsys, deps, clip pack, provenance + SESSION MANIFEST,
#       clock sampler (timestamp+uuid), session deadline clock
#   [1] eager-A references at both operating points (validation refs + SDR
#       clip outputs under the registered chunked policy)
#   [2] planted-fault validator checks: tight (+1e-3) and loose (+5e-2) —
#       both MUST FAIL; a PASS aborts the session
#   [3] C3 DC-filter operator qualification per registered frame count
#   [4] matched block-alternation pairs (plan 1.2) at 352800: (C1,C1A) once
#       (expected compile failure), then (C1,C3) (C1,C1D) (C3,H) x RESTARTS
#   [5] arm matrix, round-major (r1 of every arm, then r2, r3), priority
#       order within a round; SDR gate in r1 at BOTH operating points.
#       C1A is NOT in the matrix (its expected failure is recorded in [4]).
#   [6] C2 last: aggregate allowance; a timed-out cold attempt is not retried
#   [7] collection
# Every launch is bounded by min(cap, remaining usable time) and leaves a
# terminal JSON (scripts/session_lib.sh); unfinished mandatory cells are
# UNTESTED and the session is reported INCOMPLETE — never salvaged.
# Smoke knobs (engineering only, never evidence): SMOKE=1 selects a reduced
# matrix; ARMS / PAIRS override the lists; SKIP_C2=1.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/session_lib.sh"
C2_BUDGET_S="${C2_BUDGET_S:-2400}"
ARMS="${ARMS:-A C3 C1 H G B E D F C1D}"
PAIRS="${PAIRS:-C1:C3 C1:C1D C3:H}"
mkdir -p "$EVID" "$REF"
cd "$REPO_DIR"
exec > >(tee -a "$EVID/session1_console.log") 2>&1
echo "SESSION1 START $(date -Is) iters=$ITERS repeats=$REPEATS restarts=$RESTARTS budget=${SESSION_BUDGET_S}s reserve=${EXTRACT_RESERVE_S}s"
[ -z "${TMUX:-}" ] && echo "WARNING: not inside tmux"

echo "=== [0] preflight ==="
command -v nsys >/dev/null || { echo "FATAL: nsys missing — run setup.sh"; exit 1; }
python - <<'PYEOF' || { echo "FATAL: preflight imports failed"; exit 1; }
import torch, triton, numpy, librosa, soundfile, einops
assert torch.cuda.is_available()
print('torch', torch.__version__, '| triton', triton.__version__, '|', torch.cuda.get_device_name(0))
PYEOF
[ -d "$CLIPS_DIR" ] || { echo "FATAL: clip pack missing at $CLIPS_DIR"; exit 1; }
python - "$EVID" session1 <<'PYEOF' || { echo "FATAL: provenance / session manifest failed"; exit 1; }
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'fullmodel'))
import loader, provenance, timing
ev, session = sys.argv[1], sys.argv[2]
paths = loader.paths_from_env()
rec = provenance.create_manifest(ev, paths, timing.env_fingerprint(), session)
print('manifest', rec['manifest_id'], 'repo', rec['repo_commit'], 'msst', rec['provenance']['msst_commit'][:12],
      'ckpt', rec['provenance']['ckpt_sha256'][:12])
PYEOF
{ nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, 'cuda', torch.version.cuda)"
  nsys --version | head -1; git -C "$REPO_DIR" rev-parse HEAD; date -Is
} > "$EVID/env_snapshot_session1.txt" 2>&1
clocks_start "$EVID/clocks_session1.csv"

echo "=== [1] eager-A reference outputs (both operating points) ==="
run_one A 352800 ref --save-reference --sdr --skip-timing
run_one A 4410 ref --save-reference --sdr --skip-timing
for f in ref_A_c352800_s1001.pt ref_A_c352800_s2002.pt ref_A_c4410_s1001.pt ref_A_c4410_s2002.pt \
         sdr_A_c352800_clip00.pt sdr_A_c352800_clip07.pt sdr_A_c4410_clip00.pt sdr_A_c4410_clip07.pt manifest_id.txt; do
    [ -s "$REF/$f" ] || { echo "FATAL: reference $f missing"; exit 1; }
done

echo "=== [2] planted-fault validator checks (both must FAIL) ==="
run_one A 4410 planted_tight --planted-fault --skip-timing
run_one A 4410 planted_loose --planted-fault --planted-const 0.05 --force-tier loose --skip-timing
python3 - "$EVID/full_A_c4410_planted_tight.json" 0.001 "$EVID/full_A_c4410_planted_loose.json" 0.05 <<'PYEOF' || { echo "FATAL: a validator did not catch its planted fault"; exit 1; }
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

echo "=== [3] C3 operator qualification per registered frame count ==="
run_aux c3_operator_qualification 600 -- python - <<'PYEOF' > "$EVID/c3_operator_qualification.json" || { echo "FATAL: C3 operator qualification failed"; exit 1; }
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'fullmodel'))
import fullarms
print(json.dumps(fullarms.verify_c3_operator(), indent=1))
PYEOF
grep -q '"status": "PASS"' "$EVID/c3_operator_qualification.json" || { echo "FATAL: C3 qualification has no PASS"; exit 1; }

echo "=== [4] matched block-alternation pairs @352800 (plan 1.2) ==="
[ "${SMOKE:-0}" = "1" ] || run_matched C1 C1A 352800 m1
for r in $(seq 1 "$RESTARTS"); do
    for pr in $PAIRS; do
        run_matched "${pr%%:*}" "${pr##*:}" 352800 "m$r"
    done
done

echo "=== [5] arm matrix, round-major ==="
for r in $(seq 1 "$RESTARTS"); do
    for chunk in 352800 4410; do
        for arm in $ARMS; do
            [ "$chunk" != "352800" ] && [ "$arm" = "C1D" ] && continue
            extra=""; [ "$r" = "1" ] && extra="--sdr"
            run_one "$arm" "$chunk" "r$r" $extra
        done
    done
done

if [ "${SKIP_C2:-0}" != "1" ]; then
echo "=== [6] C2 (max-autotune) last, aggregate allowance ${C2_BUDGET_S}s ==="
C2_T0=$(date +%s)
for chunk in 352800 4410; do
    cold_timed_out=0
    for r in $(seq 1 "$RESTARTS"); do
        left=$(( C2_BUDGET_S - ($(date +%s) - C2_T0) ))
        path="$EVID/full_C2_c${chunk}_r${r}.json"
        if [ "$cold_timed_out" = "1" ]; then
            terminal_json "$path" C2 "$chunk" "r$r" UNTESTED "not attempted: identical cold attempt timed out (no-retry rule)"; continue
        fi
        if [ "$left" -lt "$T_MIN" ]; then
            terminal_json "$path" C2 "$chunk" "r$r" UNTESTED "C2 aggregate allowance exhausted"; continue
        fi
        extra=""; [ "$r" = "1" ] && extra="--sdr"
        T_COMPILE_SAVED="$T_COMPILE"; T_COMPILE=$(( left < T_COMPILE ? left : T_COMPILE ))
        run_one C2 "$chunk" "r$r" $extra
        T_COMPILE="$T_COMPILE_SAVED"
        grep -q '"status": "UNTESTED"' "$path" 2>/dev/null && [ "$r" = "1" ] && cold_timed_out=1
    done
done
fi

echo "=== [7] collection ==="
python "$REPO_DIR/fullmodel/collect_full.py" --evidence-dir "$EVID" || true
echo "SESSION1 DONE $(date -Is) elapsed=$(( $(date +%s) - T0 ))s remaining=$(remaining)s"
echo ">>> EXTRACT evidence/ OFF-POD BEFORE TERMINATE:  tar czf /root/evidence_s1_\$(date +%Y%m%d_%H%M).tgz -C $REPO_DIR evidence"
