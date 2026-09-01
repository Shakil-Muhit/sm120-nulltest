#!/bin/bash
# Version-matrix cell "SM120 x latest stable" (prereg addendum + P7).
#
# PREREG environment rule: torch is NEVER upgraded on the null-test pod before
# or among the arms — so run_all.sh runs this LAST, and the latest-stable torch
# lives in an ISOLATED venv; the arm environment is never touched.
set -u
set -o pipefail   # the probe pipes through tee — its crash must not exit 0

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVID="$REPO_DIR/evidence"
VENV="${VENV:-$HOME/work/venv_latest}"
mkdir -p "$EVID"

if [ ! -x "$VENV/bin/python" ]; then
    python -m venv "$VENV" || {
        echo "FATAL: python -m venv failed (python3-venv missing?). The check"
        echo "       can also run on a separate micro-pod (prereg allows it)."
        exit 1
    }
fi
"$VENV/bin/pip" install -q --upgrade pip \
    || { echo "FATAL: pip self-upgrade failed in $VENV"; exit 1; }

echo "[version-matrix] installing latest stable torch into $VENV (arm env untouched)"
"$VENV/bin/pip" install -q --upgrade torch \
    || { echo "FATAL: torch install into $VENV failed"; exit 1; }

# the default PyPI wheel may not carry this arch — fall back to the cu128 index
if ! "$VENV/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[version-matrix] default wheel has no working CUDA here — retrying cu128 index"
    "$VENV/bin/pip" install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu128 \
        || { echo "FATAL: cu128 torch install into $VENV failed"; exit 1; }
fi

# probe failure and a missing verdict JSON must both surface as exit 1 so
# run_all.sh's WARNING branch fires (a trailing echo would reset $? to 0)
"$VENV/bin/python" "$REPO_DIR/scripts/grouped_mm_probe.py" \
        --json "$EVID/version_matrix_sm120_latest.json" \
    | tee "$EVID/version_matrix_sm120_latest.txt" \
    || { echo "FATAL: grouped_mm probe crashed"; exit 1; }
[ -s "$EVID/version_matrix_sm120_latest.json" ] \
    || { echo "FATAL: probe wrote no JSON verdict"; exit 1; }
echo "[version-matrix] -> evidence/version_matrix_sm120_latest.{json,txt}"
