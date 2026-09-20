#!/bin/bash
# SM120 sprint — pod setup. Run once on a fresh RunPod PyTorch pod (you are root).
# Usage:  bash setup.sh
set -u   # deliberately NOT -e: fallbacks are handled explicitly

echo "=== [1/4] environment fingerprint ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| cap', torch.cuda.get_device_capability())"

echo "=== [2/4] nsight-systems ==="
if command -v nsys >/dev/null 2>&1; then
    echo "nsys already present"
else
    apt-get update -qq
    # attempt 1: whatever repos the image already carries
    if ! apt-get install -y -qq nsight-systems-cli 2>/dev/null; then
        echo "adding NVIDIA devtools repo..."
        apt-get install -y -qq wget gnupg ca-certificates
        wget -qO- https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub \
            | gpg --dearmor > /usr/share/keyrings/nvidia-devtools.gpg
        echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" \
            > /etc/apt/sources.list.d/nvidia-devtools.list
        apt-get update -qq
        # package naming drifts (nsight-systems-cli vs nsight-systems-<ver>): prefer the
        # CLI meta-package when the repo offers it (2026-09-20: candidate 2026.5.1.161),
        # else the newest versioned package; never the 2021 'nsight-systems-target' stub
        if apt-cache policy nsight-systems-cli 2>/dev/null | grep -q 'Candidate: [0-9]'; then
            PKG=nsight-systems-cli
        else
            PKG=$(apt-cache search '^nsight-systems-20' | awk '{print $1}' | sort -V | tail -1)
        fi
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${PKG:-nsight-systems-cli}"
    fi
fi
command -v nsys >/dev/null 2>&1 && nsys --version | head -1 \
    || { echo "FATAL: nsys install failed. Inspect: apt-cache search nsight-systems"; exit 1; }

echo "=== [3/4] counter check (informational; timing + nsys work even if locked) ==="
if command -v ncu >/dev/null 2>&1; then
    ncu --set basic -c 1 python -c "import torch; (torch.randn(256,256,device='cuda')@torch.randn(256,256,device='cuda')).sum().item()" 2>&1 | tail -2
else
    echo "ncu not on PATH — fine for null test"
fi

echo "=== [4/4] workspace + session hygiene ==="
apt-get install -y -qq tmux
mkdir -p ~/work/evidence
cd ~/work && echo "POD READY $(date -Is)" | tee evidence/pod_ready.txt

# ============================================================================
# Sections [5/6] and [6/6] appended per session-owner resolution (2026-08-16):
# setup.sh may be EXTENDED, not rewritten. Sections [1/4]-[4/4] above are
# byte-identical to the provided file; the closing ">>> Next: tmux" hint
# (originally the last two lines) is relocated below [6/6] so it remains the
# final output. Flagged in README "Deviations & ambiguities".
# Idempotency: every step checks-before-does — running setup.sh twice on the
# same pod is safe and fast (matches the nsys section's pattern).
# ============================================================================

# [4/4] above ended with `cd ~/work`, so a relative BASH_SOURCE (the normal
# `bash setup.sh` from the repo root) must be resolved from the ORIGINAL
# invocation directory — which is exactly what line 42's cd left in OLDPWD.
REPO_DIR="$(cd "${OLDPWD:-$PWD}" && cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSST_DIR="${MSST_DIR:-$HOME/work/msst}"

echo "=== [5/6] public model code (P8 full-forward trace ONLY — arms never touch it) ==="
# Python deps FIRST, before any gate that can exit: the arms need librosa/
# einops regardless of the P8 pin state, so a pin problem must not leave the
# pod without arm dependencies.
# Minimal set only — NOT the msst repo's full requirements.txt (flagged in
# README "Deviations"):
#   librosa einops                            — already imported by frontend.py
#   pyyaml omegaconf ml_collections soundfile wandb — utils/settings.py import chain
#   beartype rotary_embedding_torch           — models/bs_roformer/mel_band_roformer.py
#   tqdm packaging loralib                    — utils/model_utils.py, models/bs_roformer/attend.py
# (verified against the PINNED commit ea7eb9c / v1.0.21 — wandb and loralib are
#  top-level imports there, unlike at upstream HEAD)
pip install -q librosa einops pyyaml omegaconf ml_collections soundfile wandb beartype rotary_embedding_torch tqdm packaging loralib
python -c "import yaml, omegaconf, ml_collections, soundfile, wandb, beartype, rotary_embedding_torch, einops, librosa, tqdm, packaging, loralib" \
    || { echo "FATAL: P8 model-path imports failed after pip install"; exit 1; }

# Pinned commit lives in configs/msst_commit.txt (session owner fills it in).
MSST_COMMIT="$(grep -v '^[[:space:]]*#' "$REPO_DIR/configs/msst_commit.txt" 2>/dev/null | grep -m1 '[^[:space:]]' | tr -d '[:space:]')"
if [ -z "$MSST_COMMIT" ]; then
    echo "FATAL: configs/msst_commit.txt has no commit hash. Fill in the pinned"
    echo "       commit of ZFTurbo/Music-Source-Separation-Training, then re-run."
    exit 1
fi
case "$MSST_COMMIT" in
    *[!0-9a-f]*)
        echo "FATAL: '$MSST_COMMIT' in configs/msst_commit.txt is not a commit hash"; exit 1 ;;
esac
if [ "${#MSST_COMMIT}" -ne 40 ]; then
    echo "FATAL: configs/msst_commit.txt must hold the FULL 40-char commit hash"
    echo "       (got ${#MSST_COMMIT} chars: '$MSST_COMMIT') — provenance pin"
    exit 1
fi
if [ ! -d "$MSST_DIR/.git" ]; then
    git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git "$MSST_DIR" \
        || { echo "FATAL: clone of public model repo failed"; exit 1; }
fi
if [ "$(git -C "$MSST_DIR" rev-parse HEAD)" != "$MSST_COMMIT" ]; then
    git -C "$MSST_DIR" fetch --quiet origin || true
    git -C "$MSST_DIR" checkout --quiet --detach "$MSST_COMMIT" \
        || { echo "FATAL: pinned commit $MSST_COMMIT not found in $MSST_DIR"; exit 1; }
fi
echo "msst pinned at $(git -C "$MSST_DIR" rev-parse HEAD)" | tee -a ~/work/evidence/pod_ready.txt

echo "=== [6/6] checkpoint fetch (hash-verified, cached, skip-if-present) ==="
bash "$REPO_DIR/scripts/fetch_checkpoint.sh" || exit 1

echo ""
echo ">>> Next: tmux new -s run    (so a dropped browser tab can't kill the arms)"
