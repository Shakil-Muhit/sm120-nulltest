#!/bin/bash
# Fetch + sha256-verify the public MelBandRoformer checkpoint named in
# configs/checkpoint.txt.
#
# checkpoint.txt format (FINAL — session-owner resolution 2026-08-16):
#   lines starting with # are comments — purely documentation, no exceptions;
#   then EXACTLY THREE data lines, in order:
#     1. checkpoint URL
#     2. sha256 of the checkpoint file
#     3. model config path, relative to the root of the pinned ZFTurbo clone
#        (escape hatch: a path that exists relative to THIS repo's root is
#        accepted as a vendored config — configs are never URL-fetched)
#   The config is transitively hash-pinned by the clone's commit SHA (git
#   content-addressing), so it needs no separate hash.
#
# Called by setup.sh [6/6] and defensively by scripts/p8_full_trace.sh.
# Behavior (session-owner spec):
#   - download target ${CKPT_DIR:-~/work/ckpt} — env-overridable so a RunPod
#     persistent volume can be mounted there and the download survives pods
#   - skip download if the file exists AND the sha256 matches
#   - re-download on mismatch; HARD-FAIL with a clear message if the hash
#     check fails twice; never proceed silently on a bad hash
#   - HARD-FAIL if line 3's path does not exist in the pinned clone (or,
#     escape hatch, in this repo) after checkout
#
# On success writes $CKPT_DIR/manifest.env with the resolved absolute paths
# (CKPT_FILE, CONFIG_FILE, MODEL_TYPE) for scripts/p8_full_trace.sh to source.
# MODEL_TYPE is the constant 'mel_band_roformer', env-overridable.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$REPO_DIR/configs/checkpoint.txt"
CKPT_DIR="${CKPT_DIR:-$HOME/work/ckpt}"
MSST_DIR="${MSST_DIR:-$HOME/work/msst}"
MODEL_TYPE="${MODEL_TYPE:-mel_band_roformer}"

[ -f "$CFG" ] || { echo "FATAL: $CFG missing"; exit 1; }

# strip comments and blank lines -> the data lines, in order
mapfile -t DATA < <(grep -v '^[[:space:]]*#' "$CFG" | grep '[^[:space:]]' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
if [ "${#DATA[@]}" -ne 3 ]; then
    echo "FATAL: configs/checkpoint.txt must contain EXACTLY three data lines"
    echo "       (URL / sha256 / config path relative to the pinned clone),"
    echo "       found ${#DATA[@]}. Comment lines start with #."
    exit 1
fi
CKPT_URL="${DATA[0]}"
CKPT_SHA256="$(echo "${DATA[1]}" | tr 'A-F' 'a-f')"
CONFIG_REL="${DATA[2]}"

case "$CKPT_URL" in
    http://*|https://*) ;;
    *) echo "FATAL: checkpoint.txt line 1 is not an http(s) URL: '$CKPT_URL'"; exit 1 ;;
esac
case "$CKPT_SHA256" in
    *[!0-9a-f]*|'') echo "FATAL: checkpoint.txt line 2 is not a sha256: '$CKPT_SHA256'"; exit 1 ;;
esac
[ "${#CKPT_SHA256}" -eq 64 ] \
    || { echo "FATAL: checkpoint.txt line 2 must be 64 hex chars (got ${#CKPT_SHA256})"; exit 1; }
case "$CONFIG_REL" in
    /*) echo "FATAL: checkpoint.txt line 3 must be RELATIVE to the pinned clone: '$CONFIG_REL'"; exit 1 ;;
esac

mkdir -p "$CKPT_DIR"

sha_ok() {  # $1 file  $2 expected sha256
    [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

fetch_verify() {  # $1 url  $2 expected sha256  $3 dest — 2 attempts, then FATAL
    local url="$1" sha="$2" dest="$3" attempt
    if sha_ok "$dest" "$sha"; then
        echo "cached, sha256 verified: $dest"
        return 0
    fi
    for attempt in 1 2; do
        [ -f "$dest" ] && { echo "sha256 mismatch on $dest — re-downloading"; rm -f "$dest"; }
        echo "downloading (attempt $attempt): $url"
        wget -q --show-progress -O "$dest" "$url" || rm -f "$dest"
        if sha_ok "$dest" "$sha"; then
            echo "downloaded, sha256 verified: $dest"
            return 0
        fi
    done
    rm -f "$dest"
    echo "FATAL: sha256 verification failed twice for"
    echo "       $url"
    echo "       expected $sha — check configs/checkpoint.txt against the"
    echo "       published hash. NOT proceeding on a bad hash."
    exit 1
}

# strip any ?query-string before taking the basename (hugging-face style URLs)
CKPT_FILE="$CKPT_DIR/$(basename "${CKPT_URL%%\?*}")"
fetch_verify "$CKPT_URL" "$CKPT_SHA256" "$CKPT_FILE"

# config: pinned clone first; vendored-in-this-repo escape hatch second
if [ -f "$MSST_DIR/$CONFIG_REL" ]; then
    CONFIG_FILE="$MSST_DIR/$CONFIG_REL"
elif [ -f "$REPO_DIR/$CONFIG_REL" ]; then
    CONFIG_FILE="$REPO_DIR/$CONFIG_REL"
    echo "note: config resolved from THIS repo (vendored escape hatch): $CONFIG_FILE"
else
    echo "FATAL: config path from checkpoint.txt line 3 not found:"
    echo "       neither $MSST_DIR/$CONFIG_REL"
    echo "       nor     $REPO_DIR/$CONFIG_REL"
    echo "       Is the msst clone checked out at the pinned commit (setup.sh"
    echo "       [5/6])? Does the path exist at that commit?"
    exit 1
fi

{   # values quoted: the manifest is sourced by p8_full_trace.sh and paths
    # (CKPT_DIR override) may contain spaces
    echo "CKPT_FILE=\"$CKPT_FILE\""
    echo "CONFIG_FILE=\"$CONFIG_FILE\""
    echo "MODEL_TYPE=\"$MODEL_TYPE\""
} > "$CKPT_DIR/manifest.env"
echo "manifest written: $CKPT_DIR/manifest.env"
