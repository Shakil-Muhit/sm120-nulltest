#!/bin/bash
# Bring up the session pod from muhit-pc (prereg_amendments.md §5; docs/fullmodel_runbook.md).
#   1. create the RunPod secure-cloud RTX 5090 pod `melband_profiling`
#   2. wait for SSH, record the pod id/host/port in evidence-local pod.env
#   3. ship the COMMITTED tree as a git bundle (no credentials on the pod;
#      the clone keeps .git so every run records the commit hash)
#   4. ship the clip pack (audio never committed); run setup.sh on the pod
# Usage:  bash scripts/pod_up.sh [DATA_CENTER_ID]     (default EU-CZ-1)
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DCS="${1:-EU-CZ-1 EU-RO-1 EUR-IS-1 EUR-IS-2 EUR-NO-1}"   # tried in order until one has a 5090
IMAGE="runpod/pytorch:1.1.0-rc.154-cu1281-torch280-ubuntu2204"
NAME="melband_profiling"
CLIPS="${CLIPS_SRC:-$HOME/ms/data/sdr_clips}"
KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO_DIR"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "FATAL: commit the tree first (session runs the audited commit; untracked review logs are allowed — they never enter the bundle)"; exit 1; }
COMMIT=$(git rev-parse HEAD)
POD_ID=""
for DC in $DCS; do
    echo "creating pod $NAME in $DC (image $IMAGE) for commit $COMMIT"
    OUT=$(runpodctl pod create --name "$NAME" --gpu-id "NVIDIA GeForce RTX 5090" --image "$IMAGE" \
          --container-disk-in-gb 40 --ports "22/tcp" --data-center-ids "$DC" -o json 2>&1)
    echo "$OUT" | head -20
    POD_ID=$(echo "$OUT" | python3 -c "import json,sys,re; t=sys.stdin.read(); m=re.search(r'\"id\"\s*:\s*\"([a-z0-9]+)\"', t); print(m.group(1) if m else '')")
    [ -n "$POD_ID" ] && break
    echo "no pod in $DC — trying the next datacenter"
done
[ -n "$POD_ID" ] || { echo "FATAL: no 5090 pod could be created in: $DCS"; exit 1; }
echo "POD_DC=$DC" > "$REPO_DIR/pod.env"
echo "POD_ID=$POD_ID" >> "$REPO_DIR/pod.env"
echo "waiting for ssh info..."
for i in $(seq 1 60); do
    INFO=$(runpodctl ssh info "$POD_ID" 2>/dev/null)
    HOST=$(echo "$INFO" | grep -oE 'root@[0-9.]+' | head -1 | cut -d@ -f2)
    PORT=$(echo "$INFO" | grep -oE '\-p [0-9]+' | head -1 | awk '{print $2}')
    if [ -n "$HOST" ] && [ -n "$PORT" ] && ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -i "$KEY" -p "$PORT" "root@$HOST" true 2>/dev/null; then
        break
    fi
    sleep 15
done
[ -n "${HOST:-}" ] && [ -n "${PORT:-}" ] || { echo "FATAL: ssh never came up (pod $POD_ID); check runpodctl pod get $POD_ID"; exit 1; }
echo "POD_HOST=$HOST" >> "$REPO_DIR/pod.env"; echo "POD_PORT=$PORT" >> "$REPO_DIR/pod.env"
SSH="ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT root@$HOST"
SCP="scp -o StrictHostKeyChecking=no -i $KEY -P $PORT"
echo "pod up: $HOST:$PORT (id $POD_ID)"
$SSH 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; python -c "import torch,triton;print(torch.__version__, triton.__version__)"'
git bundle create /tmp/sm120.bundle HEAD main 2>/dev/null || git bundle create /tmp/sm120.bundle HEAD
$SCP /tmp/sm120.bundle "root@$HOST:/root/sm120.bundle"
$SSH "rm -rf /root/sm120-nulltest && git clone -q /root/sm120.bundle /root/sm120-nulltest && cd /root/sm120-nulltest && git checkout -q $COMMIT && git config core.hooksPath .githooks && git rev-parse HEAD"
$SSH 'mkdir -p /root/work'
$SCP -r "$CLIPS" "root@$HOST:/root/work/sdr_clips"
$SSH 'ls /root/work/sdr_clips | wc -l'
echo "running setup.sh on the pod (nsys, deps, msst @ pin, checkpoint) ..."
$SSH 'cd /root/sm120-nulltest && bash setup.sh 2>&1 | tail -15'
echo "POD READY. Next (on the pod, inside tmux):  cd /root/sm120-nulltest && tmux new -s run && bash scripts/session1.sh"
echo "ssh: $SSH"
