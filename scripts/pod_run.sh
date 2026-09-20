#!/bin/bash
# Drive the session pod from muhit-pc (after scripts/pod_up.sh wrote pod.env).
#   pod_run.sh ship        re-ship the COMMITTED tree (git bundle) to the pod, keep evidence/
#   pod_run.sh session1    launch scripts/session1.sh in a detached tmux session on the pod
#   pod_run.sh session2    launch scripts/session2.sh likewise
#   pod_run.sh tail        follow the pod's session console log
#   pod_run.sh status      tmux + GPU + last console lines
#   pod_run.sh extract     tar the pod's evidence/ into ~/ms/evidence_pod/<ts>/ (OFF-POD copy)
#   pod_run.sh terminate   terminate THIS pod by exact id (refuses if evidence was never extracted)
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
export PATH="$HOME/.local/bin:$PATH"
[ -f "$REPO_DIR/pod.env" ] || { echo "FATAL: pod.env missing (run scripts/pod_up.sh)"; exit 1; }
. "$REPO_DIR/pod.env"
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -i $KEY -p $POD_PORT root@$POD_HOST"
SCP="scp -o StrictHostKeyChecking=no -i $KEY -P $POD_PORT"
CMD="${1:-status}"
case "$CMD" in
  ship)
    cd "$REPO_DIR"
    [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "FATAL: commit first"; exit 1; }
    COMMIT=$(git rev-parse HEAD)
    git bundle create /tmp/sm120.bundle HEAD main 2>/dev/null || git bundle create /tmp/sm120.bundle HEAD
    $SCP /tmp/sm120.bundle "root@$POD_HOST:/root/sm120.bundle"
    # keep evidence/ (git-ignored) across re-ships; refuse while a session runs
    $SSH "tmux has-session -t run 2>/dev/null && { echo 'FATAL: a session is running'; exit 1; }
          set -e; cd /root/sm120-nulltest && git fetch -q /root/sm120.bundle HEAD && git checkout -q $COMMIT && git status --porcelain --untracked-files=no | wc -l && git rev-parse HEAD"
    ;;
  session1|session2)
    $SSH "cd /root/sm120-nulltest && mkdir -p evidence && { [ -e evidence/.terminating ] && { echo 'FATAL: termination lock present'; exit 1; }; tmux has-session -t run 2>/dev/null && { echo 'FATAL: a tmux session named run already exists'; exit 1; }; }
          export MSST_DIR=/root/work/msst CLIPS_DIR=/root/work/sdr_clips PYTHONDONTWRITEBYTECODE=1
          tmux new -d -s run \"bash scripts/$CMD.sh > evidence/${CMD}.log 2>&1; echo EXIT=\\\$? >> evidence/${CMD}.log\"
          sleep 3; tail -n 5 evidence/${CMD}.log"
    ;;
  tail)
    LOGN="${2:-session1}"
    $SSH "tail -n 40 -F /root/sm120-nulltest/evidence/${LOGN}.log"
    ;;
  status)
    $SSH "tmux ls 2>/dev/null || echo 'no tmux session'; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,clocks.sm --format=csv,noheader; ls /root/sm120-nulltest/evidence 2>/dev/null | wc -l; for f in /root/sm120-nulltest/evidence/session*.log; do [ -f \$f ] && { echo \"--- \$f\"; tail -n 6 \$f; }; done"
    ;;
  extract)
    set -o pipefail
    TS=$(date +%Y%m%d_%H%M)
    DEST="$HOME/ms/evidence_pod/${POD_ID}_$TS"
    mkdir -p "$DEST"
    # writers must be stopped and the content must not move during extraction (gate-06 f95)
    $SSH "tmux has-session -t run 2>/dev/null" && { echo "FATAL: a session is still running on the pod"; exit 1; }
    HASH_CMD="cd /root/sm120-nulltest && find evidence -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1"
    PRE_SHA=$($SSH "$HASH_CMD") || { echo "FATAL: remote content hash failed"; exit 1; }
    REMOTE_N=$($SSH "cd /root/sm120-nulltest && find evidence -type f | wc -l") || { echo "FATAL: cannot count remote files"; exit 1; }
    $SSH "cd /root/sm120-nulltest && tar czf - evidence" > "$DEST/evidence.tgz" || { echo "FATAL: transfer failed"; exit 1; }
    tar tzf "$DEST/evidence.tgz" > /dev/null || { echo "FATAL: archive invalid"; exit 1; }
    tar xzf "$DEST/evidence.tgz" -C "$DEST" || { echo "FATAL: extraction failed"; exit 1; }
    LOCAL_N=$(find "$DEST/evidence" -type f | wc -l)
    [ "$LOCAL_N" -eq "$REMOTE_N" ] || { echo "FATAL: file count mismatch remote=$REMOTE_N local=$LOCAL_N"; exit 1; }
    SHA=$(sha256sum "$DEST/evidence.tgz" | cut -d" " -f1)
    # content hash of the REMOTE evidence at extraction time: the receipt is
    # invalid once any later measurement changes the pod's evidence (gate-05 f77)
    REMOTE_SHA=$($SSH "$HASH_CMD") || { echo "FATAL: remote content hash failed"; exit 1; }
    [ "$REMOTE_SHA" = "$PRE_SHA" ] || { echo "FATAL: remote evidence changed during extraction (before $PRE_SHA, after $REMOTE_SHA)"; exit 1; }
    LOCAL_SHA=$(cd "$DEST" && find evidence -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
    [ "$LOCAL_SHA" = "$REMOTE_SHA" ] || { echo "FATAL: extracted content manifest differs from the remote one"; exit 1; }
    echo "POD_ID=$POD_ID DEST=$DEST FILES=$LOCAL_N SHA256=$SHA REMOTE=$REMOTE_SHA TS=$TS" >> "$REPO_DIR/pod.env.extracted"
    du -sh "$DEST"; ls "$DEST/evidence" | head -20
    echo "EXTRACTED -> $DEST ($LOCAL_N files, sha256 $SHA)"
    ;;
  terminate)
    grep -q "^POD_ID=$POD_ID " "$REPO_DIR/pod.env.extracted" 2>/dev/null \
        || { echo "FATAL: no validated extraction receipt for pod $POD_ID — extract first"; exit 1; }
    # exclusion held through the decision (gate-07 f109): the lock refuses any new
    # session launch; a running session refuses termination
    $SSH "cd /root/sm120-nulltest && touch evidence/.terminating && { tmux has-session -t run 2>/dev/null && { rm -f evidence/.terminating; echo 'FATAL: a session is running'; exit 1; }; exit 0; }" \
        || { echo "FATAL: could not take the termination lock"; exit 1; }
    LAST=$(grep "^POD_ID=$POD_ID " "$REPO_DIR/pod.env.extracted" | tail -1)
    D=$(echo "$LAST" | sed -n "s/.*DEST=\([^ ]*\).*/\1/p"); H=$(echo "$LAST" | sed -n "s/.*SHA256=\([^ ]*\).*/\1/p")
    R=$(echo "$LAST" | sed -n "s/.*REMOTE=\([^ ]*\).*/\1/p")
    [ -f "$D/evidence.tgz" ] && [ "$(sha256sum "$D/evidence.tgz" | cut -d" " -f1)" = "$H" ] \
        || { echo "FATAL: extracted archive missing or changed ($D)"; exit 1; }
    NOW=$($SSH "cd /root/sm120-nulltest && find evidence -type f ! -name .terminating -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1") || { echo "FATAL: cannot hash remote evidence"; exit 1; }
    [ -n "$R" ] && [ "$NOW" = "$R" ] || { $SSH "rm -f /root/sm120-nulltest/evidence/.terminating"; echo "FATAL: remote evidence changed since the last extraction (extract again)"; exit 1; }
    echo "terminating pod $POD_ID (melband_profiling) — production pods untouched; evidence at $D"
    runpodctl pod terminate "$POD_ID"
    ;;
  *) echo "usage: $0 ship|session1|session2|tail [name]|status|extract|terminate"; exit 2 ;;
esac
