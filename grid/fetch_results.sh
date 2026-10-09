#!/bin/bash
# Pull grid results off the cluster. RUN THIS FROM THE LAPTOP, repeatedly,
# while the array is running.
#
# This is the step that actually protects the grid. run_grid.slurm stages each
# cell's checkpoint and evaluation into $PROJECT/results on the cluster, but
# that directory is deleted 7 days after job completion with no warning and no
# backup. A Slurm job cannot initiate a connection back to the laptop, so the
# copy has to be a pull from this side.
#
# Safe to run as often as you like. rsync only transfers what is new, and
# nothing on the cluster is modified or removed.
#
# Usage:  ./grid/fetch_results.sh [--watch]          main grid, 69 cells
#         ./grid/fetch_results.sh --arms [--watch]   arms, 300 cells
#
# --arms pulls results_arms/ into grid_results_arms/ instead. Both are safe to
# run at once, in two terminals, because they touch different directories on
# both ends. The 7 day deletion clock applies to the arms exactly as it does to
# the main grid, and it is reset by each chain link, so the arms directory is
# at risk from the moment the LAST arm job finishes.

set -euo pipefail

HOST=nmamit-h200
REMOTE=/home/USERNAME/weld_generalization
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ARMS=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--arms" ]; then ARMS=1; else ARGS+=("$a"); fi
done
set -- "${ARGS[@]:-}"

if [ "$ARMS" = "1" ]; then
    REMOTE_RESULTS="results_arms"
    LOCAL="$ROOT/grid_results_arms"
    # Only the arms logs. The cluster `logs/` directory is SHARED with the main
    # grid and still holds 3,863 chain_*.log files, 91 MB, left by the
    # 6 September runaway in the project log 5h. The first arms pull dragged all of
    # them down before it had fetched a single result. Restricting the filter
    # keeps an arms pull proportional to the arms.
    LOG_FILTER=(--include='arms_*.log' --exclude='*')
    EXPECTED=$(cd "$ROOT" && python3 -c "
import sys; sys.path.insert(0, 'grid'); sys.path.insert(0, 'grid/arms')
import config_arms as A
A.assert_main_unmoved(); A.register()
print(len(A.all_arm_cells()))" 2>/dev/null || echo 300)
    LABEL="arms"
else
    REMOTE_RESULTS="results"
    LOCAL="$ROOT/grid_results"
    EXPECTED=69
    LABEL="main"
    # Unchanged for the main grid, which is finished and whose logs are already
    # local.
    LOG_FILTER=()
fi

# AN UNREACHABLE HOST IS NOT A FAILED PULL, AND THE LOG MUST SAY WHICH.
#
# The laptop has left the campus subnet three times now: the project log 10b, the
# 12 September gap, and 16 to 17 September. Each time the log filled with
# identical `rsync failed` lines that say nothing about the cause, so the
# state had to be reconstructed by hand afterwards. A pull that fails because
# the host is unreachable needs no action beyond rejoining the network. A pull
# that fails while the host IS reachable is a real problem and deserves
# attention. One line should tell them apart.
#
# The consecutive count lives in a file because each cron invocation is a
# fresh process and cannot remember the last one.
UNREACHABLE_COUNT="$LOCAL/.unreachable_count"

last_pull_age() {
    # Human readable age of the last SUCCESSFUL pull, per the staleness marker.
    # Phrased to read correctly in the log line, so the caller can always write
    # "last successful pull <this>".
    if [ ! -f "$LOCAL/.last_pull" ]; then
        echo "never recorded"
        return
    fi
    local when now delta
    when=$(date -d "$(cat "$LOCAL/.last_pull")" +%s 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ "$when" -le 0 ]; then echo "at an unreadable timestamp"; return; fi
    delta=$(( now - when ))
    printf '%dh%02dm ago' $(( delta / 3600 )) $(( (delta % 3600) / 60 ))
}

host_reachable() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "$HOST" true 2>/dev/null
}

fetch() {
    mkdir -p "$LOCAL"

    if ! host_reachable; then
        local n
        n=$(( $(cat "$UNREACHABLE_COUNT" 2>/dev/null || echo 0) + 1 ))
        echo "$n" > "$UNREACHABLE_COUNT"
        echo "$(date -Is)  [${LABEL}] UNREACHABLE ${HOST}, ${n} consecutive, last successful pull $(last_pull_age). Laptop is off the campus subnet; no action needed beyond rejoining."
        return 1
    fi
    # Reachable again. Reset before the transfer, so a real rsync failure on a
    # reachable host is never mislabelled as a continuing network outage.
    if [ -s "$UNREACHABLE_COUNT" ] && [ "$(cat "$UNREACHABLE_COUNT")" != "0" ]; then
        echo "$(date -Is)  [${LABEL}] reachable again after $(cat "$UNREACHABLE_COUNT") unreachable cycles"
    fi
    echo 0 > "$UNREACHABLE_COUNT"

    # Explicit, because `set -e` is suspended inside a function called as an
    # `if` condition. Without this the results rsync could fail and the cycle
    # would still report success from the last command in the function.
    if ! rsync -az --info=stats2 \
               "$HOST:$REMOTE/$REMOTE_RESULTS/" "$LOCAL/results/"; then
        echo "$(date -Is)  [${LABEL}] RSYNC FAILED while ${HOST} was REACHABLE. This is not a network drop, investigate. Last successful pull $(last_pull_age)."
        return 1
    fi
    # Logs are a convenience, not the artefact. A failure here is not fatal.
    rsync -az --info=stats2 "${LOG_FILTER[@]+"${LOG_FILTER[@]}"}" \
          "$HOST:$REMOTE/logs/" "$LOCAL/logs/" 2>/dev/null || true

    local cells evals raw
    cells=$(find "$LOCAL/results" -maxdepth 1 -type f -name '*.tar.gz' | wc -l)
    evals=$(find "$LOCAL/results" -maxdepth 1 -type f -name 'eval__*.json' | wc -l)
    raw=$(find "$LOCAL/results" -maxdepth 1 -type f -name 'raw__*.npz' | wc -l)
    echo "$(date -Is)  [${LABEL}] checkpoints=${cells}/${EXPECTED}  evaluations=${evals}/${EXPECTED}  raw=${raw}/${EXPECTED}  -> $LOCAL"

    # A cell with an archive but no evaluation means the training survived and
    # the evaluation did not. Worth knowing before the window closes.
    for a in "$LOCAL/results"/*.tar.gz; do
        [ -e "$a" ] || continue
        local n; n=$(basename "$a" .tar.gz)
        [ -s "$LOCAL/results/eval__${n}.json" ] || echo "  INCOMPLETE: ${n} has a checkpoint but no evaluation"
        [ -s "$LOCAL/results/raw__${n}.npz" ] || echo "  INCOMPLETE: ${n} has no raw prediction archive"
    done

    # STALENESS MARKER. A watcher that dies silently is the failure this script
    # exists to prevent, and it already happened once: the first arms watcher
    # was killed 5 minutes after launch when systemd tore down the login
    # session, and nothing was pulled for two days. Only a SUCCESSFUL cycle
    # touches this file, so its age is the age of the last real pull.
    #
    # If .last_pull is older than two hours while the chain is running, the
    # watcher is dead. Restart it.
    date -Is > "$LOCAL/.last_pull"
}

if [ "${1:-}" = "--watch" ]; then
    # A transient network drop must not kill the watcher. `set -e` would
    # normally take the whole loop down with the first failed rsync, which is
    # the one failure mode this script exists to prevent: a silently dead
    # watcher and a 7 day deletion clock still running. So a failed cycle is
    # reported and retried rather than fatal.
    fails=0
    while true; do
        if fetch; then
            fails=0
        else
            fails=$((fails + 1))
            echo "$(date -Is)  WARNING: fetch cycle failed (${fails} in a row), retrying next cycle"
            if [ "$fails" -ge 5 ]; then
                echo "$(date -Is)  ALERT: ${fails} consecutive failures. Check the SSH route to $HOST."
            fi
        fi
        sleep 1800
    done
else
    fetch
fi
