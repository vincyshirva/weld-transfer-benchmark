#!/bin/bash
#SBATCH --job-name=weld_chain
#SBATCH --partition=h200
#SBATCH --gres=gpu:7g.141gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/home/USERNAME/weld_generalization/logs/chain_%j.log

# The 69 cell generalization grid, run as a self-chaining sequential job.
#
# WHY THIS EXISTS INSTEAD OF THE JOB ARRAY.
# run_grid.slurm declares `--array=0-68%1`, which is the natural shape for this
# grid. It cannot be submitted. faculty_qos sets MaxSubmitJobsPerUser=2 and
# Slurm counts every array task as a submitted job, so a 69 task array is
# rejected outright with QOSMaxSubmitJobPerUserLimit. Measured on 5 September
# 2026: arrays of size 1 and 2 are accepted, size 3 is refused.
#
# So the grid runs as one job that walks the cells in order, and chains to a
# successor when its wall clock runs low. At most two jobs exist at any moment,
# one running and one queued, which is exactly the cap.
#
# NOTHING SCIENTIFIC CHANGES. Same cells, same order, same code, same frozen
# configuration hash. This file changes only how work is handed to the
# scheduler.
#
# Cell order is unchanged and still matters:
#   0..35   main matrix, 3 sources x 4 families x 3 seeds, all 5 targets
#   36..59  reference models for visual and twoclass, all four families, in
#           domain only, so each T_ij uses a same-family P_jj denominator
#   60..68  dirty baselines on the ORIGINAL shipped splits
# Main cells run first, so a partially finished chain still yields the
# headline matrix.
#
# THE SUCCESSOR IS SUBMITTED AT START, NOT AT EXIT, with afterany. If this job
# is killed by the wall clock, by a node fault, or by an error, the successor
# still runs and picks up from the first incomplete cell. Submitting at exit
# would break the chain on exactly the failures the chain exists to survive.
#
# Cells are idempotent. train_cell.py skips any cell whose cell.json exists, so
# a restart resumes rather than repeats. A cell interrupted mid training is
# redone from scratch, which is why the reserve below is generous.

set -uo pipefail

PROJECT=/home/USERNAME/weld_generalization
export YOLO_CONFIG_DIR=/home/USERNAME/.ultralytics_cfg
export MPLCONFIGDIR=/home/USERNAME/.matplotlib
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$PROJECT/logs" "$PROJECT/runs" "$PROJECT/results"

source /home/USERNAME/env/bin/activate
cd "$PROJECT"

N_CELLS=$(python3 -c "
import sys; sys.path.insert(0, 'grid')
import config as C
print(len(C.all_cells()))")

echo "=== chain job ${SLURM_JOB_ID} on $(hostname) ==="
echo "CPUs: $(nproc)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cells: ${N_CELLS}"
date -Is

# ---------------------------------------------------------------- successor
# Queue the next link before doing any work, so a hard kill cannot break the
# chain. Cancelled at the end if every cell is finished.
SUCCESSOR=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
                   --kill-on-invalid-dep=yes \
                   "$PROJECT/grid/run_grid_chain.slurm" 2>&1)
if [[ "$SUCCESSOR" =~ ^[0-9]+$ ]]; then
    echo "successor queued: ${SUCCESSOR} (afterany:${SLURM_JOB_ID})"
else
    echo "WARNING: could not queue a successor: ${SUCCESSOR}"
    echo "This job will still run. The chain must be restarted by hand if it"
    echo "does not finish every cell."
    SUCCESSOR=""
fi

# ------------------------------------------------------------------ deadline
END_TS=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
         | tr ' ' '\n' | awk -F= '/^EndTime=/{print $2}')
if [ -n "$END_TS" ]; then
    DEADLINE=$(date -d "$END_TS" +%s 2>/dev/null || echo 0)
else
    DEADLINE=0
fi
if [ "$DEADLINE" -le 0 ]; then
    DEADLINE=$(( $(date +%s) + 23*3600 ))
    echo "WARNING: could not read EndTime, assuming 23 h from now"
fi
echo "deadline: $(date -d @${DEADLINE} -Is)"

# Reserve enough wall clock to finish the next cell, so it is not started and
# then killed halfway. Per family, from the measured H200 rates in the project log
# with a wide margin, because production runs at 2 workers rather than the 8
# used in the probe and evaluation across 5 targets follows training.
# $1 family, $2 cell name. The cell name matters because the downloaded-data
# cells train on the ORIGINAL shipped splits, and dirty weld_surface_v2 is
# 15,045 training images against 2,074 in the audited tree, a factor of seven.
# Reserving the plain YOLO11s budget for it would start a cell that cannot
# finish, and the wall clock would kill it halfway.
reserve_for() {
    case "$2" in
        dirty_weld_surface_v2__*) echo $(( 12 * 3600 )) ; return ;;
        dirty_*)                  echo $(( 4 * 3600 ))  ; return ;;
    esac
    case "$1" in
        faster_rcnn) echo $(( 6 * 3600 )) ;;
        rtdetr_l)    echo $(( 3 * 3600 )) ;;
        *)           echo $(( 2 * 3600 )) ;;
    esac
}

# --------------------------------------------------------------------- cells
COMPLETED=0
RAN=0
STOPPED_EARLY=0

for (( IDX=0; IDX<N_CELLS; IDX++ )); do
    read -r CELL FAMILY <<< "$(python3 -c "
import sys; sys.path.insert(0, 'grid')
import config as C
c = C.all_cells()[${IDX}]
print(C.cell_name(c), c['family'])")"

    if [ -f "$PROJECT/runs/${CELL}/cell.json" ] \
       && [ -s "$PROJECT/results/eval__${CELL}.json" ] \
       && [ -s "$PROJECT/results/raw__${CELL}.npz" ] \
       && [ -s "$PROJECT/results/${CELL}.tar.gz" ]; then
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    REMAIN=$(( DEADLINE - $(date +%s) ))
    NEED=$(reserve_for "$FAMILY" "$CELL")
    if [ "$REMAIN" -lt "$NEED" ]; then
        echo "=== stopping early before cell ${IDX} (${CELL}) ==="
        echo "remaining ${REMAIN}s, reserve for ${FAMILY} is ${NEED}s"
        echo "successor ${SUCCESSOR} will resume here"
        STOPPED_EARLY=1
        break
    fi

    echo
    echo "=== cell ${IDX}/${N_CELLS} ${CELL} (${FAMILY}) ==="
    echo "remaining wall clock: ${REMAIN}s"
    date -Is

    # CIRCUIT BREAKER. A deterministic failure must not become an infinite
    # retry loop. The successor is queued at job start so a wall clock kill
    # continues the chain, but that same property means a cell that fails
    # instantly respawns instantly. On 6 September 2026 a dirty cell failed in
    # evaluation, training was then skipped as already complete, and the chain
    # burned about 3,850 four-second jobs before the environment grew too large
    # to exec anything. Two strikes on one cell now stops the chain.
    FAILDIR="$PROJECT/results/.failures"
    mkdir -p "$FAILDIR"
    FAILFILE="$FAILDIR/${CELL}"
    STRIKES=$(cat "$FAILFILE" 2>/dev/null || echo 0)
    if [ "$STRIKES" -ge 2 ]; then
        echo "FATAL: cell ${IDX} (${CELL}) has already failed ${STRIKES} times."
        echo "Stopping the chain rather than retrying. Investigate, clear"
        echo "${FAILFILE}, then resubmit grid/run_grid_chain.slurm."
        [ -n "$SUCCESSOR" ] && scancel "$SUCCESSOR" 2>/dev/null
        exit 1
    fi

    fail_cell() {
        echo $((STRIKES + 1)) > "$FAILFILE"
        echo "recorded strike $((STRIKES + 1)) for ${CELL}"
        if [ "$((STRIKES + 1))" -ge 2 ] && [ -n "$SUCCESSOR" ]; then
            echo "second strike, cancelling successor ${SUCCESSOR} to stop the chain"
            scancel "$SUCCESSOR" 2>/dev/null
        fi
    }

    if ! python3 grid/train_cell.py --index "${IDX}" --runs-dir "$PROJECT/runs"; then
        echo "FATAL: training failed on cell ${IDX} (${CELL})."
        fail_cell
        exit 1
    fi

    ARCHIVE="$PROJECT/results/${CELL}.tar.gz"
    EVAL="$PROJECT/results/eval__${CELL}.json"
    RAW="$PROJECT/results/raw__${CELL}.npz"

    # Stage the checkpoint before evaluation. If evaluation fails, the laptop
    # watcher can still retrieve the trained model and the full training record.
    tar -czf "$ARCHIVE" -C "$PROJECT/runs" "${CELL}"
    if [ ! -s "$ARCHIVE" ]; then
        echo "FATAL: training archive missing or empty after cell ${CELL}."
        fail_cell
        exit 1
    fi
    echo "staged: $ARCHIVE  ($(stat -c%s "$ARCHIVE") bytes)"

    if ! python3 grid/evaluate_cell.py --index "${IDX}" \
                --runs-dir "$PROJECT/runs" --out-dir "$PROJECT/results"; then
        echo "FATAL: evaluation failed on cell ${IDX} (${CELL})."
        fail_cell
        exit 1
    fi

    # Three artefacts must exist before this cell counts as done. Note that
    # $PROJECT/results is ON THE CLUSTER and is deleted 7 days after the job
    # finishes. This is staging, NOT backup. The copy off the cluster is a
    # pull, run from the laptop with grid/fetch_results.sh.
    for f in "$ARCHIVE" "$EVAL" "$RAW"; do
        if [ ! -s "$f" ]; then
            echo "FATAL: ${f} missing or empty after cell ${CELL}."
            echo "Results are NOT safe to leave on the cluster. Investigate"
            echo "before the 7 day deletion window closes."
            fail_cell
            exit 1
        fi
        echo "staged: $f  ($(stat -c%s "$f") bytes)"
    done

    rm -f "$FAILFILE"
    RAN=$((RAN + 1))
    COMPLETED=$((COMPLETED + 1))
    echo "=== cell ${IDX} done ==="
    date -Is
done

echo
echo "=== chain job ${SLURM_JOB_ID} summary ==="
echo "cells finished this job: ${RAN}"
echo "cells complete overall:  ${COMPLETED}/${N_CELLS}"
date -Is

if [ "$COMPLETED" -ge "$N_CELLS" ] && [ "$STOPPED_EARLY" -eq 0 ]; then
    echo "GRID COMPLETE"
    if [ -n "$SUCCESSOR" ]; then
        echo "cancelling unneeded successor ${SUCCESSOR}"
        scancel "$SUCCESSOR" 2>/dev/null || true
    fi
else
    echo "grid incomplete, successor ${SUCCESSOR} continues from cell ${COMPLETED}"
fi
