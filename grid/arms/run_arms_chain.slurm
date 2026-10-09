#!/bin/bash
#SBATCH --job-name=weld_arms
#SBATCH --partition=h200
#SBATCH --gres=gpu:7g.141gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/home/USERNAME/weld_generalization/logs/arms_%j.log

# The 300 cell arm grid, as a self-chaining sequential job.
#
# Same shape as grid/run_grid_chain.slurm and for the same reason: faculty_qos
# sets MaxSubmitJobsPerUser=2 and Slurm counts every array task as a submitted
# job, so a 300 task array is refused outright. This walks the cells in order
# in one job and queues exactly one successor, so at most two jobs exist.
#
# ORDER, and it matters if the chain runs out of wall clock:
#   0..23     LoHi control. The cheapest arm and the control every other
#             number is read against. The project log 7.
#   24..263   Arm B, k-shot. Carries the headline. The project log 6a.
#   264..299  Arm D, leave one dataset out. Most expensive per cell and least
#             load bearing. The project log 6b.
#
# THE SUCCESSOR IS SUBMITTED AT START with afterany, so a wall clock kill or a
# node fault continues the chain rather than ending it silently.
#
# THE CIRCUIT BREAKER IS THE OTHER HALF OF THAT, and it is not optional. The
# same property that survives a kill makes a deterministic failure respawn
# instantly. On 6 September 2026 that cost about 3,850 four second jobs before
# the environment grew too large to exec anything. Two strikes on one cell
# stops the chain. See the project log 5h.

set -uo pipefail

PROJECT=/home/USERNAME/weld_generalization
export YOLO_CONFIG_DIR=/home/USERNAME/.ultralytics_cfg
export MPLCONFIGDIR=/home/USERNAME/.matplotlib
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$PROJECT/logs" \
         "$PROJECT/runs_arms" "$PROJECT/results_arms"

source /home/USERNAME/env/bin/activate

# CALL THE VENV INTERPRETER BY ABSOLUTE PATH. Do not rely on PATH here.
#
# Found 16 September 2026 while relaunching arm B. Inside a batch job on this
# node, `which python3` correctly resolved to the venv while a bare `python3`
# still ran /usr/bin/python3, so `import numpy` failed on the first cell. A
# login profile hook on the node emits "environment: slurm_reminder: command
# not found" and evidently invokes python3 before the script does, which
# populates bash's command hash table; the hash is consulted in preference to
# PATH, and `which` does not consult it, so the two disagree.
#
# `hash -r` clears that table, and the explicit path makes the question moot.
# Both are here because the second is the one that cannot silently regress.
# This is the same lesson as the project log 4c, 5h and 10: resolve a path against
# something stable, never against ambient process state.
hash -r
PY=/home/USERNAME/env/bin/python3
if ! "$PY" -c 'import numpy, torch, ultralytics' 2>/dev/null; then
    echo "FATAL: ${PY} cannot import numpy, torch and ultralytics."
    exit 1
fi
echo "interpreter: ${PY}"
"$PY" -c 'import sys, numpy, torch; print("  prefix", sys.prefix, "numpy", numpy.__version__, "torch", torch.__version__)'

cd "$PROJECT"

# THE DEPLOYED CODE MUST SUPPORT WHAT THIS SCRIPT ASKS OF IT. Fail at job
# start, not four minutes into the first cell's training.
#
# Found 16 September 2026: the arm B relaunch trained cell 24 correctly and
# then died because the CLUSTER copy of evaluate_arm.py predated both the
# 6a-ii `--weights` amendment and the 6a-iii class space correction. The files
# had been uploaded by hand, one list at a time, and that one was forgotten.
# Worse than the crash: the `eval__` it wrote first was in the WRONG class
# space and looked perfectly normal.
#
# Enumerating files to upload is the bug. Sync the directory. This check is
# the backstop for the next time someone enumerates anyway.
if ! "$PY" grid/arms/evaluate_arm.py --help 2>&1 | grep -q -- '--weights'; then
    echo "FATAL: the deployed grid/arms/evaluate_arm.py does not support"
    echo "--weights, so it predates the project log 6a-ii and probably 6a-iii too."
    echo "Sync grid/arms/ to the cluster before launching."
    exit 1
fi
for needed in prediction_class_space assert_ids_within_head \
              guard_degenerate_image_ap; do
    if ! grep -q "def ${needed}" grid/arms/evaluate_arm.py; then
        echo "FATAL: deployed evaluate_arm.py lacks ${needed}, so it predates"
        echo "the the project log 6a-iii class space correction. Sync grid/arms/."
        exit 1
    fi
done
echo "evaluator: supports --weights and carries the 6a-iii class space fix"

N_CELLS=$("$PY" -c "
import sys; sys.path.insert(0, 'grid'); sys.path.insert(0, 'grid/arms')
import config_arms as A
A.assert_main_unmoved(); A.register()
print(len(A.all_arm_cells()))")

# OPTIONAL INDEX RANGE, so one arm can be rerun without touching the others.
#
# Added 16 September 2026 for the the project log 6a-viii arm B retrain. Arm B is
# indices 24..263 and is the ONLY arm that has to be redone: the 24 LoHi
# control and 36 arm D cells trained through train_cell's frozen trainers and
# were never affected. Without this the chain would happily retrain all 300,
# because 10c deleted results_arms/ from the cluster so its idempotency check
# sees nothing complete.
#
#   sbatch --export=ALL,ARM_FIRST=24,ARM_LAST=263 grid/arms/run_arms_chain.slurm
#
# Defaults to the whole grid, so an unset variable behaves exactly as before.
FIRST=${ARM_FIRST:-0}
LAST=${ARM_LAST:-$((N_CELLS - 1))}
if [ "$FIRST" -lt 0 ] || [ "$LAST" -ge "$N_CELLS" ] || [ "$FIRST" -gt "$LAST" ]; then
    echo "FATAL: bad index range ${FIRST}..${LAST} against ${N_CELLS} cells"
    exit 1
fi
N_RANGE=$(( LAST - FIRST + 1 ))

echo "=== arms chain job ${SLURM_JOB_ID} on $(hostname) ==="
echo "CPUs: $(nproc)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cells: ${N_CELLS} total, running ${FIRST}..${LAST} (${N_RANGE} cells)"
date -Is

# ---------------------------------------------------------------- successor
# THE RANGE MUST PROPAGATE. A successor that forgot it would resume at
# cell 0 and retrain the sound LoHi control and arm D cells.
SUCCESSOR=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
                   --kill-on-invalid-dep=yes \
                   --export=ALL,ARM_FIRST=${FIRST},ARM_LAST=${LAST} \
                   "$PROJECT/grid/arms/run_arms_chain.slurm" 2>&1)
if [[ "$SUCCESSOR" =~ ^[0-9]+$ ]]; then
    echo "successor queued: ${SUCCESSOR} (afterany:${SLURM_JOB_ID})"
else
    echo "WARNING: could not queue a successor: ${SUCCESSOR}"
    SUCCESSOR=""
fi

# ------------------------------------------------------------------ deadline
END_TS=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
         | tr ' ' '\n' | awk -F= '/^EndTime=/{print $2}')
DEADLINE=$([ -n "$END_TS" ] && date -d "$END_TS" +%s 2>/dev/null || echo 0)
if [ "$DEADLINE" -le 0 ]; then
    DEADLINE=$(( $(date +%s) + 23*3600 ))
    echo "WARNING: could not read EndTime, assuming 23 h from now"
fi
echo "deadline: $(date -d @${DEADLINE} -Is)"

# Reserve enough wall clock to finish the next cell rather than start and be
# killed halfway. Arm B fine-tunes 50 epochs on at most a few hundred images,
# so it is cheap. Arm D pools are the largest training sets in the whole study,
# up to 4,189 images, so they get the widest reserve.
# $1 arm, $2 family
reserve_for() {
    case "$1" in
        armb) echo $(( 1 * 3600 )) ; return ;;
    esac
    case "$1:$2" in
        armd:faster_rcnn) echo $(( 12 * 3600 )) ;;
        armd:rtdetr_l)    echo $(( 8 * 3600 ))  ;;
        armd:*)           echo $(( 5 * 3600 ))  ;;
        *:faster_rcnn)    echo $(( 6 * 3600 ))  ;;
        *:rtdetr_l)       echo $(( 3 * 3600 ))  ;;
        *)                echo $(( 2 * 3600 ))  ;;
    esac
}

COMPLETED=0
RAN=0
STOPPED_EARLY=0

for (( IDX=FIRST; IDX<=LAST; IDX++ )); do
    read -r CELL ARM FAMILY <<< "$("$PY" -c "
import sys; sys.path.insert(0, 'grid'); sys.path.insert(0, 'grid/arms')
import config_arms as A
A.assert_main_unmoved(); A.register()
c = A.all_arm_cells()[${IDX}]
print(A.arm_cell_name(c), c['arm'], c['family'])")"

    # Arm B needs BOTH endpoints, so a cell missing `last` is not complete.
    # The project log 6a-ii makes `last` the PRIMARY arm B endpoint. It used to be
    # produced on the laptop from the pulled archives, because the amendment
    # landed after the first chain was already running. There is no reason for
    # that now: one extra evaluation is trivial beside 50 epochs of training,
    # and doing it here removes a manual laptop step and the risk of the two
    # endpoints drifting out of sync.
    EXTRA_OK=1
    if [ "$ARM" = "armb" ]; then
        [ -s "$PROJECT/results_arms/eval_last__${CELL}.json" ] \
            && [ -s "$PROJECT/results_arms/raw_last__${CELL}.npz" ] \
            || EXTRA_OK=0
    fi
    if [ -f "$PROJECT/runs_arms/${CELL}/cell.json" ] \
       && [ -s "$PROJECT/results_arms/eval__${CELL}.json" ] \
       && [ -s "$PROJECT/results_arms/raw__${CELL}.npz" ] \
       && [ -s "$PROJECT/results_arms/${CELL}.tar.gz" ] \
       && [ "$EXTRA_OK" -eq 1 ]; then
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    REMAIN=$(( DEADLINE - $(date +%s) ))
    NEED=$(reserve_for "$ARM" "$FAMILY")
    if [ "$REMAIN" -lt "$NEED" ]; then
        echo "=== stopping early before cell ${IDX} (${CELL}) ==="
        echo "remaining ${REMAIN}s, reserve for ${ARM}/${FAMILY} is ${NEED}s"
        echo "successor ${SUCCESSOR} will resume here"
        STOPPED_EARLY=1
        break
    fi

    echo
    echo "=== arm cell ${IDX} (${FIRST}..${LAST}) ${CELL} (${ARM}/${FAMILY}) ==="
    echo "remaining wall clock: ${REMAIN}s"
    date -Is

    FAILDIR="$PROJECT/results_arms/.failures"
    mkdir -p "$FAILDIR"
    FAILFILE="$FAILDIR/${CELL}"
    STRIKES=$(cat "$FAILFILE" 2>/dev/null || echo 0)
    if [ "$STRIKES" -ge 2 ]; then
        echo "FATAL: cell ${IDX} (${CELL}) has already failed ${STRIKES} times."
        echo "Stopping the chain rather than retrying. Investigate, clear"
        echo "${FAILFILE}, then resubmit grid/arms/run_arms_chain.slurm."
        [ -n "$SUCCESSOR" ] && scancel "$SUCCESSOR" 2>/dev/null
        exit 1
    fi

    fail_cell() {
        echo $((STRIKES + 1)) > "$FAILFILE"
        echo "recorded strike $((STRIKES + 1)) for ${CELL}"
        if [ "$((STRIKES + 1))" -ge 2 ] && [ -n "$SUCCESSOR" ]; then
            echo "second strike, cancelling successor ${SUCCESSOR}"
            scancel "$SUCCESSOR" 2>/dev/null
        fi
    }

    if ! "$PY" grid/arms/train_arm.py --index "${IDX}" \
                --runs-dir "$PROJECT/runs_arms" \
                --main-runs-dir "$PROJECT/runs"; then
        echo "FATAL: training failed on arm cell ${IDX} (${CELL})."
        fail_cell
        exit 1
    fi

    ARCHIVE="$PROJECT/results_arms/${CELL}.tar.gz"
    EVAL="$PROJECT/results_arms/eval__${CELL}.json"
    RAW="$PROJECT/results_arms/raw__${CELL}.npz"

    # Stage the checkpoint BEFORE evaluation, so an evaluation failure never
    # strands an unarchived model.
    tar -czf "$ARCHIVE" -C "$PROJECT/runs_arms" "${CELL}"
    if [ ! -s "$ARCHIVE" ]; then
        echo "FATAL: training archive missing or empty after ${CELL}."
        fail_cell
        exit 1
    fi

    if ! "$PY" grid/arms/evaluate_arm.py --index "${IDX}" \
                --runs-dir "$PROJECT/runs_arms" \
                --out-dir "$PROJECT/results_arms"; then
        echo "FATAL: evaluation failed on arm cell ${IDX} (${CELL})."
        fail_cell
        exit 1
    fi

    # Arm B's PRIMARY endpoint, the project log 6a-ii. `best` above is the secondary
    # one and is selected on the TARGET's valid split, which spends up to 444
    # further target labels and so is not the label cost the curve claims.
    if [ "$ARM" = "armb" ]; then
        if ! "$PY" grid/arms/evaluate_arm.py --index "${IDX}" \
                    --runs-dir "$PROJECT/runs_arms" \
                    --out-dir "$PROJECT/results_arms" \
                    --weights last; then
            echo "FATAL: last-endpoint evaluation failed on arm cell ${IDX} (${CELL})."
            fail_cell
            exit 1
        fi
    fi

    # results_arms/ is ON THE CLUSTER and is deleted 7 days after the job
    # finishes. This is staging, NOT backup. The real copy is a pull from the
    # laptop with grid/fetch_results.sh --watch.
    REQUIRED=("$ARCHIVE" "$EVAL" "$RAW")
    if [ "$ARM" = "armb" ]; then
        REQUIRED+=("$PROJECT/results_arms/eval_last__${CELL}.json"
                   "$PROJECT/results_arms/raw_last__${CELL}.npz")
    fi
    for f in "${REQUIRED[@]}"; do
        if [ ! -s "$f" ]; then
            echo "FATAL: ${f} missing or empty after ${CELL}."
            fail_cell
            exit 1
        fi
        echo "staged: $f  ($(stat -c%s "$f") bytes)"
    done

    rm -f "$FAILFILE"
    RAN=$((RAN + 1))
    COMPLETED=$((COMPLETED + 1))
    echo "=== arm cell ${IDX} done ==="
    date -Is
done

echo
echo "=== arms chain job ${SLURM_JOB_ID} summary ==="
echo "cells finished this job: ${RAN}"
echo "cells complete in range: ${COMPLETED}/${N_RANGE}  (${FIRST}..${LAST})"
date -Is

if [ "$COMPLETED" -ge "$N_RANGE" ] && [ "$STOPPED_EARLY" -eq 0 ]; then
    echo "ARMS COMPLETE for range ${FIRST}..${LAST}"
    if [ -n "$SUCCESSOR" ]; then
        echo "cancelling unneeded successor ${SUCCESSOR}"
        scancel "$SUCCESSOR" 2>/dev/null || true
    fi
else
    echo "arms incomplete, successor ${SUCCESSOR} continues"
fi
