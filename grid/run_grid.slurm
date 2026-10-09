#!/bin/bash
#SBATCH --job-name=weld_grid
#SBATCH --partition=h200
#SBATCH --gres=gpu:7g.141gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-68%1
#SBATCH --output=/home/USERNAME/weld_generalization/logs/grid_%A_%a.log

# The 69 cell generalization grid, one array task per cell.
#
#   0..35   main matrix, 3 sources x 4 families x 3 seeds, all 5 targets
#   36..59  reference models for visual and twoclass, all four families, in
#           domain only, so each T_ij uses a same-family P_jj denominator
#   60..68  dirty baselines on the ORIGINAL shipped splits, in domain only,
#           for the pre-audit versus post-audit comparison
#
# Main cells run first, so a partially finished array still yields the
# headline matrix.
#
# %1 in the array spec is not a typo and must stay. The account may hold one
# GPU slice at a time, so the array has to run strictly serially. Without the
# throttle every task after the first sits in PD with reason AssocGrpGRES and
# the array looks broken.
#
# Two CPUs and two data loader workers follow the published server SOP. Eight
# workers were faster in the throughput probe, but scheduler acceptance is not
# permission to depart from the guide. Raise both only after written approval.
#
# The earlier Faster R-CNN probe used its default internal 800 pixel resize.
# The frozen grid now uses a true 640 by 640 input, so that old rate is only a
# conservative planning reference. Eight hours per cell leaves ample margin.
#
# Cells are idempotent. A cell with cell.json already written is skipped, so
# resubmitting the array after a fault or a time limit resumes rather than
# restarts.

set -euo pipefail

PROJECT=/home/USERNAME/weld_generalization
export YOLO_CONFIG_DIR=/home/USERNAME/.ultralytics_cfg
export MPLCONFIGDIR=/home/USERNAME/.matplotlib
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$PROJECT/logs" "$PROJECT/runs" "$PROJECT/results"

source /home/USERNAME/env/bin/activate
cd "$PROJECT"

IDX=${SLURM_ARRAY_TASK_ID}
echo "=== array ${SLURM_ARRAY_JOB_ID} task ${IDX} on $(hostname) ==="
echo "CPUs: $(nproc)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date -Is

python3 grid/train_cell.py --index "${IDX}" --runs-dir "$PROJECT/runs"

CELL=$(python3 -c "
import sys; sys.path.insert(0, 'grid')
import config as C
print(C.cell_name(C.all_cells()[${IDX}]))")
ARCHIVE="$PROJECT/results/${CELL}.tar.gz"
EVAL="$PROJECT/results/eval__${CELL}.json"
RAW="$PROJECT/results/raw__${CELL}.npz"

# Stage the checkpoint before evaluation. If evaluation fails, the laptop
# watcher can still retrieve the trained model and the full training record.
tar -czf "$ARCHIVE" -C "$PROJECT/runs" "${CELL}"
if [ ! -s "$ARCHIVE" ]; then
    echo "FATAL: training archive missing or empty after cell ${CELL}."
    exit 1
fi
echo "staged: $ARCHIVE  ($(stat -c%s "$ARCHIVE") bytes)"

# Evaluate in the same task, while the checkpoint is on local disk and the
# slice is already held. Splitting this into a second array would double the
# queue waits for no benefit, and the account only gets one slice at a time.
python3 grid/evaluate_cell.py --index "${IDX}" \
        --runs-dir "$PROJECT/runs" --out-dir "$PROJECT/results"

# Stage each cell's artefacts immediately, so nothing depends on the array
# finishing. Note carefully: this writes to $PROJECT/results, which is ON THE
# CLUSTER and is deleted 7 days after job completion along with everything
# else. A batch job cannot push to the laptop, so this is staging, NOT the
# backup. The actual copy off the cluster is a pull, run from the laptop with
# grid/fetch_results.sh while the array is still going.
# Three artefacts must exist before this task succeeds. The compressed raw
# predictions preserve enough information to recompute metrics after the
# cluster's deletion window.
for f in "$ARCHIVE" "$EVAL" "$RAW"; do
    if [ ! -s "$f" ]; then
        echo "FATAL: ${f} missing or empty after cell ${CELL}."
        echo "Results are NOT safe to leave on the cluster. Investigate before"
        echo "the 7 day deletion window closes."
        exit 1
    fi
    echo "staged: $f  ($(stat -c%s "$f") bytes)"
done

echo "=== task ${IDX} done ==="
date -Is
