#!/bin/bash
#SBATCH --job-name=weld_probe
#SBATCH --partition=h200
#SBATCH --gres=gpu:7g.141gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/USERNAME/weld_generalization/logs/probe_%j.log

# Throughput probe for the cross-dataset grid. This measures, it does not
# train: every model is built from YAML with random weights and every result
# is discarded. It replaces the laptop-extrapolated 4x, 8x and 25x slice
# factors with real H200 numbers, which is what step 5 needs to size the grid.
#
# 16 CPUs, not the student default of 2. The measured dataloader ceiling at 2
# workers is about 230 img/s, and a 7g.141gb slice runs well past that for
# three of the four families, so 2 CPUs would leave most of the GPU idle. On a
# full-GPU slice, 16 CPUs is the resource needed to feed it.

set -euo pipefail

PROJECT=/home/USERNAME/weld_generalization
RESULTS=$PROJECT/results

# /tmp/Ultralytics on this node is owned by another user, so ultralytics
# cannot write its settings there and errors on every import. Keep its config
# in $HOME instead.
export YOLO_CONFIG_DIR=$HOME/.ultralytics_cfg
mkdir -p "$YOLO_CONFIG_DIR" "$RESULTS"

source /home/USERNAME/env/bin/activate
cd "$PROJECT"

echo "=== job ${SLURM_JOB_ID} on $(hostname) ==="
echo "CPUs allocated: $(nproc)"
echo "Memory:         ${SLURM_MEM_PER_NODE:-?} MB"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo

echo "=== CPU dataloader ceiling, no GPU in the loop ==="
python3 code/dataloader_ceiling.py \
    --root "$PROJECT" \
    --datasets weld_surface_v2 lohi_weld \
    --workers 2 4 8 16 \
    --batches 50 \
    --out "$RESULTS/h200_dataloader_ceiling.json"
echo

echo "=== stage 1, VRAM ceiling and synthetic rate per family ==="
python3 code/gpu_throughput.py \
    --root "$PROJECT" --stage 1 \
    --batches 8 16 32 64 128 256 \
    --out "$RESULTS/h200_stage1.json"
echo

echo "=== stage 2, real epochs, worker sweep ==="
python3 code/gpu_throughput.py \
    --root "$PROJECT" --stage 2 \
    --epochs 6 --batch 32 --workers 2 8 16 \
    --out "$RESULTS/h200_stage2.json"

echo
echo "=== done, job ${SLURM_JOB_ID} ==="
