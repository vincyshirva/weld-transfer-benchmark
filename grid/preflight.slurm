#!/bin/bash
#SBATCH --job-name=weld_preflight
#SBATCH --partition=h200
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/USERNAME/weld_generalization/logs/preflight_%j.log

set -euo pipefail

PROJECT=/home/USERNAME/weld_generalization
export YOLO_CONFIG_DIR=/home/USERNAME/.ultralytics_cfg
export MPLCONFIGDIR=/home/USERNAME/.matplotlib
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$PROJECT/logs"

source /home/USERNAME/env/bin/activate
cd "$PROJECT"

python3 grid/preflight.py
python3 grid/test_metrics.py
python3 grid/test_postprocess.py
