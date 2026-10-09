#!/bin/bash
#SBATCH --job-name=weld_diag
#SBATCH --partition=h200
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/home/USERNAME/weld_generalization/logs/diag_%j.log

# Why did `source env/bin/activate` not take effect inside the arm chain job?
# Read only, no GPU, no training. Delete once the answer is recorded.

set -uo pipefail
PROJECT=/home/USERNAME/weld_generalization

echo "=== before activate ==="
echo "PATH=$PATH"
echo "which python3: $(which python3 2>&1)"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}"
echo "ARM_FIRST=${ARM_FIRST:-<unset>}  ARM_LAST=${ARM_LAST:-<unset>}"
echo "BASH_ENV=${BASH_ENV:-<unset>}"
echo "SLURM_EXPORT_ENV=${SLURM_EXPORT_ENV:-<unset>}"

echo
echo "=== sourcing activate, capturing its exit status ==="
source /home/USERNAME/env/bin/activate
echo "source exit status: $?"

echo
echo "=== after activate ==="
echo "PATH=$PATH"
echo "which python3: $(which python3 2>&1)"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}"
python3 -c 'import sys; print("sys.prefix", sys.prefix)' 2>&1 | tail -2
python3 -c 'import numpy; print("numpy", numpy.__version__)' 2>&1 | tail -2

echo
echo "=== explicit interpreter, no PATH reliance ==="
/home/USERNAME/env/bin/python3 -c \
    'import numpy, torch; print("numpy", numpy.__version__, "torch", torch.__version__)' 2>&1 | tail -2
