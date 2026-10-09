#!/bin/bash
#SBATCH --job-name=weld_verify
#SBATCH --partition=h200
#SBATCH --gres=gpu:7g.141gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/home/USERNAME/weld_generalization/logs/verify_%j.log

set -euo pipefail

source /home/USERNAME/env/bin/activate
cd /home/USERNAME/weld_generalization

echo "Slurm job: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
nvidia-smi

python3 -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA runtime:", torch.version.cuda); print("Device:", torch.cuda.get_device_name(0)); p = torch.cuda.get_device_properties(0); print("Visible VRAM GiB:", round(p.total_memory / 1024**3, 2)); x = torch.ones(1, device="cuda"); print("CUDA tensor check:", x.item())'
