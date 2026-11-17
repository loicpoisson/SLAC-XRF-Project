#!/bin/bash
# Train the learned planner on Stanford FarmShare (Slurm).
#
#   sbatch phase2_ml/farmshare_train.sbatch                # all folds, Fe Zn Mn Cu
#   sbatch phase2_ml/farmshare_train.sbatch --targets Cu.Ka --steps 500
#
# Check the partition / GPU names of the cluster first (`sinfo -o "%P %G %l"`)
# and adjust the two lines marked ADJUST. Without a GPU the job still runs on
# CPU (about 1 h per model on 8 cores; 12 models by default).
# One-time setup on the cluster: see phase2_ml/README.md ("On FarmShare").

#SBATCH --job-name=xrf-planner
#SBATCH --partition=gpu            # ADJUST to the cluster's GPU partition
#SBATCH --gres=gpu:1               # ADJUST / remove for a CPU-only run
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=phase2_ml/logs/train_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p phase2_ml/logs phase2_ml/models

# the environment created in the setup step
source "${XRF_ML_ENV:-$HOME/xrf-ml}/bin/activate"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python scripts/14_train_planner.py "$@"

# Scoring needs the independent 25 ms scans too; it is CPU work (a few minutes
# per case) and can also be done back on the laptop once the checkpoints in
# phase2_ml/models/ are copied over.
python scripts/15_evaluate_planner.py
