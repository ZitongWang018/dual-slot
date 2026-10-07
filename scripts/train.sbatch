#!/usr/bin/env bash
#SBATCH --job-name=dual-slot-native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export GPUS=8 OMP_NUM_THREADS=4
exec bash scripts/native_train.sh "${1:-${METHOD:-vanilla}}" "${@:2}"
