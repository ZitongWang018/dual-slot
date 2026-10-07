#!/usr/bin/env bash
#SBATCH --job-name=dual-slot-native-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=02:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/native_env.sh
export OMP_NUM_THREADS=4
export TRAIN_DATA="${TRAIN_DATA:?Set indexed TRAIN_DATA}"
export VALID_DATA="${VALID_DATA:?Set indexed VALID_DATA}"
export EOD_ID="${EOD_ID:?Set EOD_ID}" GPUS=8
test_workers="${NATIVE_TEST_WORKERS:-4}"
# Control FlashAttention backward reductions for the resume equivalence check.
# Formal training leaves TE's reference default unchanged.
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export RUN_ID="native-check-${SLURM_JOB_ID}"
"$PYTHON" -m pytest -q tests/test_dual_slot_native.py
"$PYTHON" scripts/check_native_gpu.py
for method in vanilla dual-slot; do
  bash scripts/native_train.sh "$method" --audit-native --exit-interval 2 \
    --eval-interval 1 --eval-iters 1 --save-interval 1 --num-workers "$test_workers" \
    --log-memory-to-tensorboard --log-validation-ppl-to-tensorboard
done
# Resume from the saved step 2; upstream restores data/RNG/Adam states and LR.
RUN_ID="native-check-${SLURM_JOB_ID}" bash scripts/native_train.sh dual-slot \
  --audit-native --exit-interval 3 --eval-interval 1 --eval-iters 1 \
  --save-interval 1 --num-workers "$test_workers"
RUN_ID="native-uninterrupted-${SLURM_JOB_ID}" bash scripts/native_train.sh dual-slot \
  --audit-native --exit-interval 3 --eval-interval 1 --eval-iters 1 \
  --save-interval 1 --num-workers "$test_workers"
"$PYTHON" scripts/compare_native_checkpoints.py \
  "$DUAL_SLOT_DATA_ROOT/checkpoints-native/native-uninterrupted-${SLURM_JOB_ID}/dual-slot/iter_0000003" \
  "$DUAL_SLOT_DATA_ROOT/checkpoints-native/native-check-${SLURM_JOB_ID}/dual-slot/iter_0000003"
for method in vanilla dual-slot; do
  bash scripts/evaluate.sh "$method" \
    "$DUAL_SLOT_DATA_ROOT/checkpoints-native/native-check-${SLURM_JOB_ID}/$method" \
    --eval-iters 1 --num-workers "$test_workers"
done
echo 'NATIVE_EIGHT_GPU_TEST_COMPLETE'
