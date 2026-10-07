#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/native_env.sh"
cd "$DUAL_SLOT_REPO_ROOT"
method="${1:?Usage: native_train.sh vanilla|dual-slot [Megatron arguments]}"
shift
case "$method" in vanilla|dual-slot) ;; *) echo 'Unknown method' >&2; exit 2 ;; esac
: "${TRAIN_DATA:?Set indexed TRAIN_DATA prefix without .bin/.idx}"
: "${VALID_DATA:?Set indexed VALID_DATA prefix without .bin/.idx}"
: "${EOD_ID:?Set tokenizer EOD ID}"
for prefix in "$TRAIN_DATA" "$VALID_DATA"; do
  [[ -f "$prefix.bin" && -f "$prefix.idx" ]] || { echo "Missing indexed dataset: $prefix" >&2; exit 2; }
done
source configs/70m.sh
run_id="${RUN_ID:-native-70m-tpp20-seed42}"
output="${OUTPUT_ROOT:-$DUAL_SLOT_DATA_ROOT/checkpoints-native}/$run_id/$method"
mkdir -p "$output"
args=("${architecture[@]}" "${schedule[@]}"
  --group-query-attention --kv-channels 64
  --normalization RMSNorm --norm-epsilon 1e-6 --qk-layernorm --swiglu --disable-bias-linear
  --position-embedding-type rope --rotary-base 1000000
  --tokenizer-type NullTokenizer --vocab-size 50303 --make-vocab-size-divisible-by 128 --eod-id "$EOD_ID"
  --seq-length 2048 --max-position-embeddings 2048
  --lr 0.0015 --min-lr 0.00015 --lr-decay-style cosine
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --clip-grad 1.0
  --init-method-std 0.02 --hidden-dropout 0 --attention-dropout 0 --seed 42
  --bf16 --transformer-impl transformer_engine --attention-backend flash
  --cross-entropy-loss-fusion --no-masked-softmax-fusion
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
  --train-data-path "$TRAIN_DATA" --valid-data-path "$VALID_DATA" --test-data-path "$VALID_DATA"
  --data-cache-path "${DATA_CACHE:-$DUAL_SLOT_DATA_ROOT/cache-native/data/$run_id}"
  --num-workers 4 --no-create-attention-mask-in-dataloader
  --log-interval 1 --log-throughput --log-energy --log-timers-to-tensorboard --tensorboard-log-interval 1
  --eval-interval 100 --eval-iters 4 --save-interval 200 --ckpt-format torch_dist
  --save "$output" --tensorboard-dir "$output/tensorboard" --run-name "$run_id-$method")
[[ "$method" == vanilla ]] || args+=(--dual-slot)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  args+=(--load "$RESUME_CHECKPOINT")
elif [[ -f "$output/latest_checkpointed_iteration.txt" ]]; then
  # A Slurm requeue must resume this exact run, never overwrite it or abort.
  args+=(--load "$output")
fi
if [[ "${SWANLAB:-0}" != 0 ]]; then
  args+=(--swanlab --swanlab-project "${SWANLAB_PROJECT:-dual-slot}")
  [[ -z "${SWANLAB_WORKSPACE:-}" ]] || args+=(--swanlab-workspace "$SWANLAB_WORKSPACE")
fi
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc-per-node="${GPUS:-8}" pretrain_native.py "${args[@]}" "$@"
