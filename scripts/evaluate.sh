#!/usr/bin/env bash
set -euo pipefail
method="${1:?Usage: evaluate.sh vanilla|dual-slot /path/native-checkpoint [Megatron arguments]}"
checkpoint="${2:?Supply native checkpoint root}"
shift 2
export RESUME_CHECKPOINT="$checkpoint"
export RUN_ID="native-eval-$(date +%Y%m%d-%H%M%S)" SWANLAB="${SWANLAB:-0}"
exec bash "$(dirname "$0")/native_train.sh" "$method" --skip-train --no-load-optim --no-load-rng "$@"
