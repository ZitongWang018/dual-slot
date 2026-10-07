#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/native_env.sh"
: "${TRAIN_DATA:?Set indexed TRAIN_DATA}"
: "${VALID_DATA:?Set indexed VALID_DATA}"
: "${EOD_ID:?Set EOD_ID}"
partition="${PARTITION:?Set your Slurm GPU partition}"
export RUN_ID="${RUN_ID:-native-70m-tpp20-$(date +%Y%m%d-%H%M%S)}"
snapshot="$DUAL_SLOT_DATA_ROOT/runs-native/$RUN_ID/source"
mkdir -p "$(dirname "$snapshot")"
mkdir "$snapshot"
cd "$DUAL_SLOT_REPO_ROOT"
tar --exclude=.git --exclude=__pycache__ --exclude=private --exclude=.venv -cf - . | tar -xf - -C "$snapshot"
cd "$snapshot"
preflight=$(sbatch --parsable --partition="$partition" --output="$DUAL_SLOT_DATA_ROOT/logs-native/test-%j.log" scripts/native_smoke.sbatch)
vanilla=$(sbatch --parsable --partition="$partition" --dependency="afterok:$preflight" --output="$DUAL_SLOT_DATA_ROOT/logs-native/vanilla-%j.log" scripts/train.sbatch vanilla)
dual=$(sbatch --parsable --partition="$partition" --dependency="afterok:$vanilla" --output="$DUAL_SLOT_DATA_ROOT/logs-native/dual-slot-%j.log" scripts/train.sbatch dual-slot)
printf 'run=%s preflight=%s vanilla=%s dual_slot=%s source=%s\n' "$RUN_ID" "$preflight" "$vanilla" "$dual" "$snapshot" | tee "$DUAL_SLOT_DATA_ROOT/logs-native/submission-$RUN_ID.txt"
