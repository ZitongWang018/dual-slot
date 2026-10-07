#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/native_env.sh"
venv="$DUAL_SLOT_DATA_ROOT/native-venv"
if [[ ! -x "$venv/bin/python" ]]; then
  "${PYTHON_BOOTSTRAP:-python3}" -m venv "$venv"
fi
export PATH="$venv/bin:$PATH"
"$PYTHON" -m pip install --no-compile --upgrade pip wheel 'setuptools<80' packaging ninja pybind11
"$PYTHON" -m pip install --no-compile torch==2.6.0 numpy==1.26.4
site=$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export CUDNN_PATH="$site/nvidia/cudnn"
export CPATH="$CUDNN_PATH/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDNN_PATH/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
for include_dir in "$site"/nvidia/*/include; do
  [[ ! -d "$include_dir" ]] || export CPATH="$include_dir:$CPATH"
done
for library_dir in "$site"/nvidia/*/lib; do
  [[ ! -d "$library_dir" ]] || export LIBRARY_PATH="$library_dir:$LIBRARY_PATH"
done
export LD_LIBRARY_PATH="$site/nvidia/cudnn/lib:$site/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
export NVTE_FRAMEWORK=pytorch MAX_JOBS="${MAX_JOBS:-8}" NVTE_BUILD_THREADS_PER_JOB=1
export NVTE_PYTORCH_FORCE_BUILD=TRUE
"$PYTHON" -m pip install --no-compile --no-build-isolation -r "$DUAL_SLOT_REPO_ROOT/requirements-native.txt"
if ! "$PYTHON" -c 'import fused_weight_gradient_mlp_cuda' 2>/dev/null; then
  # Pinned Apex for upstream Megatron's gradient-accumulation CUDA extension.
  revision=575968bc1f9127ccd61003a681472a83af4ff1a1
  apex="$DUAL_SLOT_DATA_ROOT/build-native/apex-$revision"
  archive="$DUAL_SLOT_DATA_ROOT/build-native/apex-$revision.tar.gz"
  mkdir -p "$(dirname "$apex")"
  if [[ ! -f "$apex/setup.py" ]]; then
    [[ -f "$archive" ]] || curl -fL --retry 5 --retry-all-errors \
      "https://codeload.github.com/NVIDIA/apex/tar.gz/$revision" -o "$archive"
    mkdir -p "$apex"
    tar -xzf "$archive" --strip-components=1 -C "$apex"
  fi
  APEX_CPP_EXT=1 APEX_CUDA_EXT=1 "$PYTHON" -m pip install --no-compile --no-build-isolation "$apex"
fi
"$PYTHON" "$DUAL_SLOT_REPO_ROOT/scripts/fix_apex_annotations.py"
"$PYTHON" "$DUAL_SLOT_REPO_ROOT/scripts/fix_te_wheel_metadata.py"
"$PYTHON" -m pip check
"$PYTHON" -c 'import torch, transformer_engine.pytorch, flash_attn, fused_weight_gradient_mlp_cuda, swanlab; print("Native dependencies ready", torch.__version__)'
