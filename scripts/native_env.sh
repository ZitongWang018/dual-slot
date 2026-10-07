#!/usr/bin/env bash
set -euo pipefail
export DUAL_SLOT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_root="${HOME}/.local/share/dual-slot"
export DUAL_SLOT_DATA_ROOT="${DUAL_SLOT_DATA_ROOT:-$default_root}"
export PYTHON="${DUAL_SLOT_PYTHON:-$DUAL_SLOT_DATA_ROOT/native-venv/bin/python}"
export PATH="$(dirname "$PYTHON"):$DUAL_SLOT_REPO_ROOT/scripts/toolchain:$PATH"
export TMPDIR="$DUAL_SLOT_DATA_ROOT/tmp-native"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export XDG_CACHE_HOME="$DUAL_SLOT_DATA_ROOT/cache-native"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip" TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
# Compile synchronously: forked compiler arenas cannot be unlinked safely on NFS.
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache" HF_HOME="$XDG_CACHE_HOME/huggingface"
export PYTHONPATH="$DUAL_SLOT_REPO_ROOT:$DUAL_SLOT_REPO_ROOT/vendor/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1
if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  elif [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
fi
[[ -z "${CUDA_HOME:-}" ]] || export PATH="$CUDA_HOME/bin:$PATH"
if [[ -x "$PYTHON" ]]; then
  site=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  export CUDNN_PATH="${CUDNN_PATH:-$site/nvidia/cudnn}"
  export CPATH="$CUDNN_PATH/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$CUDNN_PATH/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  for include_dir in "$site"/nvidia/*/include; do
    [[ ! -d "$include_dir" ]] || export CPATH="$include_dir:$CPATH"
  done
  for library_dir in "$site"/nvidia/*/lib; do
    [[ ! -d "$library_dir" ]] || export LIBRARY_PATH="$library_dir:$LIBRARY_PATH"
  done
  export LD_LIBRARY_PATH="$site/nvidia/cudnn/lib:$site/nvidia/cublas/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
fi
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$HF_HOME" \
  "$DUAL_SLOT_DATA_ROOT/logs-native" "$DUAL_SLOT_DATA_ROOT/checkpoints-native"
if [[ -f "$DUAL_SLOT_DATA_ROOT/private/swanlab.env" ]]; then
  set -a
  source "$DUAL_SLOT_DATA_ROOT/private/swanlab.env"
  set +a
fi
export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=0
export WANDB_MODE=disabled
