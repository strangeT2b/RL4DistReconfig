#!/usr/bin/env bash

# Central cache/temp routing for training and evaluation jobs.
# On the remote H800 box this keeps large generated files off the small root FS.

_CACHE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CACHE_REPO_ROOT="$(cd "${_CACHE_SCRIPT_DIR}/.." && pwd)"

if [[ -f "${_CACHE_REPO_ROOT}/.env" ]]; then
  set -a
  source "${_CACHE_REPO_ROOT}/.env"
  set +a
fi

if [[ -z "${PROJECT_STORAGE_ROOT:-}" ]]; then
  if [[ -d /mnt/disk2/gzt ]]; then
    PROJECT_STORAGE_ROOT=/mnt/disk2/gzt
  else
    PROJECT_STORAGE_ROOT="${_CACHE_REPO_ROOT}"
  fi
fi

export PROJECT_STORAGE_ROOT
export PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-${PROJECT_STORAGE_ROOT}/.cache}"
export PROJECT_TMP_ROOT="${PROJECT_TMP_ROOT:-${PROJECT_STORAGE_ROOT}/tmp}"

export XDG_CACHE_HOME="${PROJECT_CACHE_ROOT}"
export HF_HOME="${PROJECT_CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

export TORCH_HOME="${PROJECT_CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${PROJECT_CACHE_ROOT}/triton"
export VLLM_CACHE_ROOT="${PROJECT_CACHE_ROOT}/vllm"
export WANDB_CACHE_DIR="${PROJECT_CACHE_ROOT}/wandb"
export WANDB_DATA_DIR="${PROJECT_CACHE_ROOT}/wandb"
export WANDB_ARTIFACT_DIR="${PROJECT_CACHE_ROOT}/wandb/artifacts"
export MPLCONFIGDIR="${PROJECT_CACHE_ROOT}/matplotlib"
export NUMBA_CACHE_DIR="${PROJECT_CACHE_ROOT}/numba"
export PIP_CACHE_DIR="${PROJECT_CACHE_ROOT}/pip"

export TMPDIR="${PROJECT_TMP_ROOT}"
export TEMP="${PROJECT_TMP_ROOT}"
export TMP="${PROJECT_TMP_ROOT}"
export RAY_TMPDIR="${RAY_TMPDIR:-${_CACHE_REPO_ROOT}/.ray_tmp}"

mkdir -p \
  "${PROJECT_CACHE_ROOT}" "${PROJECT_TMP_ROOT}" \
  "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
  "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}" \
  "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${PIP_CACHE_DIR}" \
  "${TMPDIR}" "${RAY_TMPDIR}"

unset _CACHE_SCRIPT_DIR
unset _CACHE_REPO_ROOT
