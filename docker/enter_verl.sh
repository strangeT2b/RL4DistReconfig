#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-docker.1ms.run/verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2}"
HOST_ROOT="${HOST_ROOT:-/mnt/disk2/gzt}"
PROJECT_DIR="${PROJECT_DIR:-${HOST_ROOT}/RL4DistReconfig}"
VERL_DIR="${VERL_DIR:-${HOST_ROOT}/verl}"
GPUS="${GPUS:-all}"
WORKDIR="${WORKDIR:-/workspace/RL4DistReconfig}"
CONTAINER_NAME="${CONTAINER_NAME:-verl_gzt}"
if [[ "${GPUS}" == "all" ]]; then
  GPU_REQUEST="all"
else
  GPU_REQUEST="\"device=${GPUS}\""
fi

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "Project directory not found: ${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${VERL_DIR}" ]]; then
  cat >&2 <<EOF
veRL source directory not found: ${VERL_DIR}
Clone it first, for example:
  cd ${HOST_ROOT}
  git clone https://github.com/volcengine/verl.git
  cd verl
  git checkout v0.4.1
EOF
  exit 1
fi

exec docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --gpus "${GPU_REQUEST}" \
  --ipc=host \
  --network=host \
  --shm-size=10g \
  --cap-add=SYS_ADMIN \
  -v "${VERL_DIR}:/workspace/verl" \
  -v "${PROJECT_DIR}:/workspace/RL4DistReconfig" \
  -v "${HOST_ROOT}:${HOST_ROOT}" \
  -w "${WORKDIR}" \
  "${IMAGE}" \
  bash
