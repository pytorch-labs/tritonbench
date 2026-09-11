#/usr/bin bash

set -xeuo pipefail

if [ -z "${WORKSPACE_DIR:-}" ]; then
    export WORKSPACE_DIR=/workspace
fi

if [ -z "${SETUP_SCRIPT:-}" ]; then
    export SETUP_SCRIPT=${WORKSPACE_DIR}/setup_instance.sh
fi

. "${SETUP_SCRIPT}"

export PYTORCH_FILE_PATH=$(python -c "import torch; print(torch.__file__)")

PYTORCH_DIR=$(dirname "${PYTORCH_FILE_PATH}")
NVIDIA_LIB_PATHS=(
    "$(realpath "${PYTORCH_DIR}/../nvidia/cu13/lib")"
    "$(realpath "${PYTORCH_DIR}/../nvidia/cudnn/lib")"
    "$(realpath "${PYTORCH_DIR}/../nvidia/cusparselt/lib")"
    "$(realpath "${PYTORCH_DIR}/../nvidia/nccl/lib")"
    "$(realpath "${PYTORCH_DIR}/../nvidia/nvshmem/lib")"
)

# Create libcublas.so / libcublasLt.so soft links when only versioned libraries
# (e.g. libcublas.so.13) are shipped, mirroring
# ../triton/.ci/tritonbench/setup-ld-library.sh.
CUBLAS_LIB_DIR="${PYTORCH_DIR}/../nvidia/cu13/lib"
if [ ! -f "${CUBLAS_LIB_DIR}/libcublas.so" ]; then
    for cublas_library in "${CUBLAS_LIB_DIR}"/libcublas.so.*; do
        if [ -f "${cublas_library}" ]; then
            ln -sf "${cublas_library}" "${CUBLAS_LIB_DIR}/libcublas.so"
            break
        fi
    done
    for cublas_library in "${CUBLAS_LIB_DIR}"/libcublasLt.so.*; do
        if [ -f "${cublas_library}" ]; then
            ln -sf "${cublas_library}" "${CUBLAS_LIB_DIR}/libcublasLt.so"
            break
        fi
    done
fi

if [ ! -f "${CUBLAS_LIB_DIR}/libcublas.so" ]; then
    echo "ERROR: Neither libcublas.so nor libcublas.so.* exists in ${CUBLAS_LIB_DIR}" >&2
    exit 1
fi

if [ ! -f "${CUBLAS_LIB_DIR}/libcublasLt.so" ]; then
    echo "ERROR: Neither libcublasLt.so nor libcublasLt.so.* exists in ${CUBLAS_LIB_DIR}" >&2
    exit 1
fi

for NVIDIA_LIB_PATH in "${NVIDIA_LIB_PATHS[@]}"; do
    if [ -e "${NVIDIA_LIB_PATH}" ]; then
        cat <<EOF >> "${SETUP_SCRIPT}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
    fi
done
