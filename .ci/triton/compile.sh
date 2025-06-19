#!/bin/bash

set -xeuo pipefail

# Print usage
usage() {
    echo "Usage: $0 --repo <repo-path> --commit <commit-hash> --side <a|b|single>"
    exit 1
}

# remove triton installations
remove_triton() {
    # delete the original triton directory
    TRITON_PKG_DIR=$(python -c "import triton; import os; print(os.path.dirname(triton.__file__))")
    # make sure all pytorch triton has been uninstalled
    pip uninstall -y triton
    pip uninstall -y triton
    pip uninstall -y triton
    rm -rf "${TRITON_PKG_DIR}"
}

checkout_triton() {
    REPO=$1
    COMMIT=$2
    TRITON_INSTALL_DIR=$3
    TRITON_INSTALL_DIRNAME=$(basename "${TRITON_INSTALL_DIR}")
    TRITON_INSTALL_BASEDIR=$(dirname "${TRITON_INSTALL_DIR}")
    cd "${TRITON_INSTALL_BASEDIR}"
    git clone "https://github.com/${REPO}.git" "${TRITON_INSTALL_DIRNAME}"
    cd "${TRITON_INSTALL_DIR}"
    git checkout "${COMMIT}"
}

install_triton() {
    TRITON_INSTALL_DIR=$1
    cd "${TRITON_INSTALL_DIR}"
    # install main triton
    pip install ninja cmake wheel pybind11; # build-time dependencies
    pip install -r python/requirements.txt
    pip install -e .
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --repo) REPO="$2"; shift ;;
        --commit) COMMIT="$2"; shift ;;
        --side) SIDE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# BASE_CONDA_ENV and SETUP_SCRIPT must be set
if [ -z "${BASE_CONDA_ENV}" ]; then
  echo "ERROR: BASE_CONDA_ENV is not set"
  exit 1
fi
if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

# Validate arguments
if [ -z "${REPO}" ] || [ -z "${COMMIT}" ] || [ -z "${SIDE}" ]; then
    echo "Missing required arguments."
    usage
fi

if [ "${SIDE}" == "single" ]; then
    TRITON_INSTALL_DIR=/workspace/triton
    if [ -z "${CONDA_ENV}" ]; then
        echo "Must specifify CONDA_ENV if running with --side single."
        exit 1
    fi
elif [ "${SIDE}" == "a" ] || [ "${SIDE}" == "b" ]; then
    mkdir -p /workspace/abtest
    CONDA_ENV="triton_side_${SIDE}"
    TRITON_INSTALL_DIR=/workspace/abtest/${CONDA_ENV}
else
    echo "Unknown side: ${SIDE}"
    exit 1
fi

# clone BASE_CONDA_ENV
CONDA_ENV=${BASE_CONDA_ENV} . "${SETUP_SCRIPT}"
conda activate "${BASE_CONDA_ENV}"
# Remove the conda env if exists
conda remove --name "${CONDA_ENV}" -y --all || true
conda create --name "${CONDA_ENV}" -y --clone "${BASE_CONDA_ENV}"
conda activate "${CONDA_ENV}"

. "${SETUP_SCRIPT}"

remove_triton

checkout_triton "${REPO}" "${COMMIT}" "${TRITON_INSTALL_DIR}"
install_triton "${TRITON_INSTALL_DIR}"

# export Triton repo related envs
# these envs will be used in nightly runs and other benchmarks
cd "${TRITON_INSTALL_DIR}"
TRITONBENCH_TRITON_COMMIT=$(git rev-parse --verify HEAD)
TRITONBENCH_TRITON_REPO=$(git config --get remote.origin.url | sed -E 's|.*github.com[:/](.+)\.git|\1|')
echo "export TRITONBENCH_TRITON_COMMIT=${TRITONBENCH_TRITON_COMMIT}" >> /workspace/setup_instance.sh
echo "export TRITONBENCH_TRITON_REPO=${TRITON_INSTALL_DIR}" >> /workspace/setup_instance.sh
