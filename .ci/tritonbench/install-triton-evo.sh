#!/bin/bash

if [ -z "${BASE_CONDA_ENV}" ]; then
  echo "ERROR: BASE_CONDA_ENV is not set"
  exit 1
fi

if [ -z "${CONDA_ENV}" ]; then
  echo "ERROR: CONDA_ENV is not set"
  exit 1
fi

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

CONDA_ENV=${BASE_CONDA_ENV} . "${SETUP_SCRIPT}"
conda activate "${BASE_CONDA_ENV}"
# Remove the conda env if exists
conda remove --name "${CONDA_ENV}" -y --all || true
conda create --name "${CONDA_ENV}" -y --clone "${BASE_CONDA_ENV}"
conda activate "${CONDA_ENV}"

. "${SETUP_SCRIPT}"

# Install and build triton from xuzhao9/triton
cd /workspace
git clone https://github.com/xuzhao9/triton.git triton-evo
cd /workspace/triton-evo
git checkout -t origin/evo

# delete the original triton directory
TRITON_PKG_DIR=$(python -c "import triton; import os; print(os.path.dirname(triton.__file__))")
# make sure all pytorch triton has been uninstalled
pip uninstall -y triton
pip uninstall -y triton
pip uninstall -y triton
rm -rf "${TRITON_PKG_DIR}"

# install triton-evo branch
pip install ninja cmake wheel pybind11; # build-time dependencies
pip install -r python/requirements.txt
pip install -e .

# setup Triton repo related envs
# these envs will be used in nightly runs and other benchmarks
TRITONBENCH_TRITON_EVO_COMMIT=$(git rev-parse --verify HEAD)
echo "export TRITONBENCH_TRITON_EVO_COMMIT=${TRITONBENCH_TRITON_EVO_COMMIT}" >> /workspace/setup_instance.sh
echo "export TRITONBENCH_TRITON_EVO_REPO_PATH=/workspace/triton-evo" >> /workspace/setup_instance.sh
