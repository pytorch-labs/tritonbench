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

# Install and build triton from source code
cd /workspace
git clone https://github.com/triton-lang/triton.git
cd /workspace/triton
pip install ninja cmake wheel pybind11; # build-time dependencies
pip install -e python
