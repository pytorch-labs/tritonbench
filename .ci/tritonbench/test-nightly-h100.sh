#!/bin/bash

# CI script to run Triton nightly on H100
# For now, only run launch-latency benchmark
set -x

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

# Run on Triton-pytorch
conda activate pytorch
python -m benchmarks.nightly.run

# Run on Triton-main
conda activate triton-main
python -m benchmarks.nightly.run
