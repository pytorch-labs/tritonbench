#!/bin/bash
set -x

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

if [ -z "$1" ]; then
  echo "ERROR: BENCHMARK_NAME must be set as the first argument."
  exit 1
fi

. "${SETUP_SCRIPT}"

BENCHMARK_NAME=$1

tritonbench_dir=$(dirname "$(readlink -f "$0")")/../..
cd ${tritonbench_dir}

python "benchmarks/${BENCHMARK_NAME}/run.py" --ci
