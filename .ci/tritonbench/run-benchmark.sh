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

BENCHMARK_NAME=$1
shift

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --conda-env) CONDA_ENV="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

tritonbench_dir=$(dirname "$(readlink -f "$0")")/../..
cd "${tritonbench_dir}"

echo "Running ${BENCHMARK_NAME} benchmark under conda env ${CONDA_ENV}"

. "${SETUP_SCRIPT}"
python "benchmarks/${BENCHMARK_NAME}/run.py" --ci
