#!/bin/bash

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

tritonbench_dir=$(dirname "$(readlink -f "$0")")/../..
cd ${tritonbench_dir}

# probe memory available
free -h

# Install Tritonbench and all its customized packages
# Test: only install fa3
python install.py --fa3
