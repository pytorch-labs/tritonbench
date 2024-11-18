#!/bin/bash
set -x

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

# install deps
pip install psutil tabulate

# FIXME: patch hstu
python install.py --hstu

python -m unittest test.test_gpu.main -v
