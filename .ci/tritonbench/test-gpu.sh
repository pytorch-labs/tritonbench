#!/bin/bash
set -x

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

# FIXME: patch and install hstu
sudo apt-get install  -y patch
python install.py --hstu

# FIXME: install colfax
python install.py --colfax

python -m unittest test.test_gpu.main -v
