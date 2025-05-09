#!/bin/bash

# Entrypoint of the docker image

# shellcheck source=/workspace/setup_instance.sh
. "${SETUP_SCRIPT}"

cd /workspace/tritonbench || exit

python run.py
