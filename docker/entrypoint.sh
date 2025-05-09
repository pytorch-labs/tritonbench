#!/bin/bash

# Entrypoint of the docker image

. "${SETUP_SCRIPT}"

cd /workspace/tritonbench

python run.py
