#!/bin/bash
set -x

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

# Test Tritonbench operators
# TODO: test every operator, fwd+bwd
python run.py --op launch_latency --mode fwd --num-inputs 1 --test-only
python run.py --op addmm --mode fwd --num-inputs 1 --test-only
python run.py --op gemm --mode fwd --num-inputs 1 --test-only
python run.py --op sum --mode fwd --num-inputs 1 --test-only
python run.py --op softmax --mode fwd --num-inputs 1 --test-only
python run.py --op layer_norm --mode fwd --num-inputs 1 --test-only


# Segfault
# python run.py --op flash_attention --mode fwd --num-inputs 1 --test-only

# CUDA OOM
# python run.py --op jagged_layer_norm --mode fwd --num-inputs 1 --test-only
# python run.py --op jagged_mean --mode fwd --num-inputs 1 --test-only
# python run.py --op jagged_softmax --mode fwd --num-inputs 1 --test-only
# python run.py --op jagged_sum --mode fwd --num-inputs 1 --test-only
