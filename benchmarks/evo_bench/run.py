"""
Run benchmark and print tflops metrics
"""
import argparse
import os
import sys
from os.path import abspath, exists

def setup_tritonbench_cwd():
    original_dir = abspath(os.getcwd())

    for tritonbench_dir in (
        ".",
        "../../../tritonbench",
    ):
        if exists(tritonbench_dir):
            break

    if exists(tritonbench_dir):
        tritonbench_dir = abspath(tritonbench_dir)
        os.chdir(tritonbench_dir)
        sys.path.append(tritonbench_dir)
    return original_dir

evo_bench_config = {
    "flash_attention_fwd": [
        "--op",
        "flash_attention",
        "--num-inputs",
        "1",
        "--input-id",
        "4",
    ],
    "flash_attention_bwd": [
        "--op",
        "flash_attention",
        "--num-inputs",
        "1",
        "--input-id",
        "4",
        "--bwd",
    ],
    "jagged_layer_norm": [
        "--op",
        "jagged_layer_norm",
        "--num-inputs",
        "1",
        "--input-id",
        "0",
    ],
    "ragged_attention_fwd": [
        "--op",
        "ragged_attention",
        "--num-inputs",
        "1",
        "--input-id",
        "4",
        "--cudagraph",
    ],
}

def run():
    setup_tritonbench_cwd()
    from tritonbench.utils.run_utils import run_in_task
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=False)
    args = parser.parse_args()
    if args.op:
        ops = [args.op]
    else:
        ops = evo_bench_config.keys()
    for op_name in ops:
        op_args = evo_bench_config[op_name]
        run_in_task(op=op_name, op_args=op_args)

if __name__ == "__main__":
    run()
