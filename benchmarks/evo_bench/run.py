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
    "gemm": [
        "--op",
        "gemm",
        "--only",
        "triton_tutorial_matmul",
        "--m",
        "16384",
        "--n",
        "16384",
        "--k",
        "16384",
        "--cudagraph",
    ],
    "flash_attention_fwd": [
        "--op",
        "flash_attention",
        "--only",
        "triton_tutorial_flash_v2",
        "--num-inputs",
        "1",
        "--input-id",
        "6",
        "--metrics",
        "tflops",
        "--cudagraph",
    ],
    "flash_attention_bwd": [
        "--op",
        "flash_attention",
        "--only",
        "triton_tutorial_flash_v2",
        "--num-inputs",
        "1",
        "--input-id",
        "6",
        "--bwd",
        "--metrics",
        "tflops",
    ],
    "jagged_sum": [
        "--op",
        "jagged_sum",
        "--only",
        "triton_jagged_sum_no_pad_simple_fused",
        "--num-inputs",
        "1",
        "--input-id",
        "231",
        "--metrics",
        "gbps",
        "--test-only",
        "--cudagraph",
    ],
    "ragged_attention_fwd": [
        "--op",
        "ragged_attention",
        "--only",
        "hstu_triton_ragged_attention",
        "--num-inputs",
        "1",
        "--input-id",
        "4",
        "--metrics",
        "tflops",
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
