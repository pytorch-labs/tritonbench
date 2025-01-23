"""
Run gemm benchmark with a single input shape:

Print tflops metrics
"""

M = 16384
N = M
K = M

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


def run():
    setup_tritonbench_cwd()
    args = [
        "--m",
        str(M),
        "--n",
        str(N),
        "--k",
        str(K),
        "--precision",
        "fp16",
        "--only",
        "triton_tutorial_matmul",
        "--metrics",
        "tflops",
    ]
    import tritonbench
    from tritonbench.utils.parser import get_parser
    gemm_op = tritonbench.load_opbench_by_name("gemm")
    parser = get_parser()
    args, extra_args = parser.parse_known_args(args)
    gemm_bench = gemm_op(args, extra_args)
    gemm_bench.run()
    print(gemm_bench.output)


if __name__ == "__main__":
    run()
