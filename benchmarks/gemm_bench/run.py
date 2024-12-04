"""
Run gemm benchmark with a single input shape:

M = 4096
N = 4096
K = 4096

Print tflops metrics
"""

import tritonbench
from tritonbench.utils.parser import get_parser


def run():
    args = [
        "--m",
        "4096",
        "--n",
        "4096",
        "--k",
        "4096",
        "--precision",
        "fp16",
        "--only",
        "triton_tutorial_matmul",
        "--metrics",
        "tflops",
    ]
    gemm_op = tritonbench.load_opbench_by_name("gemm")
    parser = get_parser()
    args, extra_args = parser.parse_known_args(args)
    gemm_bench = gemm_op(args, extra_args)
    gemm_bench.run()
    print(gemm_bench.output)


if __name__ == "__main__":
    run()
