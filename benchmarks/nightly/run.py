"""
Tritonbench nightly run
"""
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


OPERATORS = [
    "launch_latency",
    "addmm",
    "gemm",
    "flash_attention",
]


def run():
    setup_tritonbench_cwd()
    from tritonbench.utils.runner import tritonbench_run_in_subprocess
    for op in OPERATORS:
        tritonbench_run_in_subprocess(op)
    # analyze the json files post run

if __name__ == "__main__":
    run()
