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
    from tritonbench.utils.parser import get_parser
    from tritonbench.utils.runner import run_in_task, tritonbench_run
    if "--op" in sys.argv:
        parser = get_parser()
        args, extra_args = parser.parse_args(sys.argv[2:])
        tritonbench_run(args, extra_args)
    for op in OPERATORS:
        run_in_task(op)


if __name__ == "__main__":
    run()
