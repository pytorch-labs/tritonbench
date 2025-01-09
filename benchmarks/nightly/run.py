"""
Tritonbench nightly run
"""
import os
import sys
from os.path import abspath, exists
from benchmarks.utils import setup_output_dir

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

OPERATORS = {
    "launch_latency": [],
    "addmm": [],
    "gemm": [],
    "flash_attention": [],
}


def reduce(output_dir):
    pass

def run():
    setup_tritonbench_cwd()
    from tritonbench.operators.op_task import OpTask
    output_dir = setup_output_dir()
    common_args = ["--output", output_dir]
    # Run each operator
    for op in OPERATORS:
        op_task = OpTask(op)
        op_args = OPERATORS[op]
        op_args.extend(common_args)
        op = op_task.make_operator_instance(op_args)
        op.run()
    # Reduce all operators to a single output json
    result_json_file = reduce(output_dir)


if __name__ == "__main__":
    run()