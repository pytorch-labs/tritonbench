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
    "launch_latency": [
        "--op",
        "launch_latency",
        "--metrics",
        "latency,walltime",
    ],
    "softmax": [
        "--op",
        "softmax",
        "--metrics",
        "latency,gbps,compile_time",
    ],
    "bf16_gemm": [
        "--op",
        "gemm",
        "--only",
        "aten_matmul,triton_tutorial_matmul",
        "--precision",
        "bf16",
        "--metrics",
        "latency,tflops",
        "--num-inputs",
        "4",
    ],
}


def reduce(output_files):
    # TODO: reduce and aggregate all outputs
    pass


def run():
    setup_tritonbench_cwd()
    from tritonbench.operators.op_task import OpTask
    output_dir = setup_output_dir("nightly")
    # Run each operator
    output_files = []
    for op in OPERATORS:
        op_task = OpTask(op)
        op_args = OPERATORS[op]
        output_file = output_dir.joinpath(f"{op}.csv")
        op_args.extend(["--output", str(output_file.absolute())])
        op = op_task.make_operator_instance(op_args)
        op.run()
        output_files.append(output_file)
    # Reduce all operator CSV outputs to a single output json
    result_json_file = reduce(output_files)


if __name__ == "__main__":
    run()