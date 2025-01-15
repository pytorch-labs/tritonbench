"""
Tritonbench nightly run
"""
import os
import logging
import sys
from os.path import abspath, exists
from benchmarks.utils import setup_output_dir
from tritonbench.utils.path_utils import get_cmd_parameter

logger = logging.getLogger(__name__)

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

OPERATOR_BENCHMARKS = {
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
    # TODO: reduce and aggregate all operator outputs
    pass


def run():
    setup_tritonbench_cwd()
    from tritonbench.operators.op_task import OpTask
    output_dir = setup_output_dir("nightly")
    # Run each operator
    output_files = []
    for op_bench in OPERATOR_BENCHMARKS:
        logger.info(f"[nightly] running operator benchmark: {op_bench}")
        op_name = get_cmd_parameter(OPERATOR_BENCHMARKS[op_bench], "--op")
        assert op_name, f"Could not find op name for benchmark {op_bench}: {OPERATOR_BENCHMARKS[op_bench]}"
        op_task = OpTask(op_name)
        op_args = OPERATOR_BENCHMARKS[op_bench]
        output_file = output_dir.joinpath(f"{op_bench}.csv")
        op_args.extend(["--output", str(output_file.absolute())])
        op_task.make_operator_instance(op_args)
        op_task.run()
        output_files.append(output_file)
        del op_task
    # Reduce all operator CSV outputs to a single output json
    result_json_file = reduce(output_files)
    logger.info(f"[nightly] logging result json file to {result_json_file}.")


if __name__ == "__main__":
    run()