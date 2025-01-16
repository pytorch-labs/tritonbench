"""
Tritonbench nightly run
"""
import os
import json
import logging
import sys
from os.path import abspath, exists

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        "latency,gbps",
        "--num-inputs",
        "6",
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


def reduce(output_dir, output_files):
    """aggregate all op benchmark csvs into json file"""
    aggregated_obj = { "metrics": {} }
    for result_json_file in output_files:
        with open(result_json_file, "r",) as fp:
            result_obj = json.load(fp)
            aggregated_obj["metrics"].update(result_obj)
    result_json_path = os.path.join(output_dir, "result.json")
    with open(result_json_path, "w") as fp:
        json.dump(aggregated_obj, fp, indent=4)


def run():
    setup_tritonbench_cwd()
    from tritonbench.utils.run_utils import run_in_task, setup_output_dir
    output_dir = setup_output_dir("nightly")
    # Run each operator
    output_files = []
    for op_bench in OPERATOR_BENCHMARKS:
        logger.info(f"[nightly] running operator benchmark: {op_bench}")
        op_args = OPERATOR_BENCHMARKS[op_bench]
        output_file = output_dir.joinpath(f"{op_bench}.json")
        op_args.insert(0, "run.py")
        op_args.extend(["--output-json", str(output_file.absolute())])
        run_in_task(op=op_bench, op_args=op_args)
        output_files.append(output_file)
    # Reduce all operator CSV outputs to a single output json
    result_json_file = reduce(output_dir, output_files)
    logger.info(f"[nightly] logging result json file to {result_json_file}.")


if __name__ == "__main__":
    run()