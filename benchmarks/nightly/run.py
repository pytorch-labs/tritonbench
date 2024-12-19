"""
Tritonbench nightly run
"""
import argparse
import os
import sys
from os.path import abspath, exists
import subprocess
from datetime import datetime
from pathlib import Path
import time

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

OPERATORS = {
    # Case 1: Launch latency
    "launch_latency": [
        "--op", "launch_latency",
        "--metrics", "walltime",
    ],
    # TODO: Add compile time
    # Case 2: GEMM TFLOPS
    "gemm": [
        "--op", "gemm",
        "--only", "triton_tutorial_matmul",
        "--metrics", "tflops"
    ],
    # TODO: Add compile time
    # Case 3: Flash Attention FWD_BWD TFLOPS
    "flash_attention": [
        "--op", "flash_attention",
        "--mode", "fwd_bwd",
        "--metrics", "tflops",
    ],
}


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

def setup_output_dir(create: bool=True) -> str:
    output_dir = os.path.join(CURRENT_DIR, ".data", 
        "run_{}_{}".format(os.environ["USER"], datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")))
    if create:
        Path(output_dir).mkdir(exist_ok=True, parents=True)
    return output_dir

def run_op(op: str, output_dir: str, continue_on_fail: bool=False) -> None:
    from tritonbench.utils.path_utils import REPO_PATH
    from tritonbench.utils.triton_op import IS_FBCODE
    assert op in OPERATORS, f"Operator {op} not in {OPERATORS.keys()}."
    op_task_cmd = [] if IS_FBCODE else [sys.executable, "run.py"]
    op_task_cmd.extend(OPERATORS[op])
    op_task_cmd.extend(["--output", os.path.join(output_dir, f"nightly_{op}.csv")])
    try:
        print("[tritonbench] running command: " + " ".join(op_task_cmd))
        subprocess.check_call(op_task_cmd, stdout=sys.stdout, stderr=sys.stderr, cwd=REPO_PATH)
    except subprocess.CalledProcessError:
        if continue_on_fail:
            pass
        else:
            raise
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, exiting...")
        sys.exit(1)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue-on-fail", action="store_true", help="Continue on failed operator.")
    parser.add_argument("--output-dir", default=None, help="Directory to save the results.")
    args = parser.parse_args()
    setup_tritonbench_cwd()
    if not args.output_dir:
        args.output_dir = setup_output_dir()
    for op in OPERATORS:
        run_op(op, args.output_dir, continue_on_fail=args.continue_on_fail)
    # analyze the json files post run
    from .postrun import postrun_analysis
    postrun_analysis(args.output_dir)

if __name__ == "__main__":
    run()
