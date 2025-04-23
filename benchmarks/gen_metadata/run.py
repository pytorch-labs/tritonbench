# Scripts that load operators and generate the metadata
import logging
import argparse
import os
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
from os.path import abspath, exists
import sys

REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

setup_tritonbench_cwd()

from tritonbench.operators import list_operators, load_opbench_by_name

BACKWARD_OPERATORS = []
BASELINE_OPERATORS = []
TFLOPS_OPERATORS = []
DTYPE_OPERATORS = {}

METADATA_MAPPING = {
    "backward": (BACKWARD_OPERATORS, "backward_operators.yaml"),
    "baseline": (BASELINE_OPERATORS, "baseline_operators.yaml"),
    "tflops": (TFLOPS_OPERATORS, "tflops_operators.yaml"),
    "dtype": (DTYPE_OPERATORS, "dtype_operators.yaml"),
}

def run(args: argparse.Namespace):
    operators = list_operators()
    for op in operators:
        op_bench = load_opbench_by_name(op_name=op)
        DTYPE_OPERATORS[op] = op_bench.DEFAULT_PRECISION
        if op_bench.has_baseline():
            BASELINE_OPERATORS.append(op)
        if op_bench.has_tflops():
            TFLOPS_OPERATORS.append(op)
        if op_bench.has_bwd():
            BACKWARD_OPERATORS.append(op)
    for k in METADATA_MAPPING.keys():
        obj, fname = METADATA_MAPPING[k]
        output_file = os.path.join(args.output, fname)
        with open(output_file, "w") as out:
            yaml.safe_dump(obj, out, sort_keys=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(REPO_PATH, "tritonbench","metadata"),
        help="generate metadata yaml files to the specific directory"
    )
    args = parser.parse_args()
    run(args)
