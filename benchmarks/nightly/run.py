"""
Tritonbench nightly run, dashboard: https://hud.pytorch.org/tritonbench/commit_view
Requires the operator to support the speedup metric.
"""

import argparse
import json
import logging
import os
import sys
from os.path import abspath, exists
from pathlib import Path
from typing import Any, Dict

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def reduce(run_timestamp, output_dir, output_files, args):
    """aggregate all op benchmark csvs into json file"""
    from tritonbench.utils.gpu_utils import get_nvidia_gpu_states, has_nvidia_smi
    from tritonbench.utils.path_utils import REPO_PATH
    from tritonbench.utils.run_utils import get_github_env, get_run_env

    repo_locs = {
        "tritonbench": REPO_PATH,
    }
    if args.ci and "TRITONBENCH_TRITON_REPO_PATH" in os.environ:
        repo_locs["triton"] = os.environ.get("TRITONBENCH_TRITON_REPO_PATH", None)
        repo_locs["pytorch"] = os.environ.get("TRITONBENCH_PYTORCH_REPO_PATH", None)
    aggregated_obj = {
        "name": "nightly",
        "env": get_run_env(run_timestamp, repo_locs),
        "metrics": {},
    }
    if has_nvidia_smi():
        aggregated_obj.update(
            {
                "nvidia_gpu_states": get_nvidia_gpu_states(),
            }
        )

    # Collecting GitHub environment variables when running in CI environment
    if args.ci:
        aggregated_obj["github"] = get_github_env()

    for result_json_file in output_files:
        logger.info(f"Loading output file: {result_json_file}.")
        result_json_filename = Path(result_json_file).stem
        if (
            not os.path.exists(result_json_file)
            or os.path.getsize(result_json_file) == 0
        ):
            aggregated_obj["metrics"][f"tritonbench_{result_json_filename}-pass"] = 0
            continue
        # TODO: check if all inputs pass
        aggregated_obj["metrics"][f"tritonbench_{result_json_filename}-pass"] = 1
        with open(
            result_json_file,
            "r",
        ) as fp:
            result_obj = json.load(fp)
            aggregated_obj["metrics"].update(result_obj)
    result_json_path = os.path.join(output_dir, "result.json")
    with open(result_json_path, "w") as fp:
        json.dump(aggregated_obj, fp, indent=4)
    return result_json_path


def get_operator_benchmarks() -> Dict[str, Any]:
    def _load_benchmarks(config_path: str) -> Dict[str, Any]:
        out = {}
        with open(config_path, "r") as f:
            obj = yaml.safe_load(f)
        if not obj:
            return out
        for benchmark_name in obj:
            out[benchmark_name] = (
                obj[benchmark_name]["op"],
                obj[benchmark_name]["args"].split(" "),
            )
        return out

    out = _load_benchmarks(os.path.join(CURRENT_DIR, "manual.yaml"))
    out.update(_load_benchmarks(os.path.join(CURRENT_DIR, "autogen.yaml")))
    return out


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ci", action="store_true", help="Running in GitHub Actions CI mode."
    )
    args = parser.parse_args()
    setup_tritonbench_cwd()
    from tritonbench.utils.run_utils import run_in_task, setup_output_dir

    run_timestamp, output_dir = setup_output_dir("nightly")
    # Run each operator
    output_files = []
    operator_benchmarks = get_operator_benchmarks()
    for op_bench in operator_benchmarks:
        op_name, op_args = operator_benchmarks[op_bench]
        output_file = output_dir.joinpath(f"{op_bench}.json")
        op_args.extend(["--output-json", str(output_file.absolute())])
        run_in_task(op=op_name, op_args=op_args, benchmark_name=op_bench)
        output_files.append(output_file)
    # Reduce all operator CSV outputs to a single output json
    result_json_file = reduce(run_timestamp, output_dir, output_files, args)
    logger.info(f"[nightly] logging result json file to {result_json_file}.")


if __name__ == "__main__":
    run()
