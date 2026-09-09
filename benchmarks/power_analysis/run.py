"""
Perform power and performance analysis on a Triton kernel.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from ..common import setup_tritonbench_cwd

setup_tritonbench_cwd()

from tritonbench.utils.run_utils import load_operator_by_args, run_config

# Manifold bucket path (without the ``manifold://`` scheme) that receives the
# uploaded power-analysis output directories.
MANIFOLD_DEST = "tc_bench_ci/tree/power_analysis"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tritonbench-config",
        type=str,
        required=True,
        default=None,
        help="Path to a tritonbench config file (e.g. benchmarks/run_config/*.yaml). "
        "The config is rewritten with power-analysis flags appended to its "
        "common args and then run.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Number of A/B repeats. Fills the --ab-repeat value appended to "
        "the config's common args.",
    )
    return parser


def build_power_common_args(output_dir, repeat):
    """Power-analysis flags appended to the config's common args."""
    result_json = os.path.join(output_dir, "result.json")
    # NOTE: --side-a must use the bare `--side-a=` form, not `--side-a=""`.
    # Common args are split on spaces with no shell involved, so quotes would
    # survive literally: argparse would see the value '""', which parses to a
    # phantom empty-string arg that operators reject with
    # "run.py: error: unrecognized arguments: ".
    return (
        "--power-chart "
        f"--ab-repeat={repeat} "
        "--side-a= "
        f"--output-json {result_json} "
        f"--output-dir={output_dir}"
    )


def get_output_dir():
    """Tmp dir holding this run's rewritten config and outputs.

    Under MAST (``MAST_HPC_JOB_NAME`` set) the dir is named after the MAST
    job ID so runs are identifiable; otherwise a unique
    ``tritonbench_power_analysis_<XXXXX>`` dir is created.
    """
    mast_job_id = os.environ.get("MAST_HPC_JOB_NAME")
    if mast_job_id:
        output_dir = os.path.join(
            tempfile.gettempdir(),
            re.sub(r"[^A-Za-z0-9_.-]", "_", mast_job_id),
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    return tempfile.mkdtemp(prefix="tritonbench_power_analysis_")


def rewrite_config_with_power_args(config_path, output_dir, repeat):
    """Copy `config_path` into `output_dir` with power flags in common args.

    Returns the path of the rewritten config file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    common_args = (config.get("common_args") or "").strip()
    extra_args = build_power_common_args(output_dir, repeat)
    config["common_args"] = f"{common_args} {extra_args}".strip()
    rewritten_path = os.path.join(output_dir, os.path.basename(config_path))
    with open(rewritten_path, "w") as f:
        # A wide width keeps each arg string on a single line. The default
        # width (80) folds long scalars into continuation lines, turning them
        # into multi-line strings in the rewritten config.
        yaml.safe_dump(config, f, width=4096)
    return rewritten_path


def unset_nccl_envs():
    """Remove NCCL_* variables from the environment.

    Stale NCCL plugin/config vars (e.g. from the launcher environment) can
    break the benchmark subprocesses, which inherit os.environ. Returns the
    removed vars so the caller can restore them afterwards.
    """
    removed = {}
    for key in [key for key in os.environ if key.startswith("NCCL_")]:
        removed[key] = os.environ.pop(key)
    if removed:
        logger.info(f"Unset NCCL env vars: {sorted(removed)}")
    return removed


def upload_to_manifold(local_dir):
    """Recursively upload `local_dir` under MANIFOLD_DEST."""
    dest = f"{MANIFOLD_DEST}/{os.path.basename(local_dir)}"
    cmd = ["manifold", "putr", local_dir, dest]
    logger.info(f"Uploading {local_dir} to manifold://{dest}")
    subprocess.run(cmd, check=True)
    logger.info(f"Upload complete: manifold://{dest}")


def run_with_config(config_path, repeat):
    """Run power analysis for a tritonbench config file.

    Rewrites the config with power-analysis flags, runs tritonbench with it,
    then uploads the output dir to manifold. Returns the output dir.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Tritonbench config file not found: {config_path}")
    output_dir = get_output_dir()
    logger.info(f"Power analysis output dir: {output_dir}")
    rewritten_config = rewrite_config_with_power_args(config_path, output_dir, repeat)
    logger.info(f"Rewritten tritonbench config: {rewritten_config}")
    removed_nccl_envs = unset_nccl_envs()
    try:
        run_config(rewritten_config, [])
    finally:
        os.environ.update(removed_nccl_envs)
    upload_to_manifold(output_dir)
    return output_dir


def run(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    return run_with_config(args.tritonbench_config, args.repeat)


if __name__ == "__main__":
    run(sys.argv[1:])
