import copy
import logging
import sys
import time
import subprocess

from datetime import datetime
from pathlib import Path
from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.path_utils import REPO_PATH, remove_cmd_parameter, add_cmd_parameter

from typing import Optional, List

BENCHMARKS_OUTPUT_DIR = REPO_PATH.joinpath(".benchmarks")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_in_task(op: str, op_args: Optional[List[str]]=None) -> None:
    op_task_cmd = [] if is_fbcode() else [sys.executable]
    if not op_args:
        copy_sys_argv = copy.deepcopy(sys.argv)
        copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--op")
        copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--isolate")
        copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--op-collection")
        add_cmd_parameter(copy_sys_argv, "--op", op)
        op_task_cmd.extend(copy_sys_argv)
    else:
        op_task_cmd.extend(op_args)
    try:
        logger.info("[tritonbench] Running benchmark: " + " ".join(op_task_cmd))
        subprocess.check_call(op_task_cmd, stdout=sys.stdout, stderr=sys.stderr, cwd=REPO_PATH)
    except subprocess.CalledProcessError:
        # By default, we will continue on the failed operators
        pass
    except KeyboardInterrupt:
        logger.warning("[tritonbench] KeyboardInterrupt received, exiting...")
        sys.exit(1)


def setup_output_dir(bm_name: str):
    current_timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    output_dir = BENCHMARKS_OUTPUT_DIR.joinpath(bm_name, f"run-{current_timestamp}")
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    return output_dir.absolute()
