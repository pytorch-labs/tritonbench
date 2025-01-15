import copy
import logging
import sys
import subprocess

from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.path_utils import REPO_PATH, remove_cmd_parameter, add_cmd_parameter

from typing import Optional, List

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
