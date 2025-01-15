import copy
import sys
import subprocess

from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.path_utils import REPO_PATH, remove_cmd_parameter, add_cmd_parameter

from typing import Optional, List

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
        print("[tritonbench] running benchmark: " + " ".join(op_task_cmd))
        subprocess.check_call(op_task_cmd, stdout=sys.stdout, stderr=sys.stderr, cwd=REPO_PATH)
    except subprocess.CalledProcessError:
        # By default, we will continue on the failed operators
        pass
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, exiting...")
        sys.exit(1)
