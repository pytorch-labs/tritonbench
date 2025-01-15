import copy
import sys
import subprocess

from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.path_utils import remove_cmd_parameter, add_cmd_parameter

def run_in_task(op: str) -> None:
    op_task_cmd = [] if is_fbcode() else [sys.executable]
    copy_sys_argv = copy.deepcopy(sys.argv)
    copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--op")
    copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--isolate")
    copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--op-collection")
    add_cmd_parameter(copy_sys_argv, "--op", op)
    op_task_cmd.extend(copy_sys_argv)
    try:
        print("[tritonbench] running command: " + " ".join(op_task_cmd))
        subprocess.check_call(op_task_cmd, stdout=sys.stdout, stderr=sys.stderr)
    except subprocess.CalledProcessError:
        # By default, we will continue on the failed operators
        pass
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, exiting...")
        sys.exit(1)
