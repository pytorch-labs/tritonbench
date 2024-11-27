"""
Tritonbench benchmark runner.

Note: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
      has been installed. This script intentionally does not automate or enforce setup steps.
"""
import sys
from typing import List

from tritonbench.operators_collection import list_operators_by_collection
from tritonbench.utils.gpu_utils import gpu_lockdown
from tritonbench.utils.parser import get_parser
from tritonbench.utils.runner import tritonbench_run_in_subprocess, tritonbench_run
from tritonbench.utils.triton_op import IS_FBCODE

try:
    if IS_FBCODE:
        from .fb.utils import usage_report_logger  # @manual
    else:
        usage_report_logger = lambda *args, **kwargs: None
except ImportError:
    usage_report_logger = lambda *args, **kwargs: None


def run(args: List[str] = []):
    if args == []:
        args = sys.argv[1:]
    # Log the tool usage
    usage_report_logger(benchmark_name="tritonbench")
    parser = get_parser()
    args, extra_args = parser.parse_known_args(args)

    if args.op:
        ops = args.op.split(",")
    else:
        ops = list_operators_by_collection(args.op_collection)

    # Force isolation in subprocess if testing more than one op.
    if len(ops) >= 2:
        args.isolate = True

    with gpu_lockdown(args.gpu_lockdown):
        for op in ops:
            args.op = op
            if args.isolate:
                tritonbench_run_in_subprocess(op)
            else:
                tritonbench_run(args, extra_args)


if __name__ == "__main__":
    run()
