"""
Tritonbench benchmark runner.

Note: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
      has been installed. This script intentionally does not automate or enforce setup steps.
"""

import argparse
import os
import sys
import tempfile
from typing import List

from tritonbench.operator_loader import load_opbench_by_name_from_loader
from tritonbench.operators import load_opbench_by_name
from tritonbench.operators_collection import list_operators_by_collection
from tritonbench.utils.gpu_utils import gpu_lockdown
from tritonbench.utils.parser import get_parser
from tritonbench.utils.path_utils import add_cmd_parameter, remove_cmd_parameter
from tritonbench.utils.runner import run_in_task
from tritonbench.utils.triton_op import BenchmarkOperatorResult, IS_FBCODE

try:
    if IS_FBCODE:
        from .fb.utils import usage_report_logger  # @manual
    else:
        usage_report_logger = lambda *args, **kwargs: None
except ImportError:
    usage_report_logger = lambda *args, **kwargs: None

TRITON_BENCH_CSV_DUMP_PATH = tempfile.gettempdir() + "/tritonbench/"


def _run(args: argparse.Namespace, extra_args: List[str]) -> BenchmarkOperatorResult:
    if args.operator_loader:
        Opbench = load_opbench_by_name_from_loader(args)
    else:
        Opbench = load_opbench_by_name(args.op)
    opbench = Opbench(
        tb_args=args,
        extra_args=extra_args,
    )
    try:
        opbench.run(args.warmup, args.iter)
    finally:
        metrics = opbench.output
        if not args.skip_print:
            if args.csv:
                metrics.write_csv_to_file(sys.stdout)
            else:
                print(metrics)
        if IS_FBCODE and args.log_scuba:
            from .fb.utils import log_benchmark  # @manual

            kwargs = {
                "metrics": metrics,
                "benchmark_name": args.op,
                "device": args.device,
                "logging_group": args.logging_group,
            }
            if args.production_shapes:
                from tritonbench.utils.fb.durin_data import productionDataLoader

                kwargs["weights_loader"] = productionDataLoader

            if "hardware" in args:
                kwargs["hardware"] = args.hardware
            log_benchmark(**kwargs)

        if args.plot:
            try:
                opbench.plot()
            except NotImplementedError:
                print(f"Plotting is not implemented for {args.op}")

        if args.dump_csv:
            os.makedirs(TRITON_BENCH_CSV_DUMP_PATH, exist_ok=True)
            path = metrics.write_csv(TRITON_BENCH_CSV_DUMP_PATH)
            print(f"[TritonBench] Dumped csv to {path}")
        return metrics


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

    with gpu_lockdown(args.gpu_lockdown):
        for op in ops:
            args.op = op
            if args.isolate:
                run_in_task(op)
            else:
                _run(args, extra_args)


if __name__ == "__main__":
    run()
