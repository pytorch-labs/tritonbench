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
from tritonbench.utils.parser import get_parser
from tritonbench.utils.gpu_utils import gpu_lockdown

from tritonbench.utils.triton_op import (
    BenchmarkOperatorResult,
    IS_FBCODE,
)

try:
    if IS_FBCODE:
        from .fb.run_utils import usage_report_logger
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
    if args.fwd_bwd:
        args.mode = "fwd_bwd"
    if args.bwd:
        args.mode = "bwd"
    if args.fwd_no_grad:
        args.mode = "fwd_no_grad"
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

            if "hardware" in args:
                log_benchmark(
                    metrics=metrics,
                    bencmark_name=args.op,
                    device=args.device,
                    hardware=args.hardware,
                )
            else:
                log_benchmark(
                    metrics=metrics, bencmark_name=args.op, device=args.device
                )
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
    if args.ci:
        from .ci import run_ci  # @manual

        run_ci()
        return

    if args.op:
        ops = args.op.split(",")
    else:
        ops = list_operators_by_collection(args.op_collection)

    with gpu_lockdown(args.gpu_lockdown):
        for op in ops:
            args.op = op
            _run(args, extra_args)


if __name__ == "__main__":
    run()
