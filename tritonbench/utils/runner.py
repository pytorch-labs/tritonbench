import argparse
import copy
import tempfile
import os
import sys
import subprocess

from tritonbench.operator_loader import load_opbench_by_name_from_loader
from tritonbench.operators import load_opbench_by_name
from tritonbench.utils.parser import remove_cmd_parameter, add_cmd_parameter
from tritonbench.utils.triton_op import BenchmarkOperatorResult, IS_FBCODE

from typing import List

TRITON_BENCH_CSV_DUMP_PATH = tempfile.gettempdir() + "/tritonbench/"

def tritonbench_run(args: argparse.Namespace, extra_args: List[str]) -> BenchmarkOperatorResult:
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
                    benchmark_name=args.op,
                    device=args.device,
                    hardware=args.hardware,
                    logging_op_name=args.logging_name,
                )
            else:
                log_benchmark(
                    metrics=metrics,
                    benchmark_name=args.op,
                    device=args.device,
                    logging_op_name=args.logging_name,
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

def run_in_task(op: str) -> None:
    if "--child" in sys.argv:
        sys.argv = remove_cmd_parameter(sys.argv, "--child")
        from tritonbench.utils.parser import get_parser
        parser = get_parser()
        args, extra_args = parser.parse_known_args(sys.argv[1:])
        tritonbench_run(args, extra_args)
        return
    op_task_cmd = [] if IS_FBCODE else [sys.executable]
    copy_sys_argv = copy.deepcopy(sys.argv)
    copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--op")
    copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--isolate")
    add_cmd_parameter(copy_sys_argv, "--op", op)
    add_cmd_parameter(copy_sys_argv, "--child")
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
