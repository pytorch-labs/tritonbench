import argparse

from tritonbench.utils.triton_op import (
    DEFAULT_RUN_ITERS,
    DEFAULT_WARMUP,
    IS_FBCODE,
)
from tritonbench.utils.env_utils import AVAILABLE_PRECISIONS

def get_parser(args=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--op",
        type=str,
        required=False,
        help="Operators to benchmark. Split with comma if multiple.",
    )
    parser.add_argument(
        "--op-collection",
        default="default",
        type=str,
        help="Operator collections to benchmark. Split with comma."
        " It is conflict with --op. Choices: [default, liger, all]",
    )
    parser.add_argument(
        "--mode",
        choices=["fwd", "bwd", "fwd_bwd", "fwd_no_grad"],
        default="fwd",
        help="Test mode (fwd, bwd, fwd_bwd, or fwd_no_grad).",
    )
    parser.add_argument("--bwd", action="store_true", help="Run backward pass.")
    parser.add_argument(
        "--fwd-bwd",
        action="store_true",
        help="Run both forward and backward pass.",
    )
    parser.add_argument(
        "--fwd-no-grad", action="store_true", help="Run forward pass without grad."
    )
    parser.add_argument(
        "--precision",
        "--dtype",
        choices=AVAILABLE_PRECISIONS,
        default="bypass",
        help="Specify operator input dtype/precision. Default to `bypass` - using DEFAULT_PRECISION defined in the operator.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        default=DEFAULT_WARMUP,
        help="Num of warmup runs for reach benchmark run.",
    )
    parser.add_argument(
        "--iter",
        default=DEFAULT_RUN_ITERS,
        help="Num of reps for each benchmark run.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print result as csv.",
    )
    parser.add_argument(
        "--dump-csv",
        action="store_true",
        help="Dump result as csv.",
    )
    parser.add_argument(
        "--skip-print",
        action="store_true",
        help="Skip printing result.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the result.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run in the CI mode.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Metrics to collect, split with comma. E.g., --metrics latency,tflops,speedup.",
    )
    parser.add_argument(
        "--metrics-gpu-backend",
        choices=["torch", "nvml"],
        default="torch",
        help=(
            "Specify the backend [torch, nvml] to collect metrics. In all modes, the latency "
            "(execution time) is always collected using `time.time_ns()`. The CPU peak memory "
            "usage is collected by `psutil.Process()`. In nvml mode, the GPU peak memory usage "
            "is collected by the `nvml` library. In torch mode, the GPU peak memory usage is "
            "collected by `torch.cuda.max_memory_allocated()`."
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Specify one or multiple operator implementations to run.",
    )
    parser.add_argument(
        "--baseline", type=str, default=None, help="Override default baseline."
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        help="Number of example inputs.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
    )
    parser.add_argument(
        "--input-id",
        type=int,
        default=0,
        help="Specify the start input id to run. "
        "For example, --input-id 0 runs only the first available input sample."
        "When used together like --input-id <X> --num-inputs <Y>, start from the input id <X> "
        "and run <Y> different inputs.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run this under test mode, potentially skipping expensive steps like autotuning.",
    )
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="Dump Triton IR",
    )
    parser.add_argument(
        "--gpu-lockdown",
        action="store_true",
        help="Lock down GPU frequency and clocks to avoid throttling.",
    )
    parser.add_argument(
        "--operator-loader",
        action="store_true",
        help="Benchmarking aten ops in tritonbench/operator_loader.",
    )

    if IS_FBCODE:
        parser.add_argument("--log-scuba", action="store_true", help="Log to scuba.")

    args, extra_args = parser.parse_known_args(args)
    if args.op and args.ci:
        parser.error("cannot specify operator when in CI mode")
    if not args.op and not args.op_collection:
        print(
            "Neither operator nor operator collection is specified. Running all operators in the default collection."
        )
    return parser