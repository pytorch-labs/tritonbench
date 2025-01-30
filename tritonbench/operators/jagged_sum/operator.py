import argparse
import itertools
import math
import os
import random
from typing import Callable, Generator, List, Optional, Tuple

import sys
from io import StringIO
import inspect
import torch
import triton

from .cache_utils import jagged_cache
# from triton.runtime import Autotuner
from triton.testing import do_bench

from tritonbench.utils.jagged_utils import (
    ABSOLUTE_TOLERANCE,
    generate_input_vals,
    generate_random_nested_tensors,
    get_param_fstrings,
    get_parse_op_args,
    get_plot_args,
    get_styles,
    get_tensor_bytes_limit,
    GIGABYTES_PER_BYTE,
    RANDOM_CHOICE_MARGIN,
    RELATIVE_TOLERANCE,
)

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .kernels import (
    triton_jagged_sum_kernel_simple_fused_buffer_then_sum,
    triton_jagged_sum_kernel_simple_fused_sum_then_buffer,
    triton_jagged_sum_kernel_variable_length_loop_buffer_then_sum,
    triton_jagged_sum_kernel_variable_length_loop_sum_then_buffer,
    triton_jagged_sum_kernel_simple_fused_sum_then_buffer_no_autotune,
    triton_jagged_sum_kernel_simple_fused_sum_then_buffer_new,
    triton_jagged_sum_kernel_variable_length_loop_sum_then_buffer_no_autotune,
)

    
def parse_op_args(args: List[str]):
    parser = get_parse_op_args(
        "B", "M", "seqlen", "sparsity", "sum_then_buffer", "plot_benchmarks"
    )
    return parser.parse_args(args)


# def get_best_config_from_output(output_str):
#     # Split multiple kernel outputs
#     for line in output_str.split('\n'):
#         if "best config selected:" in line:
#             config_str = line.split("best config selected:")[1].strip(";").strip()
#             config_dict = {}
#             for pair in config_str.split(","):
#                 key, value = pair.split(":")
#                 key = key.strip()
#                 value = value.strip()
#                 # Handle None value specifically
#                 if value == "None":
#                     config_dict[key] = None
#                 else:
#                     try:
#                         config_dict[key] = int(value)
#                     except ValueError:
#                         config_dict[key] = value
#             return config_dict
#     return None

def autotune_jagged_sum(kernel_fn, x, kernel_output, M, max_seqlen, grid_fn):
    # Setup configurations
    BLOCK_SIZES = [2**n for n in range(3, 7, 3)]  # [8, 64]
    NUM_WARPS = [4, 8]
    NUM_STAGES = [2, 4]
    UNROLL_FACTORS = [2, 4, 8, 16]
    
    best_ms = float('inf')
    best_config = None
    debug = True  # Set to True to see debugging info

    grid_fn = lambda meta: ((len(x.offsets()) - 1) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    # Try all configurations
    for b_r, b_m, w, s, u in itertools.product(BLOCK_SIZES, BLOCK_SIZES, NUM_WARPS, NUM_STAGES, UNROLL_FACTORS):
        config = {
            "BLOCK_SIZE_RAGGED": b_r,
            "BLOCK_SIZE_M": b_m,
            "num_warps": w,
            "num_stages": s,
            "unroll_factor": u,
        }
        
        try:
            def bench_fn():
                grid = grid_fn({"BLOCK_SIZE_M": b_m})
                return kernel_fn[grid](
                    x.values(),
                    x.offsets(),
                    kernel_output,
                    M=M,
                    MAX_SEQLEN=max_seqlen,
                    BLOCK_SIZE_RAGGED=b_r,
                    BLOCK_SIZE_M=b_m,
                    num_warps=w,
                    num_stages=s,
                    unroll_factor=u,
                )

            # Warmup run
            bench_fn()
            
            # Actual benchmark
            ms = triton.testing.do_bench(bench_fn, warmup=1, rep=1)
            
            # if debug:
            #     print(f"Config: BLOCK_SIZE_RAGGED={b_r}, BLOCK_SIZE_M={b_m}, "
            #           f"num_warps={w}, num_stages={s}, ms={ms}")
            
            if ms < best_ms:
                best_ms = ms
                best_config = config

        except triton.runtime.OutOfResources:
            if debug:
                print(f'Config failed (out of resources): BLOCK_SIZE_RAGGED={b_r}, '
                      f'BLOCK_SIZE_M={b_m}, num_warps={w}, num_stages={s}')
            continue

    # if debug and best_config:
    #     print("\nBest configuration found:")
    #     for key, value in best_config.items():
    #         print(f"{key}: {value}")
    #     print(f"Best latency: {best_ms:.3f} ms")

    return best_ms, best_config, kernel_fn

def execute_kernel_simple_fused(x, max_seqlen, sum_then_buffer):
    # print(x)
    B, M = x.shape[0], x.shape[2]
    seq_lengths = [x.offsets()[i+1].item() - x.offsets()[i].item() for i in range(len(x.offsets())-1)]
    # print(f"B {B}, M ={M}")
    # print(f"Length of x.offsets() {len(x.offsets())}")
    cached_config = jagged_cache.get_config(B, M, seq_lengths, max_seqlen)
    grid = lambda meta: ((len(x.offsets()) - 1) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    kernel_output = torch.zeros((B, M), device=x.device)

    # lengths of each sequence
    # lengths = x.offsets()
    # # print(lengths)
    seq_lengths = [x.offsets()[i+1].item() - x.offsets()[i].item() for i in range(len(x.offsets())-1)]
    # print(seq_lengths)

    if cached_config:
        config = cached_config['config']
        # print("Using cached config:", config)
        if sum_then_buffer:
            triton_jagged_sum_kernel_simple_fused_sum_then_buffer_new[grid](
                x.values(),
                x.offsets(),
                kernel_output,
                M=M,
                MAX_SEQLEN=max_seqlen,
                BLOCK_SIZE_RAGGED=config['BLOCK_SIZE_RAGGED'],
                BLOCK_SIZE_M=config['BLOCK_SIZE_M'],
                num_warps=config['num_warps'],
                num_stages=config['num_stages'],
                unroll_factor=config['unroll_factor'],
            )
        else:
            triton_jagged_sum_kernel_simple_fused_buffer_then_sum[grid](
                x.values(),
                x.offsets(),
                kernel_output,
                M=M,
                MAX_SEQLEN=max_seqlen,
                BLOCK_SIZE_RAGGED=config['BLOCK_SIZE_RAGGED'],
                BLOCK_SIZE_M=config['BLOCK_SIZE_M'],
                num_warps=config['num_warps'],
                num_stages=config['num_stages']
        )
    else:
        # print("No cached config found for M:", M)
        if sum_then_buffer:
            best_ms, best_config, kernel = autotune_jagged_sum(
                triton_jagged_sum_kernel_simple_fused_sum_then_buffer_new,
                x, kernel_output, M, max_seqlen, grid
            )

            # Use the best configuration
            if best_config:
                grid = grid({"BLOCK_SIZE_M": best_config["BLOCK_SIZE_M"]})
                kernel[grid](
                    x.values(),
                    x.offsets(),
                    kernel_output,
                    M=M,
                    MAX_SEQLEN=max_seqlen,
                    **best_config
                )
            jagged_cache.store_config(B, M, seq_lengths, max_seqlen, best_config, best_ms)

        else:
            triton_jagged_sum_kernel_simple_fused_buffer_then_sum[grid](
                x.values(),
                x.offsets(),
                kernel_output,
                M=M,
                MAX_SEQLEN=max_seqlen,
            )
        # sys.stdout = old_stdout
        # output = mystdout.getvalue()
        
        # config = get_best_config_from_output(output)
        # if config:
        #     print("Parsed config:", config)
        #     jagged_cache.store_config(M, config)
    return kernel_output


def execute_kernel_variable_length_loop(x, sum_then_buffer):
    B, M = x.shape[0], x.shape[2]
    # cached_config = jagged_cache.get_config(M)
    grid = lambda meta: ((len(x.offsets()) - 1) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    kernel_output = torch.zeros((B, M), device=x.device)

    # if cached_config:
    #     config = cached_config['config']
    #     if sum_then_buffer:
    #         triton_jagged_sum_kernel_variable_length_loop_sum_then_buffer_no_autotune[grid](
    #             x.values(),
    #             x.offsets(),
    #             kernel_output,
    #             M=M,
    #             BLOCK_SIZE_RAGGED=config['BLOCK_SIZE_RAGGED'],
    #             BLOCK_SIZE_M=config['BLOCK_SIZE_M'],
    #             num_warps=config['num_warps'],
    #             num_stages=config['num_stages']
    #         )
    #     else:
    #         triton_jagged_sum_kernel_variable_length_loop_buffer_then_sum[grid](
    #             x.values(),
    #             x.offsets(),
    #             kernel_output,
    #             M=M,
    #             BLOCK_SIZE_RAGGED=config['BLOCK_SIZE_RAGGED'],
    #             BLOCK_SIZE_M=config['BLOCK_SIZE_M'],
    #             num_warps=config['num_warps'],
    #             num_stages=config['num_stages']
    #         )
    # else:
    if sum_then_buffer:
        triton_jagged_sum_kernel_variable_length_loop_sum_then_buffer[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
        )
    else:
        triton_jagged_sum_kernel_variable_length_loop_buffer_then_sum[grid](
            x.values(),
            x.offsets(),
            kernel_output,
            M=M,
        )

    return kernel_output


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "accuracy", "best_config"]
    DEFAULT_PRECISION = "fp32"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        # sizes = [2, 6, 10, 12, 15, 18, 21]
        self.sizes = (
            list(range(2, 12, 4)) + list(range(12, 23, 3))
        )  # bias towards larger sizes, which are more representative of real-world shapes

        args = parse_op_args(self.extra_args)
        self.B = args.B
        self.M = args.M
        self.seqlen = args.seqlen
        self.sparsity = args.sparsity
        self.sum_then_buffer = args.sum_then_buffer
        self.plot_benchmarks = args.plot_benchmarks

        self.tensor_bytes_limit = get_tensor_bytes_limit(tb_args.test_only)

    @register_benchmark(baseline=True)
    def torch_jagged_sum_no_pad(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        return lambda: torch.tensor(
            [
                torch.sum(t, dim=0).tolist() for t in x.unbind()
            ],  # in 3D tensor (B, *, M), sums B 2D tensors (*, M)
            device=self.device,
            dtype=self.dtype,
        )

    @register_benchmark()
    def torch_jagged_sum_pad(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        return (
            lambda: torch.sum(
                torch.ops.aten._jagged_to_padded_dense_forward(
                    x.values(),
                    [
                        x.offsets()
                    ],  # pyre-ignore: Undefined attribute [16]: `torch._tensor.Tensor` has no attribute `offsets`.
                    max_lengths=[seqlen],  # max length of ragged dimension
                ),
                dim=1,
            )
        )  # sum along ragged dimension (dim == 1)

    @register_benchmark()
    def triton_jagged_sum_no_pad_simple_fused(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner():
            return execute_kernel_simple_fused(x, seqlen, self.sum_then_buffer)

        return _inner

    @register_benchmark()
    def triton_jagged_sum_no_pad_variable_length_loop(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner():
            return execute_kernel_variable_length_loop(x, self.sum_then_buffer)

        return _inner

    @register_benchmark()
    def torch_compile_nested_tensor_integration(
        self, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
    ):
        def _inner(x: torch.Tensor):  # sum along ragged dimension (dim == 1)
            return torch.sum(
                x, dim=x._ragged_idx
            )  # pyre-ignore: Undefined attribute [16]: `torch._tensor.Tensor` has no attribute `_ragged_idx`.

        torch_compile_func = torch.compile(_inner)
        return lambda: torch_compile_func(x)

    def get_x_val(self, example_inputs):
        if self.B is None:
            return example_inputs[1]
        if self.M is None:
            return example_inputs[2]
        if self.seqlen is None:
            return example_inputs[3]
        if self.sparsity is None:
            return example_inputs[4]

    def get_x_vals(self) -> Tuple[List[int], List[int], List[int], List[float]]:
        return generate_input_vals(
            self.B, self.M, self.seqlen, self.sparsity, self.sizes
        )

    def get_input_iter(self) -> Generator:
        """
        Generate random nested tensors of shape (B, *, M), where * is the ragged dimension
        """

        B_vals, M_vals, seqlen_vals, sparsity_vals = self.get_x_vals()

        for nt, B, M, max_seqlen, sparsity in generate_random_nested_tensors(
            B_vals,
            M_vals,
            seqlen_vals,
            sparsity_vals,
            device=self.device,
            dtype=self.dtype,
            TENSOR_BYTES_LIMIT=self.tensor_bytes_limit,
            RANDOM_CHOICE_MARGIN=RANDOM_CHOICE_MARGIN,
        ):
            yield (nt, B, M, max_seqlen, sparsity)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        return torch.allclose(
            output, baseline_output, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
        )

    @register_metric()
    def gbps(self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics):
        return (
            example_inputs[0].element_size()
            * example_inputs[0].numel()
            / metrics.latency
            * GIGABYTES_PER_BYTE
        )

    @register_metric(x_only=True)
    def input_shape(
        self, fn_name: str, example_inputs, metrics: BenchmarkOperatorMetrics
    ):
        return (
            f"B: {example_inputs[1]}",  # B
            "*",
            f"M: {example_inputs[2]}",  # M
            f"max seqlen: {example_inputs[3]}",  # seqlen
            f"sparsity: {example_inputs[4]}",  # sparsity
        )  # return (B, '*', M, max seqlen, sparsity) for each example input

    def plot(self):
        x_axis, params = get_param_fstrings(self.B, self.M, self.seqlen, self.sparsity)

        line_vals_all = [
            "torch_jagged_sum_no_pad",
            "torch_jagged_sum_pad",
            "triton_jagged_sum_no_pad_simple_fused",
            "triton_jagged_sum_no_pad_variable_length_loop",
            "torch_compile_nested_tensor_integration",
        ]
        line_names_all = [
            "PyTorch jagged sum, no padding",
            "PyTorch jagged sum, padding",
            "Triton kernel jagged sum, simple fused",
            "Triton kernel jagged sum, variable length loop",
            "Inductor, NestedTensor integration",
        ]
        styles_all = get_styles(len(line_vals_all))

        line_vals, line_names, styles = get_plot_args(
            self.plot_benchmarks, 2, line_vals_all, line_names_all, styles_all
        )

        plot_name = f"jagged-sum-perf-var-{x_axis}" + params

        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["x_axis"],
                x_vals=self.output.x_vals,
                line_arg="provider",
                line_vals=line_vals,
                line_names=line_names,
                styles=styles,
                xlabel=x_axis,
                ylabel="latency",
                plot_name=plot_name,
                args={},
            )
        )
        def _plot(x_axis, provider):
            return self.output.get_y_vals(x_axis, provider, "latency")

        save_path = (
            # os.getcwd()
            # + f"/pytorch/tritonbench/tritonbench/operators/jagged_sum/jagged_sum_performance/{plot_name}"
            os.getcwd()
            + f"{plot_name}"
        )

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
