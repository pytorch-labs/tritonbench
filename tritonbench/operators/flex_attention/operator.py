import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import flex_attention

from tritonbench.utils.input import input_filter
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode as BenchmarkMode,
    register_benchmark,
    register_metric,
    register_x_val,
)

# Default configurations for the benchmark
BATCH_SIZE = 8
NUM_HEADS = 16
SEQ_LEN = 1024
HEAD_DIM = 128
DTYPE = torch.bfloat16


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.batch_size = BATCH_SIZE
        self.seq_len = SEQ_LEN
        self.head_dim = HEAD_DIM
        self.num_heads = NUM_HEADS

    def get_input_iter(self) -> Generator:
        """Generate a single input configuration for benchmarking."""

        # Create query, key, value tensors
        q = torch.rand(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
            device="cuda",
            dtype=DTYPE,
            requires_grad=True,
        )
        k = torch.rand(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
            device="cuda",
            dtype=DTYPE,
            requires_grad=True,
        )
        v = torch.rand(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
            device="cuda",
            dtype=DTYPE,
            requires_grad=True,
        )

        # Default kernel options for flex_attention
        kernel_options = {
            "num_warps": 8,
            # "ENABLE_TMA": True,
            # "TMA_SIZE": 128,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
        }

        yield (
            q,
            k,
            v,
            kernel_options,
        )

    @register_x_val(label="(Batch, Heads, SeqLen, Dhead)")
    def get_x_val(self, args) -> str:
        """Return a string representation of the input configuration."""
        q, k, v, *_ = args
        B, H, S, D = q.shape
        return (B, H, S, D)

    @register_benchmark(baseline=True)
    def eager(
        self,
        q,
        k,
        v,
        kernel_options,
    ) -> Callable:
        """Baseline implementation using eager mode."""
        return lambda: flex_attention(q, k, v)

    @register_benchmark()
    def compiled(
        self,
        q,
        k,
        v,
        kernel_options,
    ) -> Callable:
        """Compiled implementation using torch.compile."""
        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        return lambda: compiled_fn(q, k, v, kernel_options=kernel_options)

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        o = fwd_fn()
        o_tensor = input_filter(
            lambda x: isinstance(x, torch.Tensor),
            o,
        )
        do = torch.rand_like(o_tensor)
        return lambda: o_tensor.backward(do, retain_graph=True)

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        """Return tensors whose gradients should be set to None between iterations."""
        q, k, v, *_ = args
        return [q, k, v]

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Tuple, metrics: BenchmarkOperatorMetrics
    ) -> float:
        """Calculate the number of floating point operations for flex_attention."""
        q, *_ = example_inputs
        BATCH, H, N_CTX, D_HEAD = q.shape
        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
        flops = 2 * flops_per_matmul
        if self.mode == BenchmarkMode.BWD:
            flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        elif self.mode == BenchmarkMode.FWD_BWD:
            flops *= 3.5  # 1.0(fwd) + 2.0(bwd) + 0.5(recompute)
        return flops
