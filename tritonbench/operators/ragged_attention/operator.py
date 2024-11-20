import argparse

from typing import Any, Callable, List, Optional

import torch
from tritonbench.utils.input import input_filter

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
)

from .hstu import get_test_inputs, RaggedHSTUAttn


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--max-seq-len-log2", type=int, default=9)
    parser.add_argument("--num-buckets", type=int, default=2048)
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--target-size", type=int, default=20)
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args=extra_args)
        args = parse_op_args(self.extra_args)
        self.batch_size = args.batch_size
        self.num_heads = args.heads
        self.max_seq_len = 2**args.max_seq_len_log2
        self.num_buckets = args.num_buckets
        self.sparsity = args.sparsity
        self.target_size = args.target_size
        # set a default number of inputs
        self._num_inputs = 10 if self._num_inputs is None else self._num_inputs
        self.requires_grad = not (self.mode == Mode.FWD_NO_GRAD)

    @register_benchmark()
    def hstu_triton_ragged_attention(self, qkv, seq_offsets, timestamps, num_targets):
        attn = RaggedHSTUAttn(
            self.batch_size,
            self.num_heads,
            self.max_seq_len,
            self.num_buckets,
            self.sparsity,
            self.target_size,
            self.requires_grad,
            persistent_kernel=False,
        )
        return lambda: attn(qkv, seq_offsets, timestamps, num_targets)

    # TODO: enable persistent kernels when the OSS backward is ready
    @register_benchmark(enabled=False)
    def hstu_triton_ragged_attention_persistent(self, qkv, seq_offsets, timestamps, num_targets):
        attn = RaggedHSTUAttn(
            self.batch_size,
            self.num_heads,
            self.max_seq_len,
            self.num_buckets,
            self.sparsity,
            self.target_size,
            self.requires_grad,
            persistent_kernel=True,
        )
        return lambda: attn(qkv, seq_offsets, timestamps, num_targets)

    def get_x_val(self, example_inputs):
        return (self.batch_size, self.num_heads, self.max_seq_len, self.num_buckets, self.sparsity, self.target_size)

    def get_input_iter(self):
        for _input_id in range(self._num_inputs):
            inputs = get_test_inputs(
                self.batch_size, self.num_heads, self.max_seq_len, self.sparsity, self.target_size, self.requires_grad
            )
            yield inputs

    def get_bwd_fn(self, fwd_fn: Callable[..., Any]) -> Callable[..., Any]:
        o = fwd_fn()
        o_tensor = input_filter(
            lambda x: isinstance(x, torch.Tensor),
            o,
        )
        do = torch.rand_like(o_tensor)
        fn = lambda: o_tensor.backward(do, retain_graph=True)
        return fn

    @register_metric()
    def tflops(
        self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics
    ) -> float:
        ratio = 2.0  # triangular masking
        f1 = 0.0
        f2 = 0.0
        jagged = True
        qkv, seq_offsets, timestamps, num_targets = example_inputs
        q = qkv[:, :, :128]
        v = qkv[:, :, 256:384]
        _, nheads, attn_dim = q.shape
        _, _, hidden_dim = v.shape
        max_seqlen = timestamps.size(1) - 1

        for i in range(self.batch_size):
            seq_len = (
                int((seq_offsets[i + 1] - seq_offsets[i]).item())
                if jagged
                else max_seqlen
            )
            # (QK^T), dQ = d(QK^T)K, dK^T = Q^Td(QK^T)
            f1 += 2 * self.num_heads * attn_dim * seq_len**2 // ratio
            # (QK^T)V, d(QK^T) = dOV^T, dV = (QK^T)^TdO,
            f2 += 2 * self.num_heads * hidden_dim * seq_len**2 // ratio
        if self.mode == Mode.FWD:
            tflops = f1 + f2  # computes (QK^T) and (QK^T)V
        elif self.mode == Mode.BWD:
            tflops = 3 * f1 + 2 * f2  # computes (QK^T), dQ, dK, dV, d(QK^T)
        elif self.mode == Mode.FWD_BWD:
            tflops = 4 * f1 + 3 * f2
        return tflops / metrics.latency * 1e-9
