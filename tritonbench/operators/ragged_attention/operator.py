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
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--min-seq-len-log2", type=int, default=8)
    parser.add_argument("--max-seq-len-log2", type=int, default=15)
    parser.add_argument("--num-buckets", type=int, default=2048)
    parser.add_argument("--seq-sparsity", type=float, default=0.95)
    parser.add_argument("--target-size", type=int, default=20)
    parser.add_argument("--sort-by-length", type=bool, default=True)
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
        self.attn_dim = args.attn_dim
        self.hidden_dim = args.hidden_dim
        self.min_seq_len_log2 = args.min_seq_len_log2
        self.max_seq_len_log2 = args.max_seq_len_log2
        self.num_buckets = args.num_buckets
        self.sparsity = args.seq_sparsity
        self.target_size = args.target_size
        self.sort_by_length = args.sort_by_length
        self.requires_grad = not (self.mode == Mode.FWD_NO_GRAD)

    @register_benchmark()
    def hstu_triton_ragged_attention(
        self, q, k, v, seq_offsets, timestamps, num_targets, seq_len
    ):
        attn = RaggedHSTUAttn(
            self.batch_size,
            self.num_heads,
            seq_len,
            self.num_buckets,
            self.sparsity,
            self.target_size,
            self.sort_by_length,
            self.requires_grad,
            persistent_kernel=False,
        )
        return lambda: attn(q, k, v, seq_offsets, timestamps, num_targets)

    # TODO: enable persistent kernels when the OSS backward is ready
    @register_benchmark(enabled=False)
    def hstu_triton_ragged_attention_persistent(
        self,
        q,
        k,
        v,
        seq_offsets,
        timestamps,
        num_targets,
        seq_len,
    ):
        attn = RaggedHSTUAttn(
            self.batch_size,
            self.num_heads,
            seq_len,
            self.num_buckets,
            self.sparsity,
            self.target_size,
            self.sort_by_length,
            self.requires_grad,
            persistent_kernel=True,
        )
        return lambda: attn(q, k, v, seq_offsets, timestamps, num_targets)

    def get_x_val(self, example_inputs):
        seq_len = example_inputs[-1]
        return (
            self.batch_size,
            self.num_heads,
            seq_len,
            self.num_buckets,
            self.sparsity,
            self.target_size,
            self.sort_by_length,
        )

    def get_input_iter(self):
        for seq_len in [2**i for i in range(self.min_seq_len_log2, self.max_seq_len_log2+1)]:
            yield get_test_inputs(
                self.batch_size,
                self.num_heads,
                self.attn_dim,
                self.hidden_dim,
                seq_len,
                self.sparsity,
                self.target_size,
                self.sort_by_length,
                self.requires_grad,
            )

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
    def flops(
        self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics
    ) -> float:
        ratio = 2.0  # triangular masking
        f1 = 0.0
        f2 = 0.0
        jagged = True
        q, k, v, seq_offsets, timestamps, num_targets, seq_len = example_inputs
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
            flops = f1 + f2  # computes (QK^T) and (QK^T)V
        elif self.mode == Mode.BWD:
            flops = 3 * f1 + 2 * f2  # computes (QK^T), dQ, dK, dV, d(QK^T)
        elif self.mode == Mode.FWD_BWD:
            flops = 4 * f1 + 3 * f2
        return flops
