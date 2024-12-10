# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse

from typing import Callable, Generator, List, Optional, Tuple

import torch

from ads_mkl.ops.triton.triton_matmul_layernorm import (
    halfway_layernorm,
    native_matmul_layernorm,
    triton_fused_matmul_layernorm,
    triton_matmul_layernorm,
)

from ads_mkl.ops.triton.triton_matmul_layernorm_persistent import (
    triton_fused_matmul_layernorm_persistent,
)

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, help="M dimension")
    parser.add_argument("--n", type=int, default=256, help="N dimension")
    parser.add_argument("--k", type=int, default=512, help="K dimension")
    return parser.parse_args(args)


from dataclasses import astuple, dataclass


@dataclass
class _Shape:
    m: int
    n: int
    k: int

    def unpack(self):
        return astuple(self)


def _generate_default_shapes():
    Ms = [16 * 1024, 32 * 1024, 48 * 1024, 64 * 1024]
    Ns = [256]
    Ks = [512]

    return [
        _Shape(
            m=m,
            n=n,
            k=k,
        )
        for k in Ks
        for n in Ns
        for m in Ms
    ]


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "fp16"

    DEFAULT_METRICS = ["latency", "speedup"]

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        if args.m:
            self.shapes = [
                _Shape(
                    args.m,
                    args.n,
                    args.k,
                )
            ]
        else:
            self.shapes = _generate_default_shapes()

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            m, n, k = shape.unpack()

            a = torch.randn((m, k), device=self.device, dtype=self.dtype)
            b = torch.randn((k, n), device=self.device, dtype=self.dtype)
            # pyre-ignore
            ln = torch.nn.LayerNorm((n,), device=self.device, dtype=self.dtype)

            yield (a, b, ln)

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        a, b, ln = example_inputs
        m, k = a.shape
        _, n = b.shape
        return (m, n, k)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()

        try:
            torch.testing.assert_close(output, baseline_output, atol=1e-3, rtol=0.5)
            return True
        except Exception:
            return False

    @register_benchmark(baseline=True)
    def eager(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        ln: torch.nn.LayerNorm,
    ) -> Callable:
        return lambda: native_matmul_layernorm(a, b, ln)

    @register_benchmark()
    def inductor(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        ln: torch.nn.LayerNorm,
    ) -> Callable:
        compiled_fn = torch.compile(
            lambda: native_matmul_layernorm(
                a,
                b,
                ln,
            ),
            backend="inductor",
            options={"max_autotune": True},
        )
        return compiled_fn

    @register_benchmark()
    def semi_fused_reduction_gemm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        ln: torch.nn.LayerNorm,
    ) -> Callable:
        compiled_halfway_layernorm = torch.compile(
            halfway_layernorm,
            backend="inductor",
            options={"max_autotune": True},
        )
        return lambda: triton_matmul_layernorm(
            a, b, compiled_halfway_layernorm, ln.eps, ln.weight, ln.bias
        )

    @register_benchmark()
    def fused_reduction_gemm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        ln: torch.nn.LayerNorm,
    ) -> Callable:
        return lambda: triton_fused_matmul_layernorm(a, b, ln.weight, ln.bias, ln.eps)

    @register_benchmark()
    def fused_reduction_gemm_persistent(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        ln: torch.nn.LayerNorm,
    ) -> Callable:
        return lambda: triton_fused_matmul_layernorm_persistent(
            a, b, ln.weight, ln.bias, ln.eps
        )
