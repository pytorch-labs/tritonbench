# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os

from typing import Callable, Optional, Tuple

import torch
from gen_ai.llm_inference.fb.llm.quantization.kv_quantize import quantize_kv_fp8

from tritonbench.utils.path_utils import add_ld_library_path

# [Optional] flash_attn v2
HAS_FLASH_V2 = True
try:
    from flash_attn import flash_attn_interface as flash_attn_v2
except (ImportError, IOError, AttributeError):
    HAS_FLASH_V2 = False

HAS_CUDA_124 = (
    torch.cuda.is_available() and torch.version.cuda and torch.version.cuda >= "12.4"
)

# [Optional] flash_attn v3
HAS_FLASH_V3 = True
try:
    torch_lib_path = os.path.join(os.path.dirname(__file__), "lib")
    with add_ld_library_path(torch_lib_path):
        import flash_attn_interface as flash_attn_v3
except (ImportError, IOError, AttributeError):
    try:
        from ai_codesign.gen_ai.flash_attention_v2.hopper import (
            flash_attn_interface as flash_attn_v3,
        )
    except (ImportError, IOError, AttributeError):
        HAS_FLASH_V3 = False

# [Optional] xformers backend
# try:
# import xformers  # @manual=//fair/xformers:xformers

from typing import Generator, List

# [Optional] xformers backend
HAS_XFORMERS = True
try:
    import xformers.ops.fmha as fmha  # @manual=//fair/xformers:xformers
except (ImportError, IOError, AttributeError):
    HAS_XFORMERS = False


torch.ops.load_library(
    "//deeplearning/fbgemm/fbgemm_gpu/experimental:gen_ai_attention_ops"
)

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--seq-len-q", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len-kv", type=int, default=4096, help="Batch size")
    parser.add_argument("--max-len-kv", type=int, default=8192, help="Batch size")
    parser.add_argument("--head-q", type=int, default=8, help="Number of Q heads")
    parser.add_argument("--head-kv", type=int, default=1, help="Number of KV heads")
    parser.add_argument(
        "--head-d", type=int, default=128, help="specify head dimension"
    )
    parser.add_argument("--page-size", type=int, default=256, help="Page Size")
    return parser.parse_args(args)


from dataclasses import astuple, dataclass


@dataclass
class _Shape:
    batch: int
    seq_len_q: int
    seq_len_kv: int
    max_len_kv: int
    head_q: int
    head_kv: int
    head_d: int

    def unpack(self):
        return astuple(self)


def _generate_shapes():
    HEAD_Q = 8
    HEAD_KV = 1
    HEAD_D = 128
    max_len_kv = 8192

    SEQ_LEN_KVs = [1024, 2048, 4096, 8190]
    SEQ_LEN_Qs = [1, 4]
    BATCHs = [32, 64]

    return [
        _Shape(
            batch=batch,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            max_len_kv=max_len_kv,
            head_q=HEAD_Q,
            head_kv=HEAD_KV,
            head_d=HEAD_D,
        )
        for seq_len_q in SEQ_LEN_Qs
        for batch in BATCHs
        for seq_len_kv in SEQ_LEN_KVs
    ]


CAUSAL = True


def _pack_xformer_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
]:
    batch, seq_len_q, head_q, head_d = q.shape
    _, max_len_kv, head_kv, _ = k.shape

    attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[seq_len_q] * batch,
        kv_seqlen=cache_seqlens.tolist(),
        kv_padding=max_len_kv,
    )

    q = q.view(1, -1, head_q, head_d)
    k = k.expand(-1, -1, head_q, -1).view(1, -1, head_q, k.shape[-1])
    v = v.expand(-1, -1, head_q, -1).view(1, -1, head_q, v.shape[-1])
    return q, k, v, attn_bias


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"

    DEFAULT_METRICS = ["latency", "tflops", "gbps", "speedup"]

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        if args.batch:
            self.shapes = [
                _Shape(
                    args.batch,
                    args.seq_len_q,
                    args.seq_len_kv,
                    args.max_len_kv,
                    args.head_q,
                    args.head_kv,
                    args.head_d,
                )
            ]
        else:
            self.shapes = _generate_shapes()

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            batch, seq_len_q, seq_len_kv, max_len_kv, head_q, head_kv, head_d = (
                shape.unpack()
            )
            q = torch.randn(
                (batch, seq_len_q, head_q, head_d),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )
            k_cache = torch.randn(
                (batch, max_len_kv, head_kv, head_d),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )
            v_cache = torch.randn(
                (batch, max_len_kv, head_kv, head_d),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )
            cache_seqlens = torch.tensor(
                [seq_len_kv] * batch, dtype=torch.int32, device=self.device
            )

            yield (q, k_cache, v_cache, cache_seqlens)

    @register_x_val(label="(Batch, SeqLenQ, SeqLenKV, MaxLenKV, HeadQ, HeadKV, HeadD)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int, int, int, int, int]:
        q, k_cache, v_cache, cache_seqlens = example_inputs
        batch, seq_len_q, head_q, head_d = q.shape
        _, max_len_kv, head_kv, _ = k_cache.shape
        seq_len_kv = cache_seqlens.max().item()
        return (batch, seq_len_q, seq_len_kv, max_len_kv, head_q, head_kv, head_d)

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        accuracy = True

        try:
            torch.testing.assert_close(output, baseline_output, atol=1e-3, rtol=0.5)
        except Exception:
            accuracy = False
        finally:
            return accuracy

    @register_metric()
    def gbps(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics) -> float:
        # fp16/bf16 by default
        Q_ELEMENT_SIZE = 2
        K_ELEMENT_SIZE = 2
        if "fp8qkv" in str(fn):
            Q_ELEMENT_SIZE = 1
            K_ELEMENT_SIZE = 1
        elif "fp8kv" in str(fn):
            K_ELEMENT_SIZE = 1

        def nbytes(t):
            return t.numel() * Q_ELEMENT_SIZE

        q, k_cache, v_cache, cache_seqlens = example_inputs

        q_bytes = nbytes(q)
        # k_cache: B, max_len_kv, head_kv, head_d
        k_cache_bytes = (
            k_cache.shape[-1]
            * k_cache.shape[-2]
            * int(cache_seqlens.sum().item())
            * K_ELEMENT_SIZE
        )
        v_cache_bytes = k_cache_bytes
        gb = (q_bytes + k_cache_bytes + v_cache_bytes) * 1e-9
        return gb / metrics.latency * 1e3

    @register_metric()
    def flops(self, fn, example_inputs, metrics: BenchmarkOperatorMetrics) -> float:
        q, k_cache, v_cache, cache_seqlens = example_inputs

        batch, seq_len_q, head_q, head_d = q.shape
        total_kv_length = int(cache_seqlens.sum().item())

        qk_gemm = 2 * total_kv_length * seq_len_q * head_q * head_d

        attention_flops = qk_gemm * 2

        return attention_flops

    @register_benchmark(enabled=HAS_FLASH_V2)
    def fa2_kvcache(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        return lambda: flash_attn_v2.flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            causal=CAUSAL,
        )

    @register_benchmark(enabled=HAS_FLASH_V3)
    def fa3_kvcache_gqa_heuristic(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        return lambda: flash_attn_v3.flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            causal=CAUSAL,
            pack_gqa=True,
            num_splits=0,
        )

    @register_benchmark(enabled=HAS_XFORMERS and HAS_FLASH_V3)
    def fa3_mha_varlen_fwd(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        _q, _k, _v, attn_bias = _pack_xformer_input(q, k_cache, v_cache, cache_seqlens)
        return lambda: fmha.memory_efficient_attention_forward(
            _q,
            _k,
            _v,
            attn_bias,
            op=fmha.flash3.FwOp,
        ).view(q.shape)

    @register_benchmark(enabled=HAS_FLASH_V2 and HAS_XFORMERS)
    def fa2_mha_varlen_fwd(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        _q, _k, _v, attn_bias = _pack_xformer_input(q, k_cache, v_cache, cache_seqlens)
        return lambda: fmha.memory_efficient_attention_forward(
            _q,
            _k,
            _v,
            attn_bias,
            op=fmha.flash.FwOp,
        ).view(q.shape)

    @register_benchmark(baseline=True, enabled=HAS_XFORMERS)
    def triton_splitk(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        _q, _k, _v, attn_bias = _pack_xformer_input(q, k_cache, v_cache, cache_seqlens)

        return lambda: fmha.memory_efficient_attention_forward(
            _q,
            _k,
            _v,
            attn_bias,
            op=fmha.triton_splitk.FwOp,
        ).view(q.shape)

    @register_benchmark(enabled=HAS_FLASH_V3)
    def fa3_kvcache_fp8qkv(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        _q = q.to(torch.float8_e4m3fn)
        _k_cache = k_cache.to(torch.float8_e4m3fn)
        _v_cache = v_cache.to(torch.float8_e4m3fn)
        return lambda: flash_attn_v3.flash_attn_with_kvcache(
            q=_q,
            k_cache=_k_cache,
            v_cache=_v_cache,
            cache_seqlens=cache_seqlens,
            causal=CAUSAL,
            pack_gqa=True,
            num_splits=0,
        )

    @register_benchmark(enabled=True)
    def fbgemm_gqa_fp8kv(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Callable:
        kv_cache_quant_num_groups = 1

        k_fp8 = quantize_kv_fp8(k_cache, kv_cache_quant_num_groups).view(torch.uint8)
        v_fp8 = quantize_kv_fp8(v_cache, kv_cache_quant_num_groups).view(torch.uint8)
        return lambda: torch.ops.fbgemm.gqa_attn_splitk(
            q,
            k_fp8,
            v_fp8,
            cache_seqlens,
            1.0,  # qk_scale
            num_split_ks=16,
            kv_cache_quant_num_groups=kv_cache_quant_num_groups,
            use_tensor_cores=True,
            cache_logical_dtype_int=1,  # FP8 = 1
        )
