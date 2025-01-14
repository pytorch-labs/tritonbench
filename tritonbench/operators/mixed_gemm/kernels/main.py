# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch

# Load the custom op library
torch.ops.load_library(
    "//pytorch/tritonbench/tritonbench/operators/mixed_gemm/kernels:w2a16_gemm_lib"
)

from .quantize import dequantize_int2_to_bf16, quantize_bf16_to_int2


def main():
    M = 1024
    N = 8192
    K = 8192

    dtype = torch.bfloat16

    X = torch.randn([M, K], dtype=dtype, device="cuda")
    W = torch.tensor(
        [[-2.0, -1.0, 0, 1.0] * (K // 4) for _ in range(N)],
        dtype=dtype,
        device="cuda",
    )

    WQ = quantize_bf16_to_int2(W)

    out = torch.ops.mixed_gemm.w2a16_gemm(X, WQ)

    WQ = WQ.transpose(0, 1).contiguous().transpose(0, 1)

    print(X.shape, X.dtype, X.stride())
    print(WQ.shape, WQ.dtype, WQ.stride())
    print(out.shape, out.dtype, out.stride())

    W_dequant = dequantize_int2_to_bf16(WQ)

    out_ref = torch.matmul(X, W)
    out_ref_dequant = torch.matmul(X, W_dequant)

    print("==== CUTLASS ====")
    print(out.shape, out.dtype)
    print(out[0])
    print(out[1])
    print("==== Reference ====")
    print(out_ref.shape, out_ref.dtype)
    print(out_ref[0])
    print(out_ref[1])
    print("==== Reference Dequant ====")
    print(out_ref_dequant.shape, out_ref_dequant.dtype)
    print(out_ref_dequant[0])
    print(out_ref_dequant[1])


if __name__ == "__main__":
    main()
