// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor w2a16(const at::Tensor& X, const at::Tensor& WQ);

TORCH_LIBRARY_FRAGMENT(mixed_gemm, m) {
  m.def("w2a16_gemm(Tensor X, Tensor WQ) -> Tensor");
}

TORCH_LIBRARY_IMPL(mixed_gemm, CUDA, m) {
  m.impl("w2a16_gemm", w2a16);
}
