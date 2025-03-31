// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <vector>

#include "h100.cu"

std::vector<at::Tensor> tk_attention_forward(at::Tensor &q, at::Tensor &k,
                                             at::Tensor &v, bool causal) {
  TORCH_CHECK(q.dim() == 4);
  TORCH_CHECK(k.dim() == 4);
  TORCH_CHECK(v.dim() == 4);

  // Batch sizes
  TORCH_CHECK(q.size(0) == k.size(0));
  TORCH_CHECK(q.size(0) == v.size(0));

  // Sequence length
  TORCH_CHECK(k.size(1) == v.size(1));

  // Num heads
  TORCH_CHECK(q.size(2) == k.size(2));
  TORCH_CHECK(q.size(2) == v.size(2));

  // Embedding per head
  TORCH_CHECK(q.size(3) == k.size(3));

  auto batch = q.size(0);
  auto seq_len = q.size(2);
  auto head_dim = q.size(3);
  auto is_causal = causal;
  auto qo_heads = q.size(1);
  auto kv_heads = k.size(1);
  // check to see that these dimensions match for all inputs
  TORCH_CHECK(q.size(0) == batch,
              "Q batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(k.size(0) == batch,
              "K batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(v.size(0) == batch,
              "V batch dimension - idx 0 - must match for all inputs");

  TORCH_CHECK(
      q.size(2) == seq_len,
      "Q sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      k.size(2) == seq_len,
      "K sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      v.size(2) == seq_len,
      "V sequence length dimension - idx 2 - must match for all inputs");

  TORCH_CHECK(
      q.size(3) == head_dim,
      "Q head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      k.size(3) == head_dim,
      "K head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      v.size(3) == head_dim,
      "V head dimension - idx 3 - must match for all non-vector inputs");

  TORCH_CHECK(k.size(1) == v.size(1),
              "k and v must have the same number of heads");

  TORCH_CHECK(qo_heads >= kv_heads,
              "qo_heads must be greater than or equal to kv_heads");
  TORCH_CHECK(qo_heads % kv_heads == 0,
              "qo_heads must be divisible by kv_heads");

  // check to see that these dimensions match for all inputs
  TORCH_CHECK(q.size(0) == batch,
              "Q batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(k.size(0) == batch,
              "K batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(v.size(0) == batch,
              "V batch dimension - idx 0 - must match for all inputs");

  TORCH_CHECK(
      q.size(2) == seq_len,
      "Q sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      k.size(2) == seq_len,
      "K sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      v.size(2) == seq_len,
      "V sequence length dimension - idx 2 - must match for all inputs");

  TORCH_CHECK(
      q.size(3) == head_dim,
      "Q head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      k.size(3) == head_dim,
      "K head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      v.size(3) == head_dim,
      "V head dimension - idx 3 - must match for all non-vector inputs");

  TORCH_CHECK(qo_heads >= kv_heads,
              "QO heads must be greater than or equal to KV heads");
  TORCH_CHECK(qo_heads % kv_heads == 0,
              "QO heads must be divisible by KV heads");
  TORCH_CHECK(q.size(1) == qo_heads,
              "QO head dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(k.size(1) == kv_heads,
              "KV head dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(v.size(1) == kv_heads,
              "KV head dimension - idx 1 - must match for all inputs");

  auto hr = qo_heads / kv_heads;

  c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
  c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
  c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();

  bf16 *d_q = reinterpret_cast<bf16 *>(q_ptr);
  bf16 *d_k = reinterpret_cast<bf16 *>(k_ptr);
  bf16 *d_v = reinterpret_cast<bf16 *>(v_ptr);

  // for the returned outputs
  at::Tensor o = at::empty(
      {static_cast<const uint>(batch), static_cast<const uint>(qo_heads),
       static_cast<const uint>(seq_len), static_cast<const uint>(head_dim)},
      v.options());

  at::Tensor l_vec = at::empty(
      {static_cast<const uint>(batch), static_cast<const uint>(qo_heads),
       static_cast<const uint>(seq_len), static_cast<const uint>(1)},
      at::TensorOptions()
          .dtype(at::kFloat)
          .device(q.device())
          .memory_format(at::MemoryFormat::Contiguous));

  bf16 *o_ptr = reinterpret_cast<bf16 *>(o.data_ptr<c10::BFloat16>());
  bf16 *d_o = reinterpret_cast<bf16 *>(o_ptr);

  float *l_ptr = reinterpret_cast<float *>(l_vec.data_ptr<float>());
  float *d_l = reinterpret_cast<float *>(l_ptr);

  cudaDeviceSynchronize();
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (head_dim == 64) {
    using q_tile = st_bf<fwd_attend_ker_tile_dims<64>::qo_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;
    using k_tile = st_bf<fwd_attend_ker_tile_dims<64>::kv_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;
    using v_tile = st_bf<fwd_attend_ker_tile_dims<64>::kv_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<64>::qo_height,
                                    fwd_attend_ker_tile_dims<64>::tile_width>>;
    using o_tile = st_bf<fwd_attend_ker_tile_dims<64>::qo_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;

    using q_global = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_global = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_global = gl<bf16, -1, -1, -1, -1, v_tile>;
    using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;

    using globals = fwd_globals<64>;

    q_global qg_arg{d_q, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    k_global kg_arg{d_k, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    v_global vg_arg{d_v, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    l_global lg_arg{d_l, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads), 1U,
                    static_cast<unsigned int>(seq_len)};
    o_global og_arg{d_o, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 64U};

    globals g{qg_arg,
              kg_arg,
              vg_arg,
              lg_arg,
              og_arg,
              static_cast<int>(seq_len),
              static_cast<int>(hr)};

    auto mem_size = kittens::MAX_SHARED_MEMORY;
    auto threads = NUM_WORKERS * kittens::WARP_THREADS;

    // TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0,
    // "sequence length must be divisible by 192");
    dim3 grid(seq_len / (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4),
              qo_heads, batch);

    if (is_causal) {
      cudaFuncSetAttribute(fwd_attend_ker<64, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           mem_size);

      fwd_attend_ker<64, true>
          <<<grid, (32 * NUM_WORKERS), mem_size, stream>>>(g);
    } else {
      cudaFuncSetAttribute(fwd_attend_ker<64, false>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           mem_size);

      fwd_attend_ker<64, false>
          <<<grid, (32 * NUM_WORKERS), mem_size, stream>>>(g);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaStreamSynchronize(stream);
  }

  if (head_dim == 128) {
    using q_tile = st_bf<fwd_attend_ker_tile_dims<128>::qo_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;
    using k_tile = st_bf<fwd_attend_ker_tile_dims<128>::kv_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;
    using v_tile = st_bf<fwd_attend_ker_tile_dims<128>::kv_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<128>::qo_height,
                                    fwd_attend_ker_tile_dims<128>::tile_width>>;
    using o_tile = st_bf<fwd_attend_ker_tile_dims<128>::qo_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;

    using q_global = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_global = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_global = gl<bf16, -1, -1, -1, -1, v_tile>;
    using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;

    using globals = fwd_globals<128>;

    q_global qg_arg{d_q, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 128U};
    k_global kg_arg{d_k, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 128U};
    v_global vg_arg{d_v, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 128U};
    l_global lg_arg{d_l, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads), 1U,
                    static_cast<unsigned int>(seq_len)};
    o_global og_arg{d_o, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 128U};

    globals g{qg_arg,
              kg_arg,
              vg_arg,
              lg_arg,
              og_arg,
              static_cast<int>(seq_len),
              static_cast<int>(hr)};

    auto mem_size = kittens::MAX_SHARED_MEMORY;
    auto threads = NUM_WORKERS * kittens::WARP_THREADS;

    // TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0,
    // "sequence length must be divisible by 192");
    dim3 grid(seq_len / (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4),
              qo_heads, batch);

    if (is_causal) {
      cudaFuncSetAttribute(fwd_attend_ker<128, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           mem_size);

      fwd_attend_ker<128, true>
          <<<grid, (32 * NUM_WORKERS), mem_size, stream>>>(g);
    } else {
      cudaFuncSetAttribute(fwd_attend_ker<128, false>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           mem_size);

      fwd_attend_ker<128, false>
          <<<grid, (32 * NUM_WORKERS), mem_size, stream>>>(g);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaStreamSynchronize(stream);
  }

  return {o, l_vec};
  cudaDeviceSynchronize();
}

// Abstract implementation
std::vector<at::Tensor> tk_attention_forward_meta(at::Tensor &q, at::Tensor &k,
                                                  at::Tensor &v, bool causal) {
  TORCH_CHECK(q.dim() == 4);
  TORCH_CHECK(k.dim() == 4);
  TORCH_CHECK(v.dim() == 4);

  // Batch sizes
  TORCH_CHECK(q.sym_size(0) == k.sym_size(0));
  TORCH_CHECK(q.sym_size(0) == v.sym_size(0));

  // Sequence length
  TORCH_CHECK(k.sym_size(1) == v.sym_size(1));

  // Num heads
  TORCH_CHECK(q.sym_size(2) == k.sym_size(2));
  TORCH_CHECK(q.sym_size(2) == v.sym_size(2));

  // Embedding per head
  TORCH_CHECK(q.sym_size(3) == k.sym_size(3));

  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
  TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bf16");
  TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be bf16");
  return std::vector<at::Tensor>();
}

TORCH_LIBRARY_FRAGMENT(tk, m) {
  m.def("attention_forward(Tensor q, Tensor k, Tensor v, bool causal) -> "
        "List[Tensor]");
}

TORCH_LIBRARY_IMPL(tk, CUDA, m) {
  m.impl("attention_forward", tk_attention_forward);
}

TORCH_LIBRARY_IMPL(tk, Meta, m) {
  m.impl("attention_forward", TORCH_FN(tk_attention_forward_meta));
}
