// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <mutex>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include "h100.cu"

void tk_attention_forward(
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& o,
    at::Tensor& l,
    bool causal) {
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
  // check to see that these dimensions match for all inputs
  TORCH_CHECK(
      q.size(0) == batch,
      "Q batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(
      k.size(0) == batch,
      "K batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(
      v.size(0) == batch,
      "V batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(
      l.size(0) == batch,
      "L batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(
      o.size(0) == batch,
      "O batch dimension - idx 0 - must match for all inputs");

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
      l.size(2) == seq_len,
      "L sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      o.size(2) == seq_len,
      "O sequence length dimension - idx 2 - must match for all inputs");

  TORCH_CHECK(
      q.size(3) == head_dim,
      "Q head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      k.size(3) == head_dim,
      "K head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      v.size(3) == head_dim,
      "V head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      o.size(3) == head_dim,
      "O head dimension - idx 3 - must match for all non-vector inputs");

  // check if GQA
  auto qo_heads = q.size(1);
  auto kv_heads = k.size(1);

  TORCH_CHECK(
      k.size(1) == v.size(1), "k and v must have the same number of heads");
  TORCH_CHECK(
      q.size(1) == o.size(1), "q and o must have the same number of heads");

  TORCH_CHECK(
      qo_heads >= kv_heads,
      "qo_heads must be greater than or equal to kv_heads");
  TORCH_CHECK(
      qo_heads % kv_heads == 0, "qo_heads must be divisible by kv_heads");

  auto heads_ratio = qo_heads / kv_heads;
  auto is_causal = causal;

  c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
  c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
  c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();
  c10::BFloat16* o_ptr = o.data_ptr<c10::BFloat16>();
  float* l_ptr = l.data_ptr<float>();

  bf16* d_q = reinterpret_cast<bf16*>(q_ptr);
  bf16* d_k = reinterpret_cast<bf16*>(k_ptr);
  bf16* d_v = reinterpret_cast<bf16*>(v_ptr);
  bf16* d_o = reinterpret_cast<bf16*>(o_ptr);
  float* d_l = reinterpret_cast<float*>(l_ptr);

  CUtensorMap* tma_q_d;
  CUtensorMap* tma_k_d;
  CUtensorMap* tma_v_d;
  CUtensorMap* tma_o_d;
  CUtensorMap* tma_l_d;

  if (head_dim == 64) {
    tma_q_d = tma::allocate_and_create_tensor_map<st_bf<
        fwd_attend_ker_tile_dims<64>::qo_height,
        fwd_attend_ker_tile_dims<64>::tile_width>>(
        d_q,
        batch * qo_heads * seq_len /
            (fwd_attend_ker_tile_dims<64>::qo_height * kittens::TILE_DIM));
    tma_k_d = tma::allocate_and_create_tensor_map<st_bf<
        fwd_attend_ker_tile_dims<64>::kv_height,
        fwd_attend_ker_tile_dims<64>::tile_width>>(
        d_k,
        batch * kv_heads * seq_len /
            (fwd_attend_ker_tile_dims<64>::kv_height * kittens::TILE_DIM));
    tma_v_d = tma::allocate_and_create_tensor_map<st_bf<
        fwd_attend_ker_tile_dims<64>::kv_height,
        fwd_attend_ker_tile_dims<64>::tile_width>>(
        d_v,
        batch * kv_heads * seq_len /
            (fwd_attend_ker_tile_dims<64>::kv_height * kittens::TILE_DIM));
    tma_o_d = tma::allocate_and_create_tensor_map<st_bf<
        fwd_attend_ker_tile_dims<64>::qo_height,
        fwd_attend_ker_tile_dims<64>::tile_width>>(
        d_o,
        batch * qo_heads * seq_len /
            (fwd_attend_ker_tile_dims<64>::qo_height * kittens::TILE_DIM));
    tma_l_d = tma::allocate_and_create_tensor_map<col_vec<st_fl<
        fwd_attend_ker_tile_dims<64>::qo_height,
        fwd_attend_ker_tile_dims<64>::tile_width>>>(
        d_l,
        batch * qo_heads * seq_len /
            (fwd_attend_ker_tile_dims<64>::qo_height * kittens::TILE_DIM));
  }

  if (head_dim == 128) {
    tma_q_d = tma::allocate_and_create_tensor_map<st_bf<
        fwd_attend_ker_tile_dims<128>::qo_height,
        fwd_attend_ker_tile_dims<128>::tile_width>>(
        d_q,
        batch * qo_heads * seq_len /
            (fwd_attend_ker_tile_dims<128>::qo_height * kittens::TILE_DIM));
    tma_k_d = tma::allocate_and_create_tensor_map<st_bf<
        fwd_attend_ker_tile_dims<128>::kv_height,
        fwd_attend_ker_tile_dims<128>::tile_width>>(
        d_k,
        batch * kv_heads * seq_len /
            (fwd_attend_ker_tile_dims<128>::kv_height * kittens::TILE_DIM));
    tma_v_d = tma::allocate_and_create_tensor_map<st_bf<
        fwd_attend_ker_tile_dims<128>::kv_height,
        fwd_attend_ker_tile_dims<128>::tile_width>>(
        d_v,
        batch * kv_heads * seq_len /
            (fwd_attend_ker_tile_dims<128>::kv_height * kittens::TILE_DIM));
    tma_o_d = tma::allocate_and_create_tensor_map<st_bf<
        fwd_attend_ker_tile_dims<128>::qo_height,
        fwd_attend_ker_tile_dims<128>::tile_width>>(
        d_o,
        batch * qo_heads * seq_len /
            (fwd_attend_ker_tile_dims<128>::qo_height * kittens::TILE_DIM));
    tma_l_d = tma::allocate_and_create_tensor_map<col_vec<st_fl<
        fwd_attend_ker_tile_dims<128>::qo_height,
        fwd_attend_ker_tile_dims<128>::tile_width>>>(
        d_l,
        batch * qo_heads * seq_len /
            (fwd_attend_ker_tile_dims<128>::qo_height * kittens::TILE_DIM));
  }

  auto mem_size = kittens::MAX_SHARED_MEMORY;
  auto threads = NUM_WORKERS * kittens::WARP_THREADS;

  TORCH_CHECK(
      seq_len % (CONSUMER_WARPGROUPS * kittens::TILE_DIM * 4) == 0,
      "sequence length must be divisible by 192");
  dim3 grid(
      seq_len / (CONSUMER_WARPGROUPS * kittens::TILE_DIM * 4),
      batch * qo_heads,
      1);

  if (is_causal && head_dim == 64) {
    cudaFuncSetAttribute(
        fwd_attend_ker<64, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size);

    fwd_attend_ker<64, true><<<grid, threads, mem_size>>>(
        seq_len, heads_ratio, tma_q_d, tma_k_d, tma_v_d, tma_o_d, tma_l_d);
  }

  if (is_causal && head_dim == 128) {
    cudaFuncSetAttribute(
        fwd_attend_ker<128, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size);

    fwd_attend_ker<128, true><<<grid, threads, mem_size>>>(
        seq_len, heads_ratio, tma_q_d, tma_k_d, tma_v_d, tma_o_d, tma_l_d);
  }

  if (!is_causal && head_dim == 64) {
    cudaFuncSetAttribute(
        fwd_attend_ker<64, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size);

    fwd_attend_ker<64, false><<<grid, threads, mem_size>>>(
        seq_len, heads_ratio, tma_q_d, tma_k_d, tma_v_d, tma_o_d, tma_l_d);
  }

  if (!is_causal && head_dim == 128) {
    cudaFuncSetAttribute(
        fwd_attend_ker<128, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size);

    fwd_attend_ker<128, false><<<grid, threads, mem_size>>>(
        seq_len, heads_ratio, tma_q_d, tma_k_d, tma_v_d, tma_o_d, tma_l_d);
  }

  CHECK_CUDA_ERROR(cudaGetLastError());
  cudaDeviceSynchronize();
}

// Abstract implementation
void tk_attention_forward_meta(
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& o,
    at::Tensor& l,
    bool causal) {
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
  TORCH_CHECK(o.scalar_type() == at::kBFloat16, "o must be bf16");
  return;
}

TORCH_LIBRARY_FRAGMENT(tk, m) {
  m.def(
      "attention_forward(Tensor q, Tensor k, Tensor v, Tensor o, Tensor l, bool causal) -> ()");
}

TORCH_LIBRARY_IMPL(tk, CUDA, m) {
  m.impl("attention_forward", tk_attention_forward);
}

TORCH_LIBRARY_IMPL(tk, Meta, m) {
  m.impl("attention_forward", TORCH_FN(tk_attention_forward_meta));
}
