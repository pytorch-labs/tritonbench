/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include <iostream>

#include <ATen/ATen.h>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

#include "helper.h"

using namespace cute;

template <int TB_M, int TB_N, int TB_K, typename INPUT_DTYPE>
at::Tensor w2a16_kernel(
    const at::Tensor& X, // FP16/BF16
    const at::Tensor& WQ // INT2, packed in INT8
) {
  TORCH_CHECK(X.is_cuda() && X.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  // X: [M, K]
  // WQ: [N, K]
  int M = X.size(0);
  int K = X.size(1);
  int N = WQ.size(0);

  auto O = at::empty({M, N}, X.options());

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// GEMM kernel configurations
  /////////////////////////////////////////////////////////////////////////////////////////////////
  using MmaType = INPUT_DTYPE;
  using QuantType = cutlass::int2b_t;

  // A matrix configuration
  using ElementA = MmaType;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  // B matrix configuration
  using ElementB = QuantType;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  // Layout transposes
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  // C/D matrix configuration
  using ElementOut = INPUT_DTYPE;
  using LayoutOut = cutlass::layout::RowMajor;
  constexpr int AlignmentOut = 128 / cutlass::sizeof_bits<ElementOut>::value;

  // Kernel configurations
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<Int<TB_M>, Int<TB_N>, Int<TB_K>>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  // Epilogue
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          EpilogueTileType,
          ElementAccumulator,
          ElementCompute,
          ElementOut,
          typename cutlass::layout::LayoutTranspose<LayoutOut>::type,
          AlignmentOut,
          ElementOut,
          typename cutlass::layout::LayoutTranspose<LayoutOut>::type,
          AlignmentOut,
          EpilogueSchedule>::CollectiveOp;

  // MainLoop for convert-only mode
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementB,
          // LayoutB,
          LayoutB_Transpose,
          AlignmentB,
          ElementA,
          // LayoutA,
          LayoutA_Transpose,
          AlignmentA,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  // Kernel definition
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  // Final GEMM adapter
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // using StrideA = typename GemmKernel::StrideA;
  // using StrideB = typename GemmKernel::StrideB;
  // Stride definitions
  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
  using StrideOut = typename GemmKernel::StrideC;

  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideOut stride_Out =
      cutlass::make_cute_packed_stride(StrideOut{}, cute::make_shape(N, M, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K},
      {reinterpret_cast<ElementB*>(WQ.data_ptr()),
       stride_B,
       reinterpret_cast<ElementA*>(X.data_ptr()),
       stride_A},
      {{},
       (ElementOut*)O.data_ptr(),
       stride_Out,
       (ElementOut*)O.data_ptr(),
       stride_Out}};

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());

  return O;
}

template <typename INPUT_DTYPE>
at::Tensor dispatch_w2a16_kernel(const at::Tensor& X, const at::Tensor& WQ) {
  // template <int TB_M, int TB_N, int TB_K, typename INPUT_DTYPE>
  return w2a16_kernel<128, 128, 128, INPUT_DTYPE>(X, WQ);
}

at::Tensor w2a16(const at::Tensor& X, const at::Tensor& WQ) {
  if (X.dtype() == at::kHalf) {
    return dispatch_w2a16_kernel<cutlass::half_t>(X, WQ);
  } else if (X.dtype() == at::kBFloat16) {
    return dispatch_w2a16_kernel<cutlass::bfloat16_t>(X, WQ);
  } else {
    throw std::runtime_error("DType of the activation (X) is not supported");
  }
}
