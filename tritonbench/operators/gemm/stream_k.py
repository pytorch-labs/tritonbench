"""
Stream-K matmul: https://arxiv.org/abs/2301.03598

Implementation from: https://github.com/ROCm/triton/blob/902b8329cadfb23b8ff4cbb2b2162eed61c3b47f/python/perf-kernels/streamk/streamk_kernel_atomic.py#L1

This kernel has some known numerical issues due to the use of atomic_add.
"""

import os

import torch

import triton
import triton.language as tl

from tritonbench.utils.env_utils import is_hip_mi300

from .triton_matmul_configs import get_full_amd_config_space

if os.environ.get("FULL_AUTOTUNING_AMD", "0") == "1" and torch.version.hip is not None:
    tuning_configs = get_full_amd_config_space(False)
else:
    tuning_configs = [
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
    ]


@triton.autotune(
    configs=tuning_configs,
    key=["M", "N", "K"],
)
@triton.heuristics(
    values={
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def streamk_gemm(
    A,
    B,
    C,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias_m,
    stride_bias_n,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    STREAMK_TILES: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    USE_BUFFER_OPS: tl.constexpr,
):
    if USE_BUFFER_OPS:
        tl.assume(M > 0)
        tl.assume(N > 0)
        tl.assume(K > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    if stride_bias_m:
        tl.assume(stride_bias_m >= 0)
    if stride_bias_n:
        tl.assume(stride_bias_n >= 0)

    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    iters_per_tile = tl.cdiv(K, BLOCK_K)
    total_tiles = num_pid_m * num_pid_n
    total_full_tiles = total_tiles - STREAMK_TILES

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    for tile_id in range(pid, total_full_tiles, NUM_SMS):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        rk = tl.arange(0, BLOCK_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        loop_k = tl.cdiv(K, BLOCK_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

        if HAS_BIAS:
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            bias_ = rn[None, :] * stride_bias_n + rm[:, None] * stride_bias_m
            bias = tl.load(
                bias_ptr + (tl.broadcast_to(bias_, (BLOCK_M, BLOCK_N))),
                mask=mask,
            ).to(acc.dtype)

        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_K + tl.arange(0, BLOCK_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]

        if HAS_BIAS:
            acc += bias

        c = acc.to(C.type.element_ty)
        tl.store(C_, c, mask=mask)

    tl.assume(pid >= 0)
    total_streamk_iters = STREAMK_TILES * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // NUM_SMS
    streamk_remainder_iters = total_streamk_iters % NUM_SMS

    start_iter = (
        total_full_tiles * iters_per_tile
        + pid * streamk_iters_pcu
        + tl.minimum(pid, streamk_remainder_iters)
    )

    last_iter = (
        total_full_tiles * iters_per_tile
        + (pid + 1) * streamk_iters_pcu
        + tl.minimum(pid + 1, streamk_remainder_iters)
    )
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        tile_iter_end = start_iter + (iters_per_tile - remainder)
        tile_id = start_iter // iters_per_tile
        end_iter = tl.minimum(tile_iter_end, last_iter)

        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) % M
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) % N
        rk = tl.arange(0, BLOCK_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        A_BASE = (
            A
            + rm[:, None] * stride_am
            + rk[None, :] * stride_ak
            + BLOCK_K * stride_ak * remainder
        )
        B_BASE = (
            B
            + rk[:, None] * stride_bk
            + rn[None, :] * stride_bn
            + BLOCK_K * stride_bk * remainder
        )
        A_BASE = tl.multiple_of(A_BASE, (1, 16))
        B_BASE = tl.multiple_of(B_BASE, (16, 1))

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
        if HAS_BIAS:
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            bias_ = rn[None, :] * stride_bias_n + rm[:, None] * stride_bias_m
            bias = tl.load(
                bias_ptr + (tl.broadcast_to(bias_, (BLOCK_M, BLOCK_N))),
                mask=mask,
            ).to(acc.dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                if EVEN_M:
                    a = tl.load(A_BASE)
                else:
                    mask_a = (rm < M)[:, None]
                    a = tl.load(A_BASE, mask=mask_a, other=0.0)
                if EVEN_N:
                    b = tl.load(B_BASE)
                else:
                    mask_b = (rn < N)[None, :]
                    b = tl.load(B_BASE, mask=mask_b, other=0.0)
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_K
                k_mask = global_k_offset + rk < K
                if EVEN_M:
                    a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                else:
                    mask_a = (rm < M)[:, None]
                    a = tl.load(A_BASE, mask=k_mask[None, :] & mask_a, other=0.0)
                if EVEN_N:
                    b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
                else:
                    mask_b = (rn < N)[None, :]
                    b = tl.load(B_BASE, mask=k_mask[:, None] & mask_b, other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]

        if HAS_BIAS:
            acc += bias

        c = acc.to(C.type.element_ty)
        tl.atomic_add(C_, c, mask=mask, sem="relaxed")

        start_iter = end_iter


def streamk_matmul(a, b, bias=None):
    M, K = a.shape
    _, N = b.shape
    dtype = a.dtype

    c = torch.zeros((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TODO: Remove in the future and move to triton heuristics
    # This is inconsistent for the full autotuning case.
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    if K > (M + N) * 2:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64

    # TODO: Figure out a way move this to @triton.heuristics to enable autotuning.
    # We need min(total_programs_streamk, total_tiles) for the launch grid.
    # We could maybe just launch with total_tiles, although this might hurt performance for smaller GEMMs.
    total_tiles = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
    total_programs_streamk = min(NUM_SMS, total_tiles)
    total_tiles_streamk = total_tiles % total_programs_streamk

    # Setting total_tiles_streamk to 0 will force StreamK to use only classical blocking
    # total_tiles_streamk = 0

    # Debugging code:
    # for two-tile Stream-K + data-parallel from original paper
    # if total_tiles - total_tiles_streamk > total_programs_streamk:
    #    total_tiles_streamk += total_programs_streamk
    # remaining tiles are computed using classical blocking

    # total_blocked_tiles = total_tiles - total_tiles_streamk
    # print("total_blocked_tiles", total_blocked_tiles)

    grids = min(total_programs_streamk, total_tiles)

    """
    print("total_tiles: ", total_tiles)
    print("total_tiles_streamk: ", total_tiles_streamk)
    print("total_programs_streamk: ", total_programs_streamk)
    print("grids: ", grids)
    """

    if bias is not None:
        is_bias_1d = bias.dim() == 1
        if not is_bias_1d:
            bias_stride_m = bias.stride(0)
            bias_stride_n = bias.stride(1)
        elif bias.dim() == 1:
            bias_stride_m = 0
            bias_stride_n = bias.stride(0)
    else:
        is_bias_1d = 0
        bias_stride_m = 0
        bias_stride_n = 0

    # Compute the number of XCDs for our launch grid
    # This is used to swizzle the launch grid across the XCDs
    NUM_XCDS = 1
    if is_hip_mi300():
        # Each MI300-series XCD has 38 SMs
        # TODO: programmatically query the number of XCDs on the device
        # We need this check for the chiplet swizzle to deal with the PID remapping
        # If we don't have enough programs this can cause issues (i.e., incomplete tiles)
        if grids % 38 == 0:
            NUM_XCDS = grids // 38
    # print("NUM_XCDS: ", grids, NUM_XCDS)
    use_buffer_ops = os.environ.get("AMDGCN_USE_BUFFER_OPS", "0") == "1"
    streamk_gemm[(grids,)](
        a,
        b,
        c,
        bias,  # bias_ptr
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias_stride_m,
        bias_stride_n,
        c.stride(0),
        c.stride(1),
        HAS_BIAS=bias is not None,
        NUM_SMS=grids,
        STREAMK_TILES=total_tiles_streamk,
        NUM_XCDS=NUM_XCDS,
        USE_BUFFER_OPS=use_buffer_ops,
    )

    # print(c)
    # print(a @ b)
    return c
