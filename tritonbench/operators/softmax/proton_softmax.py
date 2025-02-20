import torch
import triton
import triton.intraprof as proton  # @manual=//triton:triton

import triton.language as tl

import numpy as np

SLOT = 256

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    profile_mem,  # *Pointer* to profile memory.
):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def proton_softmax(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    grid = lambda META: (n_rows,)

    proton_grid = proton.const_grid(
        grid,
        # config from autotune
        autotune_configs=[],
        # local variables that used in grid lambda function
        func_args={"n_rows": n_rows},
        # copy all named args except `proton_slots` and `profile_mem` in the kernel callsite
        BLOCK_SIZE=BLOCK_SIZE,
    )
    pconfig = proton.IntraKernelConfig(
        num_warps=num_warps, proton_slots=SLOT
    )
    profile_size = proton.intra_kernel_memsize(np.prod(proton_grid), pconfig)
    profile_mem = torch.empty(profile_size, device="cuda", dtype=torch.uint32)

    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    kernel_info = softmax_kernel[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
        profile_mem=profile_mem,  # *Pointer* to profile memory.
        proton_slots=SLOT,
    )
    proton.dump_chrome_trace(
        np.prod(proton_grid), pconfig, profile_mem, "chrome_trace.json", kernel_info
    )
    return y
