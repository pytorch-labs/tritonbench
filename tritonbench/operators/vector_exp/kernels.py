import triton
import triton.language as tl

from tritonbench.kernels.profile import time


@triton.jit
def triton_exp_kernel(
    x_ptr,  # *Pointer* to input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
    profile_mem=None,  # *Pointer* to profile_mem.
):
    if profile_mem is not None:
        start = time()
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.exp(x)
    # Write exp(x) back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

    if profile_mem is not None:
        end = time()
        tl.store(profile_mem + pid, end - start)
