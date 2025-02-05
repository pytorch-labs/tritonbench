import triton


@triton.jit
def nop_kernel():
    pass
