"""
Common kernels used for profiling.
"""

import triton
import triton.language as tl

from tritonbench.utils.env_utils import is_cuda, is_hip


IS_CUDA = tl.constexpr(is_cuda())
IS_HIP = tl.constexpr(is_hip())


@triton.jit
def time(_semantic=None):
    if IS_HIP:
        return tl.inline_asm_elementwise(
            """
            s_memrealtime $0
            s_waitcnt vmcnt(0)
            """,
            "=r",
            [],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )
    elif IS_CUDA:
        return tl.extra.cuda.globaltimer()
    else:
        tl.static_assert(False, "Unsupported platform")
