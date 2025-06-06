# utils to identify triton versions


def has_warp_spec():
    import triton.language as tl

    return hasattr(tl, "async_task")


def has_new_tma():
    import triton
    import triton.language as tl

    return hasattr(triton, "set_allocator") and hasattr(tl, "make_tensor_descriptor")
