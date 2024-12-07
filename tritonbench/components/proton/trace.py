from typing import Callable, Optional

import triton.profiler as proton


def proton_trace(
    session_id: int,
    scope_name: str,
    fn: Callable,
    warmup: int,
    flops: Optional[int] = None,
    bytes: Optional[int] = None,
):
    # warmup
    for _ in range(warmup):
        fn()
    metrics_dict = {}
    if flops:
        metrics_dict["flops"] = flops
    if bytes:
        metrics_dict["bytes"] = bytes
    proton.activate(session_id)
    with proton.scope(scope_name, metrics=metrics_dict):
        fn()
    proton.deactivate(session_id)
