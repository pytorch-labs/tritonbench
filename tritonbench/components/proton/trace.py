import triton.profiler as proton

from typing import Callable, Optional

def proton_trace(scope_name: str, fn: Callable, warmup: int, flops: Optional[int]=None, bytes: Optional[int]=None):
    # warmup
    for _ in range(warmup):
        fn()
    metrics_dict = {}
    if flops:
        metrics_dict["flops"] = flops
    if bytes:
        metrics_dict["bytes"] = bytes
    with proton.scope(scope_name, metrics=metrics_dict):
        fn()
