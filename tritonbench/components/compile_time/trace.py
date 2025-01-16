from typing import Callable, Dict

import torch
from tritonbench.utils.env_utils import fresh_triton_cache, is_fbcode

if is_fbcode():
    from triton.fb.triton_util import triton_add_listener, TritonHook


def fbcode_do_compile_time_in_task(fn: Callable) -> Dict[str, float]:
    # not yet getting results that make sense to me
    detailed_data = {}
    with fresh_triton_cache():

        def _inner(**kwargs):
            stats = kwargs.get("stats", {})
            if not stats:
                return
            if "compile_time_stats" in stats:
                detailed_data["compile_time_stats"] = stats["compile_time_stats"]

        triton_add_listener(TritonHook.POST_COMPILE, _inner)
        fn()
    if "compile_time_stats" in detailed_data:
        return detailed_data["compile_time_stats"]
    return None


def do_compile_time_in_task(fn: Callable) -> float:
    with fresh_triton_cache():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
    latency_with_compile = start_event.elapsed_time(end_event)
    return latency_with_compile
