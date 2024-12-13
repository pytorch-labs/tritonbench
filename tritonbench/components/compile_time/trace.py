import torch
from tritonbench.utils.env_utils import fresh_triton_cache

from typing import Callable

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
