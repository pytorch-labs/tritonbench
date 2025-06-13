from typing import Any

import torch

import triton
from torch.profiler import profile, ProfilerActivity

has_pt2 = False
from torch._dynamo.utils import CHROMIUM_EVENT_LOG, chromium_event_timed

if CHROMIUM_EVENT_LOG is not None:
    has_pt2 = True


CACHE_CLEAR_KERNEL = "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)"


def do_bench_cuda_time(
    fn: Any,
    warmup: int,
    rep: int,
    grad_to_none: bool,
    use_cuda_graphs: bool = False,
    bypass_fail: bool = False,
) -> float:
    """
    Return the aggregated CUDA time of a benchmarked operator backend.
    """

    di = triton.runtime.driver.active.get_device_interface()

    def synchronize_with_timing():
        di.synchronize()

    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    synchronize_with_timing()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    with profile(
        schedule=torch.profiler.schedule(
            wait=0, warmup=n_warmup, active=n_repeat, repeat=1
        ),
        # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
        activities=[ProfilerActivity.CUDA],
        with_stack=False,
        record_shapes=False,
    ) as prof:
        for _iters in range(n_warmup + n_repeat):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            cache.zero_()
            fn()
            prof.step()
        synchronize_with_timing()

    prof_averages = prof.key_averages(group_by_input_shape=False)
    # remove all zero cuda time events and cache clear kernel
    cuda_events = [
        event
        for event in prof_averages
        if event.device_time > 0 and event.key != CACHE_CLEAR_KERNEL
    ]
    print(
        prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total")
    )
    # return sum of all cuda events in ms
    return sum(e.device_time for e in cuda_events) / 1e3
