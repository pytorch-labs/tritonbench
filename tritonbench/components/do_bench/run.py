import torch
import triton


def do_bench_wrapper(
    fn,
    warmup,
    rep,
    grad_to_none,
    use_cuda_graphs: bool = False,
    bypass_fail: bool = False,
):
    """Wrapper to triton's do_bench to gain latency."""
    if use_cuda_graphs:
        with torch.cuda.stream(torch.cuda.Stream()):
            return triton.testing.do_bench_cudagraph(
                fn,
                rep=rep,
                return_mode="all",
                grad_to_none=grad_to_none,
            )
    else:
        try:
            return triton.testing.do_bench(
                fn,
                warmup=warmup,
                rep=rep,
                return_mode="all",
                grad_to_none=grad_to_none,
            )
        except Exception as e:
            if not bypass_fail:
                raise e
            return None
