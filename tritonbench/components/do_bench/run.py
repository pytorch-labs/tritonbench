import torch
import triton
import statistics
from dataclasses import dataclass

from typing import Optional, List

@dataclass
class Latency:
    times: List[float]

    def __str__(self):
        """By default, use p50"""
        return self.to_str()

    @property
    def p50(self):
        return statistics.median(self.times)

    def __add__(self, other):
        return self.p50 + other
    
    def __sub__(self, other):
        return self.p50 - other
    
    def __mul__(self, other):
        return self.p50 * other

    def __truediv__(self, other):
        return self.p50 / other

    def __floordiv__(self, other):
        return self.p50 // other

    def to_str(self, mode="p50"):
        if mode == "p50":
            return self.p50
        elif mode == "with_variance":
            min = min(self.time)
            max = max(self.times)
            max_variance = max((max - self.p50), (self.p50 - min))
            return f"{self.p50} (+/-{max_variance})"
        elif mode == "max":
            return max(self.times)
        elif mode == "min":
            return min(self.times)
        elif mode == "mean":
            return statistics.mean(self.times)
        else:
            raise ValueError(f"Unsupported latency output mode: {mode}")

def do_bench_wrapper(
    fn,
    warmup,
    rep,
    grad_to_none,
    use_cuda_graphs: bool = False,
    bypass_fail: bool = False,
) -> Optional[Latency]:
    """Wrapper to triton's do_bench to gain latency."""
    if use_cuda_graphs:
        with torch.cuda.stream(torch.cuda.Stream()):
            return Latency(times=triton.testing.do_bench_cudagraph(
                fn,
                rep=rep,
                return_mode="all",
                grad_to_none=grad_to_none,
            ))
    else:
        try:
            return Latency(times=triton.testing.do_bench(
                fn,
                warmup=warmup,
                rep=rep,
                return_mode="all",
                grad_to_none=grad_to_none,
            ))
        except Exception as e:
            if not bypass_fail:
                raise e
            return None
