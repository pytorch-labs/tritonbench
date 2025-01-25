import statistics

from typing import List, Optional

import torch
import triton


class Latency:
    times: List[float]

    def __init__(self, times):
        self.times = times

    def __str__(self):
        """By default, use p50"""
        return self.to_str()

    @property
    def p50(self):
        return statistics.median(self.times)

    def __add__(self, other):
        return self.p50 + other.p50 if isinstance(other, Latency) else self.p50 + other

    def __sub__(self, other):
        return self.p50 - other.p50 if isinstance(other, Latency) else self.p50 - other

    def __mul__(self, other):
        return self.p50 * other.p50 if isinstance(other, Latency) else self.p50 * other

    def __truediv__(self, other):
        return self.p50 / other.p50 if isinstance(other, Latency) else self.p50 / other

    def __floordiv__(self, other):
        return (
            self.p50 // other.p50 if isinstance(other, Latency) else self.p50 // other
        )

    def __str__(self):
        return self.to_str()

    def to_str(self, mode="p50") -> str:
        if mode == "p50":
            return str(self.p50)
        elif mode == "with_variance":
            min_val = min(self.times)
            max_val = max(self.times)
            max_variance = max((max_val - self.p50), (self.p50 - min_val)) / self.p50
            return f"{self.p50:6f} (±{max_variance * 100:.2f}%)"
        elif mode == "max":
            return str(max(self.times))
        elif mode == "min":
            return str(min(self.times))
        elif mode == "mean":
            return str(statistics.mean(self.times))
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
            return Latency(
                times=triton.testing.do_bench_cudagraph(
                    fn,
                    rep=rep,
                    return_mode="all",
                    grad_to_none=grad_to_none,
                )
            )
    else:
        try:
            return Latency(
                times=triton.testing.do_bench(
                    fn,
                    warmup=warmup,
                    rep=rep,
                    return_mode="all",
                    grad_to_none=grad_to_none,
                )
            )
        except Exception as e:
            if not bypass_fail:
                raise e
            return None
