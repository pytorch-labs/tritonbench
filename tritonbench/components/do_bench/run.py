import statistics
import time

from typing import List, Optional

import torch
import triton

NS_TO_MS = 1e-6


class Latency:
    times: List[float]

    def __init__(self, times):
        self.times = self._remove_outliers_iqr(times)

    def __str__(self):
        """By default, use p50"""
        return self.to_str()

    def _remove_outliers_iqr(self, data):
        """
        Removes outliers from a list of floats using the IQR method.

        Args:
            data: A list of floats.

        Returns:
            A new list with outliers removed.
        """
        starting_length = len(data)
        if starting_length <= 3:
            return data
        if not data:
            return []

        data.sort()
        quantiles = statistics.quantiles(data, n=100)
        q1 = quantiles[25]
        q3 = quantiles[75]
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        filtered_data = [x for x in data if lower_bound <= x and x <= upper_bound]
        end_len = len(filtered_data)
        if end_len != starting_length:
            print(
                f"Removed {starting_length - end_len} outliers from {starting_length} samples"
            )
        return filtered_data

    @property
    def p50(self):
        return statistics.median_low(self.times)

    def __add__(self, other):
        return self.p50 + other.p50 if isinstance(other, Latency) else self.p50 + other

    def __radd__(self, other):
        return other.p50 + self.p50 if isinstance(other, Latency) else other + self.p50

    def __sub__(self, other):
        return self.p50 - other.p50 if isinstance(other, Latency) else self.p50 - other

    def __rsub__(self, other):
        return other.p50 - self.p50 if isinstance(other, Latency) else other - self.p50

    def __mul__(self, other):
        return self.p50 * other.p50 if isinstance(other, Latency) else self.p50 * other

    def __rmul__(self, other):
        return other.p50 * self.p50 if isinstance(other, Latency) else other * self.p50

    def __truediv__(self, other):
        return self.p50 / other.p50 if isinstance(other, Latency) else self.p50 / other

    def __rtruediv__(self, other):
        return other.p50 / self.p50 if isinstance(other, Latency) else other / self.p50

    def __floordiv__(self, other):
        return (
            self.p50 // other.p50 if isinstance(other, Latency) else self.p50 // other
        )

    def __rfloordiv__(self, other):
        return (
            other.p50 // self.p50 if isinstance(other, Latency) else other // self.p50
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


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(torch, return_mode)(times).item()


def _do_bench_cpu(
    fn, warmup, rep=20, grad_to_none=None, quantiles=None, return_mode="mean"
):
    """Measure latency of a function on CPU."""
    assert return_mode in ["min", "max", "mean", "median", "all"]
    fn()
    # Estimate the runtime of the function
    t0 = time.time_ns()
    for _ in range(5):
        fn()
    t1 = time.time_ns()
    estimate_ms = (t1 - t0) * NS_TO_MS / 5

    # compute number of warmup and repeat
    if estimate_ms == 0:
        n_repeat = 1000
        n_warmup = 1000
    else:
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))
    # Warm-up
    for _ in range(n_warmup):
        fn()
    times_ms = []
    # Benchmark
    for _i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # record time of `fn`
        t0 = time.time_ns()
        fn()
        t1 = time.time_ns()
        times_ms.append((t1 - t0) * NS_TO_MS)
    times = torch.tensor(times_ms, dtype=torch.float)
    return _summarize_statistics(times, quantiles, return_mode)


def do_bench_wrapper(
    fn,
    warmup,
    rep,
    grad_to_none,
    device: str = "cuda",
    use_cuda_graphs: bool = False,
    bypass_fail: bool = False,
) -> Optional[Latency]:
    """Wrapper to triton's do_bench to gain latency."""
    try:
        if device == "cpu":
            return Latency(
                times=_do_bench_cpu(
                    fn,
                    warmup=warmup,
                    rep=rep,
                    return_mode="all",
                    grad_to_none=grad_to_none,
                )
            )
        elif use_cuda_graphs:
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
