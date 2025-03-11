import statistics

from typing import List, Optional

import torch
import triton


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
