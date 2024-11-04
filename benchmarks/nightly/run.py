"""
Tritonbench nightly run
"""
from tritonbench.utils.parser import run_in_task


OPERATORS = [
    "launch_latency",
]


def run():
    for op in OPERATORS:
        run_in_task(op)


if __name__ == "__main__":
    run()