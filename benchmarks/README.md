# User custom benchmarks using TritonBench as a library

In addition to the default benchmarking script `run.py`, we also welcome users to contribute
their own benchmarks that use TritonBench as a library under this directory.

Benefits of contributing user's own benchmarks:

1. Clear separation of code ownership. Each directory is independent under this directory.

2. The default benchmark script is optimized for comparing different kernels that perform
   the same math operation on the same input. Users can check-in their own analysis instead
   of the focus on default benchmark script.

3. We can help onboard user's custom benchmarks to the CI and publish the numbers on PyTorch HUD.
