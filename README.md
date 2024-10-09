# TritonBench

TritonBench is a collection of PyTorch operators used to evaluation the performance of [Triton](https://github.com/triton-lang/triton),
and its integration with PyTorch.


## Installation

The benchmark suite should be self-contained of its dependencies. To install, follow the steps below.


Step 1: clone the repository and checkout all submodules

```
$ git clone https://github.com/pytorch-labs/tritonbench.git
$ git submodule update --init --recursive
```

Step 2: run install.py

```
$ python install.py
```

By default, it will install the latest PyTorch nightly release and use the Triton version bundled with it.

## Basic Usage

To benchmark an operator, use the following command:

```
$ python run.py --op gemm
```

## Submodules

We depend on the following projects as a source of customized Triton or CUTLASS kernels:

* (Required) [FBGEMM](https://github.com/pytorch/FBGEMM)
* (Required) [kernels](https://github.com/triton-lang/kernels)
* (Required) [generative-recommenders](https://github.com/facebookresearch/generative-recommenders)
* (Optional) [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
* (Optional) [cutlass-kernels](https://github.com/ColfaxResearch/cutlass-kernels)
* (Optional) [flash-attention](https://github.com/Dao-AILab/flash-attention)


## License
TritonBench is BSD 3-Clause licensed, as found in the LICENSE file.
