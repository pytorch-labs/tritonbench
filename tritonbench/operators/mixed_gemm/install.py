import torch
from tools.cuda_builder import build_cuda_kernel

def test():
    torch.load_library("mixed_gemm/kernels/w2a16_gemm.so")

def install():
    build_cuda_kernel("kernels/w2a16_gemm.cu")
    test()


if __name__ == "__main__":
    install()
    test()