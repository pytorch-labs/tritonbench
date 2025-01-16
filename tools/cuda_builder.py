"""
Build customized CUDA or CUTLASS kernels hosted in Tritonbench repo.
"""

import os
import subprocess
import torch

from pathlib import Path

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent
TORCH_CUTLASS_PATH = REPO_PATH.joinpath("submodules", "cutlass")
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
TORCH_BASE_PATH = Path(torch.__file__).parent

COMPILER_FLAGS = [
    f"-I{CUDA_HOME}/include",
    f"-Wl,-rpath,'{CUDA_HOME}/lib64'",
    f"-Wl,-rpath,'{CUDA_HOME}/lib'",
]
NVCC_GENCODE = "-gencode=arch=compute_90a,code=[sm_90a]"

NVCC_FLAGS = [
    NVCC_GENCODE,
    "--use_fast_math",
    "-forward-unknown-to-host-compiler",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-forward-unknown-to-host-compiler",
    "--use_fast_math",
    "-Xcompiler=-fno-strict-aliasing",
    "-Xcompiler=-fPIE",
    "-Xcompiler=-lcuda",
    "-DNDEBUG",
    "-DCUTLASS_TEST_LEVEL=0",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
    "-DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
]

COMPILER_FLAGS = [
    f"-I{str(TORCH_CUTLASS_PATH.joinpath('include').resolve())}",
    f"-I{str(TORCH_CUTLASS_PATH.joinpath('examples', 'commmon').resolve())}",
    f"-I{str(TORCH_CUTLASS_PATH.joinpath('tools', 'util', 'include').resolve())}",
    f"-I{CUDA_HOME}/include",
    f"-Wl,-rpath,'{CUDA_HOME}/lib64'",
    f"-Wl,-rpath,'{CUDA_HOME}/lib'",
]
LINKER_FLAGS = [
    "--shared",
    "-fPIC",
    f"-L{str(TORCH_BASE_PATH.joinpath('lib').resolve())}",
    "-ltorch",
    "-ltorch_cuda",
    "-lc10",
    "-lc10_cuda",
    "-lcuda",
    "-lcudadevrt",
    "-lcudart_static",
    "-lcublas",
    "-lrt",
    "-lpthread",
    "-ldl",
]

def build_cuda_kernel(sources):
    # compile colfax_cutlass kernels
    output_dir = REPO_PATH.joinpath(".data", "cuda_kernels")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["nvcc"]
    cmd.extend(COMPILER_FLAGS)
    cmd.extend(NVCC_FLAGS)
    cmd.extend(sources)
    cmd.extend(LINKER_FLAGS)
    print(" ".join(cmd))
    print(str(output_dir.resolve()))
    subprocess.check_call(cmd, cwd=str(output_dir.resolve()))
    colfax_cutlass_lib = str(output_dir.joinpath(FMHA_SOURCES[-1]).resolve())
