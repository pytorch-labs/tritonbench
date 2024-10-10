import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from utils.cuda_utils import CUDA_VERSION_MAP, DEFAULT_CUDA_VERSION
from utils.python_utils import pip_install_requirements
from utils.git_utils import checkout_submodules

REPO_PATH = Path(os.path.abspath(__file__)).parent
FA3_PATH = REPO_PATH.joinpath("submodules", "flash-attention", "hopper")
FBGEMM_PATH = REPO_PATH.joinpath("submodules", "FBGEMM", "fbgemm_gpu")

def install_jax(cuda_version=DEFAULT_CUDA_VERSION):
    jax_package_name = CUDA_VERSION_MAP[cuda_version]["jax"]
    jax_nightly_html = (
        "https://storage.googleapis.com/jax-releases/jax_nightly_releases.html"
    )
    # install instruction:
    # https://jax.readthedocs.io/en/latest/installation.html
    # pip install -U --pre jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
    cmd = ["pip", "install", "--pre", jax_package_name, "-f", jax_nightly_html]
    subprocess.check_call(cmd)
    # Test jax installation
    test_cmd = [sys.executable, "-c", "import jax"]
    subprocess.check_call(test_cmd)


def install_fbgemm():
    cmd = ["pip", "install", "-r", "requirements.txt"]
    subprocess.check_call(cmd, cwd=str(FBGEMM_PATH.resolve()))
    # Build target A100(8.0) or H100(9.0, 9.0a)
    cmd = [
        sys.executable,
        "setup.py",
        "install",
        "--package_variant=genai",
        "-DTORCH_CUDA_ARCH_LIST=8.0;9.0;9.0a",
    ]
    subprocess.check_call(cmd, cwd=str(FBGEMM_PATH.resolve()))


def test_fbgemm():
    print("Checking fbgemm_gpu installation...", end="")
    cmd = [sys.executable, "-c", "import fbgemm_gpu.experimental.gen_ai"]
    subprocess.check_call(cmd)
    print("OK")


def install_cutlass():
    from utils.cutlass_kernels.install import install_colfax_cutlass
    install_colfax_cutlass()


def install_fa():
    cmd = [sys.executable, "setup.py", "install"]
    subprocess.check_call(cmd, cwd=str(FA3_PATH.resolve()))


def install_liger():
    # Liger-kernel has a conflict dependency `triton` with pytorch,
    # so we need to install it without dependencies
    cmd = ["pip", "install", "liger-kernel", "--no-deps"]
    subprocess.check_call(cmd)


def install_tk():
    from utils.tk.install import install_tk
    install_tk()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--fbgemm", action="store_true", help="Install FBGEMM GPU")
    parser.add_argument(
        "--cutlass", action="store_true", help="Install optional CUTLASS kernels"
    )
    parser.add_argument(
        "--fa", action="store_true", help="Install optional flash_attention kernels"
    )
    parser.add_argument("--jax", action="store_true", help="Install jax nightly")
    parser.add_argument("--tk", action="store_true", help="Install ThunderKittens")
    parser.add_argument("--liger", action="store_true", help="Install Liger-kernel")
    parser.add_argument("--all", action="store_true", help="Install all custom kernel repos")
    parser.add_argument("--test", action="store_true", help="Run tests")
    args = parser.parse_args()

    # install framework dependencies
    pip_install_requirements("requirements.txt")
    # checkout submodules
    checkout_submodules(REPO_PATH)
    if args.fbgemm or args.all:
        logging.info("[tritonbench] installing FBGEMM...")
        install_fbgemm()
    if args.fa or args.all:
        logging.info("[tritonbench] installing flash-attn and fa3...")
        install_fa()
    if args.cutlass or args.all:
        logging.info("[tritonbench] installing cutlass-kernels...")
        install_cutlass()
    if args.jax or args.all:
        logging.info("[tritonbench] installing jax...")
        install_jax()
    if args.tk or args.all:
        logging.info("[tritonbench] installing thunderkittens...")
        install_tk()
    if args.liger or args.all:
        logging.info("[tritonbench] installing liger-kernels...")
        install_liger()
    # Run tests to check installation
    if args.test:
        test_fbgemm()
