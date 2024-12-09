import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from tools.cuda_utils import CUDA_VERSION_MAP, DEFAULT_CUDA_VERSION
from tools.git_utils import checkout_submodules
from tools.python_utils import pip_install_requirements
from tools.torch_utils import is_hip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_PATH = Path(os.path.abspath(__file__)).parent
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


def install_fa2(compile=False, ci=False):
    if compile:
        # compile from source (slow)
        env = os.environ.copy()
        # limit max jobs to save memory in CI
        if ci:
            env["MAX_JOBS"] = "4"
        FA2_PATH = REPO_PATH.joinpath("submodules", "flash-attention")
        cmd = [sys.executable, "setup.py", "install"]
        subprocess.check_call(cmd, cwd=str(FA2_PATH.resolve()), env=env)
    else:
        # Install the pre-built binary
        cmd = ["pip", "install", "flash-attn", "--no-build-isolation"]
        subprocess.check_call(cmd)


def install_fa3(ci=False):
    FA3_PATH = REPO_PATH.joinpath("submodules", "flash-attention", "hopper")
    env = os.environ.copy()
    # limit max jobs to save memory in CI
    if ci:
        env["MAX_JOBS"] = "4"
    cmd = [sys.executable, "setup.py", "install"]
    subprocess.check_call(cmd, cwd=str(FA3_PATH.resolve()), env=env)


def install_liger():
    # Liger-kernel has a conflict dependency `triton` with pytorch,
    # so we need to install it without dependencies
    cmd = ["pip", "install", "liger-kernel", "--no-deps"]
    subprocess.check_call(cmd)


def setup_hip(args: argparse.Namespace):
    # We have to disable all third-parties that donot support hip/rocm
    args.all = False
    args.liger = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--fbgemm", action="store_true", help="Install FBGEMM GPU")
    parser.add_argument(
        "--colfax", action="store_true", help="Install optional Colfax CUTLASS kernels"
    )
    parser.add_argument(
        "--fa2", action="store_true", help="Install optional flash_attention 2 kernels"
    )
    parser.add_argument(
        "--fa2-compile",
        action="store_true",
        help="Install optional flash_attention 2 kernels from source.",
    )
    parser.add_argument(
        "--fa3", action="store_true", help="Install optional flash_attention 3 kernels"
    )
    parser.add_argument("--jax", action="store_true", help="Install jax nightly")
    parser.add_argument("--tk", action="store_true", help="Install ThunderKittens")
    parser.add_argument("--liger", action="store_true", help="Install Liger-kernel")
    parser.add_argument("--xformers", action="store_true", help="Install xformers")
    parser.add_argument(
        "--all", action="store_true", help="Install all custom kernel repos"
    )
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--ci", action="store_true", help="Indicate running in CI environment.")
    args = parser.parse_args()

    if args.all and is_hip():
        setup_hip(args)

    # install framework dependencies
    pip_install_requirements("requirements.txt")
    # checkout submodules
    checkout_submodules(REPO_PATH)
    # install submodules
    if args.fbgemm or args.all:
        logger.info("[tritonbench] installing FBGEMM...")
        install_fbgemm()
    if args.fa2 or args.all:
        logger.info("[tritonbench] installing fa2 from source...")
        install_fa2(compile=True, ci=args.ci)
    if args.fa3 or args.all:
        logger.info("[tritonbench] installing fa3...")
        install_fa3(ci=args.ci)
    if args.colfax or args.all:
        logger.info("[tritonbench] installing colfax cutlass-kernels...")
        from tools.cutlass_kernels.install import install_colfax_cutlass

        install_colfax_cutlass()
    if args.jax or args.all:
        logger.info("[tritonbench] installing jax...")
        install_jax()
    if args.tk or args.all:
        logger.info("[tritonbench] installing thunderkittens...")
        from tools.tk.install import install_tk

        install_tk()
    if args.liger or args.all:
        logger.info("[tritonbench] installing liger-kernels...")
        install_liger()
    if args.xformers or args.all:
        logger.info("[tritonbench] installing xformers...")
        from tools.xformers.install import install_xformers

        install_xformers()
    logger.info("[tritonbench] installation complete!")
    # run tests to check installation
    if args.test:
        test_fbgemm()
