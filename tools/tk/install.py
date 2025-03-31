import os
import subprocess
import sys
from pathlib import Path

import torch

CUDA_HOME = (
    "/usr/local/cuda" if "CUDA_HOME" not in os.environ else os.environ["CUDA_HOME"]
)
REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
TK_PATH = REPO_PATH.joinpath("submodules", "ThunderKittens")
PATCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tk.patch")

TORCH_BASE_PATH = Path(torch.__file__).parent


def patch_tk():
    try:
        subprocess.check_output(
            [
                "patch",
                "-p1",
                "--forward",
                "-i",
                PATCH_FILE,
                "-r",
                "/tmp/rej",
            ],
            cwd=TK_PATH,
        )
    except subprocess.SubprocessError as e:
        output_str = str(e.output)
        if "previously applied" in output_str:
            return
        else:
            print(str(output_str))
            sys.exit(1)


def test_tk_attn_h100_fwd():
    environ = os.environ.copy()
    if not environ.get("LD_LIBRARY_PATH"):
        environ["LD_LIBRARY_PATH"] = f"{CUDA_HOME}/lib64:{TORCH_BASE_PATH}/lib"
    else:
        environ["LD_LIBRARY_PATH"] = (
            f"{CUDA_HOME}/lib64:{TORCH_BASE_PATH}/lib:{environ['LD_LIBRARY_PATH']}"
        )
    cmd = [
        sys.executable,
        "-c",
        "import thunderkittens as tk; tk.mha_forward; tk.fp8_gemm",
    ]
    subprocess.check_call(cmd, env=environ)


def install_tk():
    patch_tk()
    cmd = [sys.executable, "setup.py", "install"]
    subprocess.check_call(cmd, cwd=TK_PATH)
    test_tk_attn_h100_fwd()
