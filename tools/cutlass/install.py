import os
import subprocess

from pathlib import Path

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CUTLASS_PATH = REPO_PATH.joinpath("submodules", "cutlass")

def install_cutlass():
    cmd = ["pip", "install", "-e", "."]
    subprocess.check_call(cmd, cwd=str(CUTLASS_PATH.resolve()))
