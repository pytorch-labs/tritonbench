import os
import subprocess

from pathlib import Path


REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
AITER_PATH = REPO_PATH.joinpath("submodules", "aiter")

def install_aiter():
    cmd = ["python", "setup.py", "develop"]
    subprocess.check_call(cmd, cwd=AITER_PATH)
