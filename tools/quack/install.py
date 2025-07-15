import os
import subprocess

from pathlib import Path


REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CURRENT_DIR = Path(os.path.abspath(__file__)).parent
QUACK_PATH = REPO_PATH.joinpath("submodules", "quack")


def install_quack():
    cmd = ["pip", "install", "-e", "."]
    subprocess.check_call(cmd, cwd=QUACK_PATH)
