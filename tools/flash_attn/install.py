import os
import subprocess
import sys

from pathlib import Path

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CUR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def patch_fa3():
    patches = ["hopper.patch"]
    for patch_file in patches:
        patch_file_path = os.path.join(CUR_DIR, patch_file)
        submodule_path = str(
            REPO_PATH.joinpath("submodules", "flash-attention").absolute()
        )
        try:
            subprocess.check_output(
                [
                    "patch",
                    "-p1",
                    "--forward",
                    "-i",
                    patch_file_path,
                    "-r",
                    "/tmp/rej",
                ],
                cwd=submodule_path,
            )
        except subprocess.SubprocessError as e:
            output_str = str(e.output)
            if "previously applied" in output_str:
                return
            else:
                print(str(output_str))
                sys.exit(1)


def install_fa3():
    patch_fa3()
    FA3_PATH = REPO_PATH.joinpath("submodules", "flash-attention", "hopper")
    env = os.environ.copy()
    # limit nvcc memory usage on the CI machine
    env["MAX_JOBS"] = "8"
    env["NVCC_THREADS"] = "1"
    cmd = ["pip", "install", "-e", "."]
    subprocess.check_call(cmd, cwd=str(FA3_PATH.resolve()), env=env)
