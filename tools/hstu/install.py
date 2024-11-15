import os
import subprocess
import sys
from pathlib import Path

PATCH_DIR = str(
    Path(__file__)
    .parent.parent.parent.joinpath("submodules", "generative-recommenders")
    .absolute()
)
PATCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hstu.patch")


def install_hstu():
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
            cwd=PATCH_DIR,
        )
    except subprocess.SubprocessError as e:
        output_str = str(e.output)
        if "previously applied" in output_str:
            return
        else:
            print(str(output_str))
            sys.exit(1)
