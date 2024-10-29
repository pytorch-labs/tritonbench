import os
import sys

from pathlib import Path

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
SUBMODULE_PATH = REPO_PATH.joinpath("submodules")


class add_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


class add_ld_library_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.os_environ = os.environ.copy()
        library_path = os.environ.get("LD_LIBRARY_PATH")
        if not library_path:
            os.environ["LD_LIBRARY_PATH"] = self.path
        else:
            os.environ["LD_LIBRARY_PATH"] = f"{library_path}:{self.path}"

    def __exit__(self, exc_type, exc_value, traceback):
        os.environ = self.os_environ.copy()
