from .path_utils import REPO_PATH


def load_library(library_path: str):
    import torch

    prefix, _delimiter, so_file = library_path.partition("/")
    so_full_path = REPO_PATH.joinpath("utils", prefix, ".data", so_file).resolve()
    torch.ops.load_library(str(so_full_path))
