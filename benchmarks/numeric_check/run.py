import argparse
import pickle
import os
import torch

from typing import Any
from pathlib import Path

def find_pkl_files(path: Path):
    abs_path = path.absolute()
    return [f for f in os.listdir(abs_path) if os.path.isfile(os.path.join(abs_path, f)) and f.endswith(".pkl")]

def common_pkl_files(pkl_files_a, pkl_files_b):
    set_a = set(pkl_files_a)
    set_b = set(pkl_files_b)
    return list(set_a.intersection(set_b)), list(set_a - set_b), list(set_b - set_a)

def check_tensor_numeric(a, b):
    assert isinstance(a, tuple), "Out A must be a tuple."
    assert isinstance(b, tuple), "Out B must be a tuple."
    assert len(a) == len(b), f"A and B must be equal length, but len_a={len(a)}, len_b={len(b)}"
    for i in range(len(a)):
        tensor_a = a[i]
        tensor_b = b[i]
        torch.testing.assert_close(tensor_a, tensor_b)

def load_data_from_pickle_file(pickle_file_path) -> Any:
    with open(pickle_file_path, "rb") as pfp:
        data = pickle.load(pfp)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", help="Side A of the output.")
    parser.add_argument("--b", help="Side B of the output.")
    args = parser.parse_args()
    pkl_files_a = find_pkl_files(args.a)
    pkl_files_b = find_pkl_files(args.b)
    common_files, a_only_files, b_only_files = common_pkl_files(pkl_files_a, pkl_files_b)
    for common_file in common_files:
        data_a = load_data_from_pickle_file(os.path.join(args.a, common_file))
        data_b = load_data_from_pickle_file(os.path.join(args.b, common_file))
        check_tensor_numeric(data_a, data_b)
    print("A and B sides numeric match.")
