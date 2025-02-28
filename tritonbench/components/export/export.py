"""
Serialize pickled tensors to directory.
"""
from pathlib import Path
import pickle

from typing import Callable, Any

def get_fn_output(fn_mode: str, fn: Callable):
    pass

def export_data(x_val: str, input: Any, fn_mode: str, fn: Callable, export_type: str, export_dir: str):
    # pickle naming convention
    # x_<x_val>-input.pkl
    # x_<x_val>-<fn_name>-fwd-output.pkl
    # x_<x_val>-<fn_name>-bwd-output.pkl
    export_path = Path(export_dir)
    assert export_path.exists(), f"Export path {export_dir} must exist."
    if export_type == "input" or export_type =="both":
        input_file_name = f"x_{x_val}-input.pkl"
        input_file_path = export_path.joinpath(input_file_name)
        with open(input_file_path, "w") as ifp:
            pickle.dump(ifp, input)
    if export_type == "output" or export_type == "both":
        output_file_name = f"x_{x_val}-{fn._name}-{fn_mode}-output.pkl"
        output_file_path = export_path.joinpath(output_file_name)
        output = get_fn_output(fn_mode, fn)
        with open(output_file_path, "w") as ofp:
            pickle.dump(ofp, output)
