"""
Serialize pickled tensors to directory.
"""
from pathlib import Path
import pickle
from tritonbench.utils.input import input_cast

from typing import Callable, Any

def get_input_gradients(inputs):
    all_inputs = []
    input_cast(lambda x: True, lambda y: all_inputs.append(y), inputs)
    return [x.grad for x in all_inputs]

def export_data(x_val: str, inputs: Any, fn_mode: str, fn: Callable, export_type: str, export_dir: str):
    # pickle naming convention
    # x_<x_val>-input.pkl
    # x_<x_val>-<fn_name>-fwd-output.pkl
    # x_<x_val>-<fn_name>-bwd-output.pkl
    assert export_dir, f"Export dir must be specified."
    export_path = Path(export_dir)
    assert export_path.exists(), f"Export path {export_dir} must exist."
    if export_type == "input" or export_type =="both":
        input_file_name = f"x_{x_val}-input.pkl"
        input_file_path = export_path.joinpath(input_file_name)
        with open(input_file_path, "wb") as ifp:
            pickle.dump(inputs, ifp)
    if export_type == "output" or export_type == "both":
        output_file_name = f"x_{x_val}-{fn._name}-{fn_mode}-output.pkl"
        output_file_path = export_path.joinpath(output_file_name)
        if fn_mode == "fwd":
            output = fn()
        elif fn_mode == "bwd":
            # output of the backward pass are the input gradients
            output = get_input_gradients(inputs)
        with open(output_file_path, "wb") as ofp:
            pickle.dump(output, ofp)
