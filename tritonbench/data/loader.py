import importlib
from typing import Any


def get_input_loader(tritonbench_op: Any, op: str, input: str):
    if hasattr(tritonbench_op, "aten_op_name"):
        loader_type = "aten"
    else:
        raise RuntimeError(f"Unsupported op: {op}")

    generator_module = importlib.import_module(
        f".input_loaders.{loader_type}", package=__package__
    )
    input_iter_getter = generator_module.get_input_iter
    input_iter = input_iter_getter(tritonbench_op, op, input)
    return input_iter
