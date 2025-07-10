from typing import Dict

from tritonbench.operator_loader.aten import get_aten_loader_cls_by_name, list_aten_ops

OP_LOADERS = {
    "aten": get_aten_loader_cls_by_name,
}


def is_loader_op(op_name: str) -> bool:
    loader_ops = list_loader_operators()
    return op_name in loader_ops


def list_loader_operators() -> Dict[str, str]:
    """
    Return a list of loader operators and their category (e.g., aten).
    """
    aten_ops = {k: "aten" for k in list_aten_ops()}
    return aten_ops


def get_op_loader_bench_cls_by_name(op_name: str):
    loader_ops = list_loader_operators()
    if op_name not in loader_ops:
        raise ValueError(f"{op_name} is not found in the operator loader.")
    cls_loader = OP_LOADERS[loader_ops[op_name]]
    return cls_loader(op_name)
