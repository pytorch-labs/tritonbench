from tritonbench.operator_loader import list_loader_operators
from tritonbench.operators import list_operators


def get_all_operators():
    return list_operators() + list(list_loader_operators().keys())
