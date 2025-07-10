from tritonbench.operator_loader import list_loader_operators


def get_aten_operators():
    loader_operators = list_loader_operators()
    return [k for k in loader_operators.keys() if loader_operators[k] == "aten"]
