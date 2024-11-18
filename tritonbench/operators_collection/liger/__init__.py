liger_operators = [
    "embedding",
    "rms_norm",
    "rope",
    "jsd",
    "fused_linear_jsd",
    "cross_entropy",
    "fused_linear_cross_entropy",
    "geglu",
    "gather_gemv",
    "kl_div",
    "swiglu",
]


def get_operators():
    return liger_operators
