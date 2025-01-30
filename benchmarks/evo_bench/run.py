"""
Run benchmark and print tflops metrics
"""

import tritonbench
from tritonbench.utils.parser import get_parser

evo_bench_config = {
    "flash_attention_fwd": [
        "--op",
        "flash_attention",
        "--num-inputs",
        "1",
        "--input-id",
        "4",
    ],
    "flash_attention_bwd": [
        "--op",
        "flash_attention",
        "--num-inputs",
        "1",
        "--input-id",
        "4",
        "--bwd",
    ],
    "jagged_layer_norm": [
        "--op",
        "jagged_layer_norm",
        "--num-inputs",
        "1",
        "--input-id",
        "0",
    ],
    "ragged_attention_fwd": [
        "--op",
        "ragged_attention",
        "--num-inputs",
        "1",
        "--input-id",
        "4",
    ],
    "ragged_attention_bwd": [
        "--op",
        "ragged_attention",
        "--num-inputs",
        "1",
        "--input-id",
        "4",
        "--bwd",
    ],
}

def run():
    from tritonbench.utils.run_utils import run_in_task
    for op_name in evo_bench_config:
        op_args = evo_bench_config[op_name]
        run_in_task(op=op_name, op_args=op_args)

if __name__ == "__main__":
    run()
