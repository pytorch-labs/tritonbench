"""
Run flash_attention benchmark with a single input shape:
BATCH: 4

SEQ_LEN: 16384

Print tflops metrics
"""

import tritonbench
from tritonbench.utils.parser import get_parser


def run():
    args = ["--batch", "4", "--seq-len", "16384", "--n-heads", "32", "--d-head", "64", "--precision", "bf16", "--bwd", "--only", "triton_tutorial_flash_v2", "--causal", "--metrics", "tflops"]
    flash_attn_op = tritonbench.load_opbench_by_name("flash_attention")
    parser = get_parser()
    args, extra_args = parser.parse_known_args(args)
    flash_attn_bench = flash_attn_op(args, extra_args)
    flash_attn_bench.run()
    print(flash_attn_bench.output)

if __name__ == "__main__":
    run()
