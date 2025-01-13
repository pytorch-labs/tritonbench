import tritonbench
from tritonbench.utils.parser import get_parser
import numpy as np
import torch

def run():
    # Define benchmark parameters
    args = [
        "--B",           # Batch size
        "32",
        "--M",          # Hidden dimension size
        "4096",
        "--seqlen",     # Maximum sequence length
        "4096",
        "--sparsity",   # Optional sparsity parameter
        "0.0",
        "--sum-then-buffer",  # Whether to sum before buffering (1 for True, 0 for False)
        "0",
    ]

    # Get parser and parse arguments
    parser = get_parser()
    parsed_args, extra_args = parser.parse_known_args(args)
    
    # Convert namespace to dictionary for easier access
    args_dict = vars(parsed_args)
    
    # Print parsed arguments for debugging
    # print("Parsed arguments:", args_dict)
    
    try:
        # Load the jagged sum benchmark operation
        jagged_sum_op = tritonbench.load_opbench_by_name("jagged_sum")
        
        # Create benchmark instance with parsed arguments
        jagged_sum_bench = jagged_sum_op(parsed_args, extra_args)
        jagged_sum_bench.run()
        
        # Print results
        # print("Benchmark results:")
        print(jagged_sum_bench.output)
        
    except Exception as e:
        print(f"Error during benchmark execution: {str(e)}")
        raise

if __name__ == "__main__":
    run()