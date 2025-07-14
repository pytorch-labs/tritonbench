import torch
import torch.nn.functional as F
from triton_jagged_layer_norm import triton_jagged_layer_norm

# Test the Triton kernel
def test_triton_jagged_layer_norm():
    # Create test data
    B = 4  # batch size
    M = 128  # feature dimension
    
    # Create random sequence lengths
    seq_lengths = torch.tensor([10, 15, 8, 12])
    total_length = seq_lengths.sum().item()
    
    # Create offsets tensor
    offsets = torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)])
    
    # Create random values
    values = torch.randn(total_length, M, device='cuda', dtype=torch.float32)
    
    # Run Triton kernel
    output_triton = triton_jagged_layer_norm(values, offsets, eps=1e-6)
    
    # Compare with PyTorch reference
    # For each sequence, apply layer norm separately
    output_ref = torch.empty_like(values)
    for i in range(B):
        start = offsets[i].item()
        end = offsets[i+1].item()
        if start < end:
            seq_values = values[start:end]
            # Layer norm across both sequence and feature dimensions
            normalized = F.layer_norm(seq_values, seq_values.shape, eps=1e-6)
            output_ref[start:end] = normalized
    
    # Check if outputs match
    max_diff = (output_triton - output_ref).abs().max().item()
    print(f"Maximum difference between Triton and PyTorch: {max_diff}")
    
    # Check if they're close
    if torch.allclose(output_triton, output_ref, atol=1e-5, rtol=1e-5):
        print("✓ Triton kernel produces correct results!")
    else:
        print("✗ Triton kernel results don't match PyTorch reference")
        
    # Print some statistics
    print(f"\nTest configuration:")
    print(f"  Batch size: {B}")
    print(f"  Feature dimension: {M}")
    print(f"  Sequence lengths: {seq_lengths.tolist()}")
    print(f"  Total elements: {total_length * M}")

if __name__ == "__main__":
    test_triton_jagged_layer_norm()