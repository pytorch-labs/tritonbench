import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def jagged_layer_norm_kernel(
    values_ptr,  # pointer to values tensor
    offsets_ptr,  # pointer to offsets tensor
    output_ptr,  # pointer to output tensor
    B,  # batch size
    M,  # feature dimension
    eps,  # epsilon for numerical stability
    BLOCK_M: tl.constexpr,  # block size for M dimension
):
    # Program ID is the batch index
    batch_idx = tl.program_id(0)
    
    # Load the start and end offsets for this batch
    start_offset = tl.load(offsets_ptr + batch_idx)
    end_offset = tl.load(offsets_ptr + batch_idx + 1)
    seq_len = end_offset - start_offset
    
    # Early exit if sequence length is 0
    if seq_len == 0:
        return
    
    # Compute mean across the sequence and feature dimensions
    mean_acc = tl.zeros([1], dtype=tl.float32)
    
    # First pass: compute mean
    for seq_idx in range(seq_len):
        row_idx = start_offset + seq_idx
        for m_start in range(0, M, BLOCK_M):
            m_offs = m_start + tl.arange(0, BLOCK_M)
            mask = m_offs < M
            
            # Load values
            vals = tl.load(
                values_ptr + row_idx * M + m_offs,
                mask=mask,
                other=0.0
            ).to(tl.float32)
            
            # Accumulate sum
            mean_acc += tl.sum(vals, axis=0)
    
    # Compute mean
    total_elements = seq_len * M
    mean = mean_acc / total_elements
    
    # Second pass: compute variance
    var_acc = tl.zeros([1], dtype=tl.float32)
    
    for seq_idx in range(seq_len):
        row_idx = start_offset + seq_idx
        for m_start in range(0, M, BLOCK_M):
            m_offs = m_start + tl.arange(0, BLOCK_M)
            mask = m_offs < M
            
            # Load values
            vals = tl.load(
                values_ptr + row_idx * M + m_offs,
                mask=mask,
                other=0.0
            ).to(tl.float32)
            
            # Compute squared differences
            diff = vals - mean
            var_acc += tl.sum(diff * diff, axis=0)
    
    # Compute variance and standard deviation
    var = var_acc / total_elements
    std = tl.sqrt(var + eps)
    
    # Third pass: normalize and write output
    for seq_idx in range(seq_len):
        row_idx = start_offset + seq_idx
        for m_start in range(0, M, BLOCK_M):
            m_offs = m_start + tl.arange(0, BLOCK_M)
            mask = m_offs < M
            
            # Load values
            vals = tl.load(
                values_ptr + row_idx * M + m_offs,
                mask=mask,
                other=0.0
            ).to(tl.float32)
            
            # Normalize
            normalized = (vals - mean) / std
            
            # Store output
            tl.store(
                output_ptr + row_idx * M + m_offs,
                normalized,
                mask=mask
            )


def triton_jagged_layer_norm(values: torch.Tensor, offsets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Performs layer normalization on a jagged tensor.
    
    Args:
        values: The values tensor of shape (total_length, M) where total_length is the sum of all sequence lengths
        offsets: The offsets tensor of shape (B+1,) indicating the start/end indices for each batch
        eps: Small value for numerical stability
        
    Returns:
        Normalized values tensor of the same shape as input
    """
    B = offsets.numel() - 1
    M = values.shape[1]
    
    # Allocate output tensor
    output = torch.empty_like(values)
    
    # Determine block size
    BLOCK_M = min(triton.next_power_of_2(M), 1024)
    
    # Launch kernel
    jagged_layer_norm_kernel[(B,)](
        values,
        offsets,
        output,
        B,
        M,
        eps,
        BLOCK_M=BLOCK_M,
    )
    
    return output


# Optimized version with better memory access patterns
@triton.jit
def jagged_layer_norm_kernel_v2(
    values_ptr,  # pointer to values tensor
    offsets_ptr,  # pointer to offsets tensor  
    output_ptr,  # pointer to output tensor
    mean_ptr,  # pointer to store means (for potential gradient computation)
    rstd_ptr,  # pointer to store reciprocal std (for potential gradient computation)
    B,  # batch size
    M,  # feature dimension
    eps,  # epsilon for numerical stability
    BLOCK_M: tl.constexpr,  # block size for M dimension
):
    # Program ID is the batch index
    batch_idx = tl.program_id(0)
    
    # Load the start and end offsets for this batch
    start_offset = tl.load(offsets_ptr + batch_idx)
    end_offset = tl.load(offsets_ptr + batch_idx + 1)
    seq_len = end_offset - start_offset
    
    # Early exit if sequence length is 0
    if seq_len == 0:
        return
    
    # Process in blocks along M dimension for better cache usage
    m_block_idx = tl.program_id(1)
    m_start = m_block_idx * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    
    # First pass: compute partial sums for mean
    partial_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    for seq_idx in range(seq_len):
        row_idx = start_offset + seq_idx
        vals = tl.load(
            values_ptr + row_idx * M + m_offs,
            mask=m_mask,
            other=0.0
        ).to(tl.float32)
        partial_sum += vals
    
    # Reduce across blocks to get mean (simplified - in practice would need atomic ops)
    mean = tl.sum(partial_sum) / (seq_len * M)
    
    # Second pass: compute partial variance
    partial_var = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    for seq_idx in range(seq_len):
        row_idx = start_offset + seq_idx
        vals = tl.load(
            values_ptr + row_idx * M + m_offs,
            mask=m_mask,
            other=0.0
        ).to(tl.float32)
        diff = vals - mean
        partial_var += diff * diff
    
    # Compute variance and reciprocal std
    var = tl.sum(partial_var) / (seq_len * M)
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd if provided
    if mean_ptr:
        tl.store(mean_ptr + batch_idx, mean)
    if rstd_ptr:
        tl.store(rstd_ptr + batch_idx, rstd)
    
    # Third pass: normalize
    for seq_idx in range(seq_len):
        row_idx = start_offset + seq_idx
        vals = tl.load(
            values_ptr + row_idx * M + m_offs,
            mask=m_mask,
            other=0.0
        ).to(tl.float32)
        
        normalized = (vals - mean) * rstd
        
        tl.store(
            output_ptr + row_idx * M + m_offs,
            normalized,
            mask=m_mask
        )