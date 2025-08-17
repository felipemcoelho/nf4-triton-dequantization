"""
Fastest NF4 Dequantization Implementation
Uses torch tensor operations optimally
"""

import torch

# Pre-allocated global tensors
_GLOBAL_LUT = None
_LAST_DEVICE = None

def triton_dequantize_nf4(module):
    """
    Fastest possible NF4 dequantization
    Optimized for Tesla T4's specific characteristics
    """
    global _GLOBAL_LUT, _LAST_DEVICE
    
    weight = module.weight
    quant_state = weight.quant_state
    
    qweight = weight.data  # [m, n//2]
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    dtype = quant_state.dtype
    device = qweight.device
    
    m = module.out_features
    n = module.in_features
    
    # Cache LUT on device
    if _LAST_DEVICE != device:
        _GLOBAL_LUT = torch.tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ], dtype=torch.float32, device=device)
        _LAST_DEVICE = device
    
    # Bit manipulation constants
    blocks_per_row = (n + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Reshape scales with minimal overhead
    total_blocks = m * blocks_per_row
    if absmax.numel() == total_blocks:
        absmax = absmax.view(m, blocks_per_row)
    elif absmax.numel() == blocks_per_row:
        absmax = absmax.view(1, -1).expand(m, -1)
    
    total_absmax32 = m * absmax32_per_row
    if absmax32.numel() == total_absmax32:
        absmax32 = absmax32.view(m, absmax32_per_row)
    elif absmax32.numel() == absmax32_per_row:
        absmax32 = absmax32.view(1, -1).expand(m, -1)
    
    # Convert to float32 for computation
    absmax_f = absmax.to(torch.float32)
    absmax32_f = absmax32.to(torch.float32)
    
    # Extract nibbles with minimal overhead
    # Use the smallest integer type that works
    qweight_int = qweight.to(torch.uint8) if qweight.dtype != torch.uint8 else qweight
    
    # Vectorized nibble extraction
    low_nibbles = (qweight_int & 0xF)
    high_nibbles = (qweight_int >> 4)
    
    # Batch lookup - faster than individual indexing
    all_nibbles = torch.cat([low_nibbles.view(-1), high_nibbles.view(-1)])
    all_values = _GLOBAL_LUT[all_nibbles.long()]
    
    # Split back into low and high
    split_point = low_nibbles.numel()
    low_values = all_values[:split_point].view(m, n // 2)
    high_values = all_values[split_point:].view(m, n // 2)
    
    # Create output with optimal memory layout
    output = torch.empty((m, n), dtype=torch.float32, device=device)
    
    # Interleave using the fastest method for Tesla T4
    # Slice assignment is optimized in PyTorch
    output[:, 0::2] = low_values
    output[:, 1::2] = high_values
    
    # Apply scales with minimal operations
    # Combine scales first, then apply once
    scale_factor = 0.00787401574803149606
    
    # Vectorized scale application
    for abs32_idx in range(absmax32_per_row):
        # Get blocks for this absmax32 group
        block_start = abs32_idx * 4
        block_end = min(block_start + 4, blocks_per_row)
        
        if block_start >= blocks_per_row:
            break
        
        # Precompute combined scale for this group
        group_absmax32 = absmax32_f[:, abs32_idx:abs32_idx+1] * scale_factor
        
        # Apply to all blocks in group at once if possible
        if block_end - block_start == 1:
            # Single block
            col_start = block_start * 64
            col_end = min(col_start + 64, n)
            output[:, col_start:col_end] *= (absmax_f[:, block_start] * group_absmax32).unsqueeze(1)
        else:
            # Multiple blocks - vectorize as much as possible
            for block_idx in range(block_start, block_end):
                col_start = block_idx * 64
                col_end = min(col_start + 64, n)
                combined_scale = absmax_f[:, block_idx:block_idx+1] * group_absmax32
                output[:, col_start:col_end] *= combined_scale
    
    # Convert to target dtype efficiently
    if dtype == torch.float32:
        return output
    elif dtype == torch.float16:
        return output.half()
    else:
        return output.to(dtype)


def reset_triton_dequantize_state():
    """Reset global state."""
    global _GLOBAL_LUT, _LAST_DEVICE
    _GLOBAL_LUT = None
    _LAST_DEVICE = None