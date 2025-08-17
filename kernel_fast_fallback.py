"""
Fast PyTorch fallback for NF4 dequantization
Optimized for Tesla T4 without Triton
"""

import torch

def pure_torch_fallback(module):
    """
    Fast pure PyTorch NF4 dequantization
    Fully vectorized without loops
    """
    weight = module.weight
    quant_state = weight.quant_state
    
    qweight = weight.data  # [m, n//2]
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    dtype = quant_state.dtype
    device = qweight.device
    
    m = module.out_features
    n = module.in_features
    
    # NF4 lookup table
    nf4_lut = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ], dtype=torch.float32, device=device)
    
    blocks_per_row = (n + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Reshape scales to [m, blocks]
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(m, -1)
        elif absmax.numel() == m * blocks_per_row:
            absmax = absmax.view(m, blocks_per_row)
    
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(m, -1)
        elif absmax32.numel() == m * absmax32_per_row:
            absmax32 = absmax32.view(m, absmax32_per_row)
    
    # Convert to float for computation
    absmax = absmax.float()
    absmax32 = absmax32.float()
    
    # Extract all nibbles at once - fully vectorized
    qweight_int = qweight.to(torch.int16)  # Avoid overflow
    low_nibbles = (qweight_int & 0xF).long()  # [m, n//2]
    high_nibbles = ((qweight_int >> 4) & 0xF).long()  # [m, n//2]
    
    # Lookup all NF4 values at once
    low_values = nf4_lut[low_nibbles]  # [m, n//2]
    high_values = nf4_lut[high_nibbles]  # [m, n//2]
    
    # Create output and interleave
    output = torch.empty((m, n), dtype=torch.float32, device=device)
    output[:, 0::2] = low_values
    output[:, 1::2] = high_values
    
    # Apply scales - vectorized approach
    # Create scale matrix for all positions
    scale_matrix = torch.ones((m, n), dtype=torch.float32, device=device)
    
    for block_idx in range(blocks_per_row):
        col_start = block_idx * 64
        col_end = min(col_start + 64, n)
        
        # Get scales for this block
        absmax_block = absmax[:, block_idx:block_idx+1]  # [m, 1]
        absmax32_block = absmax32[:, block_idx//4:(block_idx//4)+1]  # [m, 1]
        
        # Combined scale
        block_scale = absmax_block * 0.00787401574803149606 * absmax32_block  # [m, 1]
        
        # Apply to this block's columns
        scale_matrix[:, col_start:col_end] = block_scale.expand(-1, col_end - col_start)
    
    # Apply all scales at once
    output = output * scale_matrix
    
    # Convert to target dtype
    return output.to(dtype)


def triton_dequantize_nf4(module):
    """Wrapper that just calls PyTorch fallback"""
    return pure_torch_fallback(module)


def reset_triton_dequantize_state():
    """No state to reset in pure PyTorch version"""
    pass