"""
Optimized NF4 Dequantization V2
Target: 1.15x+ speedup by reducing overhead
"""

import torch
import torch.nn.functional as F

# Pre-computed global constants
_CACHE = {}

def _init_cache(device):
    """Initialize cached tensors for device."""
    if device not in _CACHE:
        _CACHE[device] = {
            'lut': torch.tensor([
                -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
            ], dtype=torch.float32, device=device),
            'scale_const': 0.00787401574803149606,
            'mask_low': 0xF,
            'mask_high_shift': 4
        }
    return _CACHE[device]


def triton_dequantize_nf4(module):
    """
    V2 Optimized NF4 dequantization
    Minimizes overhead through better memory access patterns
    """
    weight = module.weight
    quant_state = weight.quant_state
    
    # Extract parameters
    qweight = weight.data  # [m, n//2]
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    dtype = quant_state.dtype
    device = qweight.device
    
    m, n_half = qweight.shape
    n = n_half * 2
    
    # Get cached constants
    cache = _init_cache(device)
    nf4_lut = cache['lut']
    scale_const = cache['scale_const']
    
    # Compute dimensions
    blocks_per_row = (n + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Fast reshape without error checking for speed
    # Trust that the input is valid
    if absmax.numel() == m * blocks_per_row:
        absmax = absmax.view(m, blocks_per_row).float()
    else:
        absmax = absmax.view(1, blocks_per_row).expand(m, -1).float()
    
    if absmax32.numel() == m * absmax32_per_row:
        absmax32 = absmax32.view(m, absmax32_per_row).float()
    else:
        absmax32 = absmax32.view(1, absmax32_per_row).expand(m, -1).float()
    
    # Optimized nibble extraction
    # Work with uint8 directly
    qweight_u8 = qweight.view(torch.uint8) if qweight.dtype != torch.uint8 else qweight
    
    # Extract nibbles in one operation
    low_nibbles = (qweight_u8 & cache['mask_low']).long()
    high_nibbles = ((qweight_u8 >> cache['mask_high_shift']) & cache['mask_low']).long()
    
    # Batch lookup - use gather for speed
    low_values = torch.gather(nf4_lut.unsqueeze(0).expand(m, -1), 1, 
                              low_nibbles.view(m, -1))
    high_values = torch.gather(nf4_lut.unsqueeze(0).expand(m, -1), 1, 
                               high_nibbles.view(m, -1))
    
    # Ensure correct shape
    low_values = low_values.view(m, n_half)
    high_values = high_values.view(m, n_half)
    
    # Pre-allocate output for speed
    output = torch.empty((m, n), dtype=torch.float32, device=device)
    
    # Fast interleaving
    output[:, 0::2] = low_values
    output[:, 1::2] = high_values
    
    # Optimized scale application
    # Minimize loop overhead by processing multiple blocks at once
    if blocks_per_row == 1:
        # Fast path: single block
        output *= (absmax[:, 0:1] * scale_const * absmax32[:, 0:1])
    elif blocks_per_row <= 4:
        # Fast path: few blocks (common case)
        base_scale = scale_const * absmax32[:, 0:1]
        for i in range(blocks_per_row):
            col_start = i * 64
            col_end = min(col_start + 64, n)
            output[:, col_start:col_end] *= (absmax[:, i:i+1] * base_scale)
    else:
        # General case: minimize operations
        for abs32_idx in range(absmax32_per_row):
            block_start = abs32_idx * 4
            block_end = min(block_start + 4, blocks_per_row)
            
            # Pre-compute group scale once
            group_scale = scale_const * absmax32[:, abs32_idx:abs32_idx+1]
            
            # Vectorized application
            for block_idx in range(block_start, block_end):
                col_start = block_idx * 64
                col_end = min(col_start + 64, n)
                output[:, col_start:col_end] *= (absmax[:, block_idx:block_idx+1] * group_scale)
    
    # Efficient dtype conversion
    return output.to(dtype) if dtype != torch.float32 else output


def reset_triton_dequantize_state():
    """Clear cache."""
    global _CACHE
    _CACHE.clear()