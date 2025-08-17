"""
Extreme Optimization for NF4 Dequantization
Target: 1.15x+ speedup on Tesla T4
"""

import torch

# Global cache for lookup table and buffers
_CACHE = {}

def triton_dequantize_nf4(module):
    """
    Extreme optimized NF4 dequantization
    Every nanosecond counts
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
    
    # Cache key for reusable tensors
    cache_key = (device, m, n)
    
    # Get or create cached tensors
    if cache_key not in _CACHE:
        _CACHE[cache_key] = {
            'lut': torch.tensor([
                -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
            ], dtype=torch.float32, device=device),
            'output': torch.empty((m, n), dtype=torch.float32, device=device),
            'low_mask': 0xF,
            'high_shift': 4
        }
    
    cache = _CACHE[cache_key]
    nf4_lut = cache['lut']
    output = cache['output']
    
    # Constants computed once
    blocks_per_row = (n + 63) >> 6  # Faster than // 64
    absmax32_per_row = (blocks_per_row + 3) >> 2  # Faster than // 4
    n_half = n >> 1  # Faster than // 2
    
    # Ultra-fast reshape - no safety checks
    absmax_numel = absmax.numel()
    absmax32_numel = absmax32.numel()
    
    # Direct reshape based on size
    if absmax_numel == m * blocks_per_row:
        absmax_reshaped = absmax.view(m, blocks_per_row).float()
    else:
        absmax_reshaped = absmax.view(1, blocks_per_row).expand(m, -1).float()
    
    if absmax32_numel == m * absmax32_per_row:
        absmax32_reshaped = absmax32.view(m, absmax32_per_row).float()
    else:
        absmax32_reshaped = absmax32.view(1, absmax32_per_row).expand(m, -1).float()
    
    # Fastest nibble extraction - minimize type conversions
    # Use byte operations directly when possible
    if qweight.dtype == torch.uint8:
        # Work directly with uint8
        low_nibbles = (qweight & cache['low_mask']).long()
        high_nibbles = ((qweight >> cache['high_shift']) & cache['low_mask']).long()
    else:
        # Convert once
        qweight_bytes = qweight.byte()
        low_nibbles = (qweight_bytes & cache['low_mask']).long()
        high_nibbles = ((qweight_bytes >> cache['high_shift']) & cache['low_mask']).long()
    
    # Lookup values - use advanced indexing
    low_values = torch.index_select(nf4_lut, 0, low_nibbles.view(-1)).view(m, n_half)
    high_values = torch.index_select(nf4_lut, 0, high_nibbles.view(-1)).view(m, n_half)
    
    # Fastest interleaving using strided copy
    output[:, 0::2] = low_values
    output[:, 1::2] = high_values
    
    # Extreme scale optimization
    # Precompute constant
    SCALE_CONST = 0.00787401574803149606
    
    # Special case optimizations
    if blocks_per_row == 1:
        # Single block - one multiplication
        scale = absmax_reshaped[:, 0:1] * SCALE_CONST * absmax32_reshaped[:, 0:1]
        output *= scale
    elif blocks_per_row == 2:
        # Two blocks - unroll completely
        scale0 = absmax_reshaped[:, 0:1] * SCALE_CONST * absmax32_reshaped[:, 0:1]
        scale1 = absmax_reshaped[:, 1:2] * SCALE_CONST * absmax32_reshaped[:, 0:1]
        output[:, 0:64] *= scale0
        output[:, 64:min(128, n)] *= scale1
    elif blocks_per_row <= 4:
        # Up to 4 blocks - partial unroll
        base_scale = SCALE_CONST * absmax32_reshaped[:, 0:1]
        for i in range(blocks_per_row):
            col_start = i << 6  # i * 64
            col_end = min(col_start + 64, n)
            output[:, col_start:col_end] *= (absmax_reshaped[:, i:i+1] * base_scale)
    else:
        # General case - minimize inner loop overhead
        # Process entire absmax32 groups at once
        for abs32_idx in range(absmax32_per_row):
            block_start = abs32_idx << 2  # abs32_idx * 4
            block_end = min(block_start + 4, blocks_per_row)
            
            # Precompute group scale
            group_scale = SCALE_CONST * absmax32_reshaped[:, abs32_idx:abs32_idx+1]
            
            # Unroll inner loop for common case (4 blocks)
            if block_end - block_start == 4:
                for i in range(4):
                    block_idx = block_start + i
                    col_start = block_idx << 6
                    col_end = min(col_start + 64, n)
                    output[:, col_start:col_end] *= (absmax_reshaped[:, block_idx:block_idx+1] * group_scale)
            else:
                # Handle remainder
                for block_idx in range(block_start, block_end):
                    col_start = block_idx << 6
                    col_end = min(col_start + 64, n)
                    output[:, col_start:col_end] *= (absmax_reshaped[:, block_idx:block_idx+1] * group_scale)
    
    # Minimize dtype conversion overhead
    if dtype == torch.float32:
        return output
    elif dtype == torch.float16:
        return output.half()
    elif dtype == torch.bfloat16:
        return output.bfloat16()
    else:
        return output.to(dtype)


def reset_triton_dequantize_state():
    """Clear cache to free memory."""
    global _CACHE
    _CACHE.clear()