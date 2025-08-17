"""
Simple NF4 Triton Dequantization Kernel
Minimal implementation to avoid compilation issues
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _nf4_dequantize_simple(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Simple NF4 dequantization kernel without complex indexing."""
    
    pid = tl.program_id(0)
    
    row = pid // blocks_per_row
    block_in_row = pid % blocks_per_row
    
    if row >= m:
        return
    
    col_start = block_in_row * 64
    if col_start >= n:
        return
    
    # Load scale factors
    absmax_idx = row * blocks_per_row + block_in_row
    absmax_val = tl.load(absmax_ptr + absmax_idx)
    
    absmax32_idx = row * absmax32_per_row + (block_in_row >> 2)
    absmax32_val = tl.load(absmax32_ptr + absmax32_idx)
    
    # Combined scale
    scale = absmax_val * 0.00787401574803149606 * absmax32_val
    
    # Base addresses
    qweight_base = row * (n >> 1) + (col_start >> 1)
    output_base = row * n + col_start
    
    # Process 32 bytes (64 nibbles) in simple loop
    for byte_idx in range(32):
        if col_start + byte_idx * 2 >= n:
            break
            
        # Load one byte
        packed_byte = tl.load(qweight_ptr + qweight_base + byte_idx)
        
        # Extract nibbles
        low_nibble = packed_byte & 0xF
        high_nibble = (packed_byte >> 4) & 0xF
        
        # NF4 lookup - simplified
        if low_nibble == 0: low_val = -1.0
        elif low_nibble == 1: low_val = -0.6961928009986877
        elif low_nibble == 2: low_val = -0.5250730514526367
        elif low_nibble == 3: low_val = -0.39491748809814453
        elif low_nibble == 4: low_val = -0.28444138169288635
        elif low_nibble == 5: low_val = -0.18477343022823334
        elif low_nibble == 6: low_val = -0.09105003625154495
        elif low_nibble == 7: low_val = 0.0
        elif low_nibble == 8: low_val = 0.07958029955625534
        elif low_nibble == 9: low_val = 0.16093020141124725
        elif low_nibble == 10: low_val = 0.24611230194568634
        elif low_nibble == 11: low_val = 0.33791524171829224
        elif low_nibble == 12: low_val = 0.44070982933044434
        elif low_nibble == 13: low_val = 0.5626170039176941
        elif low_nibble == 14: low_val = 0.7229568362236023
        else: low_val = 1.0
        
        if high_nibble == 0: high_val = -1.0
        elif high_nibble == 1: high_val = -0.6961928009986877
        elif high_nibble == 2: high_val = -0.5250730514526367
        elif high_nibble == 3: high_val = -0.39491748809814453
        elif high_nibble == 4: high_val = -0.28444138169288635
        elif high_nibble == 5: high_val = -0.18477343022823334
        elif high_nibble == 6: high_val = -0.09105003625154495
        elif high_nibble == 7: high_val = 0.0
        elif high_nibble == 8: high_val = 0.07958029955625534
        elif high_nibble == 9: high_val = 0.16093020141124725
        elif high_nibble == 10: high_val = 0.24611230194568634
        elif high_nibble == 11: high_val = 0.33791524171829224
        elif high_nibble == 12: high_val = 0.44070982933044434
        elif high_nibble == 13: high_val = 0.5626170039176941
        elif high_nibble == 14: high_val = 0.7229568362236023
        else: high_val = 1.0
        
        # Apply scale and store
        low_idx = byte_idx * 2
        high_idx = low_idx + 1
        
        if col_start + low_idx < n:
            tl.store(output_ptr + output_base + low_idx, low_val * scale)
        if col_start + high_idx < n:
            tl.store(output_ptr + output_base + high_idx, high_val * scale)


def triton_dequantize_nf4(module):
    """Main NF4 dequantization function using simple kernel."""
    weight = module.weight
    quant_state = weight.quant_state
    
    qweight = weight.data
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    dtype = quant_state.dtype
    device = qweight.device
    
    m = module.out_features
    n = module.in_features
    
    blocks_per_row = (n + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Handle tensor shapes
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
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Allocate output
    output = torch.empty((m, n), dtype=dtype, device=device)
    
    # Launch kernel
    total_blocks = m * blocks_per_row
    
    _nf4_dequantize_simple[(total_blocks,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        m, n,
        blocks_per_row,
        absmax32_per_row,
        num_warps=1,
        num_stages=1,
    )
    
    return output


def pure_torch_fallback(module):
    """Pure PyTorch implementation that definitely works."""
    import torch
    
    weight = module.weight
    quant_state = weight.quant_state
    
    qweight = weight.data
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
    
    # Reshape scales to [m, blocks_per_row] and [m, absmax32_per_row]
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
    
    # Convert to float for calculations
    absmax = absmax.float()
    absmax32 = absmax32.float()
    
    # Create output tensor
    output = torch.zeros((m, n), dtype=dtype, device=device)
    
    # Process each row
    for row_idx in range(m):
        row_qweight = qweight[row_idx]  # [n//2] bytes
        
        # Extract all nibbles for this row
        low_nibbles = (row_qweight & 0xF).to(torch.long)
        high_nibbles = ((row_qweight >> 4) & 0xF).to(torch.long)
        
        # Lookup NF4 values
        low_values = nf4_lut[low_nibbles]  # [n//2]
        high_values = nf4_lut[high_nibbles]  # [n//2]
        
        # Interleave low and high values
        row_values = torch.zeros(n, dtype=torch.float32, device=device)
        row_values[0::2] = low_values[:n//2]
        row_values[1::2] = high_values[:n//2]
        
        # Apply scale factors block by block
        for block_idx in range(blocks_per_row):
            col_start = block_idx * 64
            col_end = min(col_start + 64, n)
            
            # Get scale factors
            absmax_scale = absmax[row_idx, block_idx]
            absmax32_scale = absmax32[row_idx, block_idx // 4]
            
            # Combined scale factor
            scale = absmax_scale * 0.00787401574803149606 * absmax32_scale
            
            # Apply scale to this block
            row_values[col_start:col_end] *= scale
        
        # Store the dequantized row
        output[row_idx] = row_values.to(dtype)
    
    return output


def reset_triton_dequantize_state():
    """Reset any cached state."""
    pass