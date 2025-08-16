"""
Minimal NF4 Triton Dequantization Kernel
Simplified to isolate performance issues
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _nf4_dequantize_kernel_simple(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Simplest possible NF4 dequantization kernel."""
    
    pid = tl.program_id(0)
    
    # Each program handles one NF4 block (64 elements)
    row = pid // blocks_per_row
    block_idx = pid % blocks_per_row
    
    if row >= m:
        return
    
    col_start = block_idx * 64
    if col_start >= n:
        return
    
    # Load scales (double dequantization)
    absmax = tl.load(absmax_ptr + row * blocks_per_row + block_idx).to(tl.float32)
    absmax32 = tl.load(absmax32_ptr + row * absmax32_per_row + (block_idx >> 2)).to(tl.float32)
    scale = absmax * 0.00787401574803149606 * absmax32
    
    # Base addresses
    qweight_base = row * (n >> 1) + (col_start >> 1)
    output_base = row * n + col_start
    
    # Process 32 bytes (64 nibbles)
    for i in tl.static_range(32):
        # Load one byte
        packed = tl.load(qweight_ptr + qweight_base + i)
        
        # Extract nibbles
        low = packed & 0xF
        high = (packed >> 4) & 0xF
        
        # NF4 decode - simple if-else chain
        # Low nibble
        if low == 0:
            low_val = -1.0
        elif low == 1:
            low_val = -0.6961928009986877
        elif low == 2:
            low_val = -0.5250730514526367
        elif low == 3:
            low_val = -0.39491748809814453
        elif low == 4:
            low_val = -0.28444138169288635
        elif low == 5:
            low_val = -0.18477343022823334
        elif low == 6:
            low_val = -0.09105003625154495
        elif low == 7:
            low_val = 0.0
        elif low == 8:
            low_val = 0.07958029955625534
        elif low == 9:
            low_val = 0.16093020141124725
        elif low == 10:
            low_val = 0.24611230194568634
        elif low == 11:
            low_val = 0.33791524171829224
        elif low == 12:
            low_val = 0.44070982933044434
        elif low == 13:
            low_val = 0.5626170039176941
        elif low == 14:
            low_val = 0.7229568362236023
        else:
            low_val = 1.0
        
        # High nibble
        if high == 0:
            high_val = -1.0
        elif high == 1:
            high_val = -0.6961928009986877
        elif high == 2:
            high_val = -0.5250730514526367
        elif high == 3:
            high_val = -0.39491748809814453
        elif high == 4:
            high_val = -0.28444138169288635
        elif high == 5:
            high_val = -0.18477343022823334
        elif high == 6:
            high_val = -0.09105003625154495
        elif high == 7:
            high_val = 0.0
        elif high == 8:
            high_val = 0.07958029955625534
        elif high == 9:
            high_val = 0.16093020141124725
        elif high == 10:
            high_val = 0.24611230194568634
        elif high == 11:
            high_val = 0.33791524171829224
        elif high == 12:
            high_val = 0.44070982933044434
        elif high == 13:
            high_val = 0.5626170039176941
        elif high == 14:
            high_val = 0.7229568362236023
        else:
            high_val = 1.0
        
        # Scale and store
        tl.store(output_ptr + output_base + i * 2, low_val * scale)
        tl.store(output_ptr + output_base + i * 2 + 1, high_val * scale)


def triton_dequantize_nf4(module):
    """Main NF4 dequantization function."""
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
    _nf4_dequantize_kernel_simple[(total_blocks,)](
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