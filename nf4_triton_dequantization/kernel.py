"""
Optimized NF4 Triton Dequantization Kernel
Ultra-fast implementation for Tesla T4
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _nf4_dequantize_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Ultra-optimized NF4 kernel with coalesced memory access."""
    
    pid = tl.program_id(0)
    
    row = pid // blocks_per_row
    block_in_row = pid % blocks_per_row
    
    if row >= m:
        return
    
    col_start = block_in_row * 64
    if col_start >= n:
        return
    
    # Load scales
    absmax_idx = row * blocks_per_row + block_in_row
    absmax32_idx = row * absmax32_per_row + (block_in_row >> 2)
    
    absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx).to(tl.float32)
    scale = absmax * 0.00787401574803149606 * absmax32
    
    # Base pointers
    qbase = row * (n >> 1) + (col_start >> 1)
    obase = row * n + col_start
    
    # Load all 32 bytes at once for coalesced access
    offs = tl.arange(0, 32)
    packed = tl.load(qweight_ptr + qbase + offs)
    
    # Extract nibbles
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    
    # Optimized NF4 lookup using single expression
    # Low nibbles
    lv = tl.where(low < 8,
         tl.where(low < 4,
           tl.where(low < 2,
             tl.where(low == 0, -1.0, -0.6961928009986877),
             tl.where(low == 2, -0.5250730514526367, -0.39491748809814453)),
           tl.where(low < 6,
             tl.where(low == 4, -0.28444138169288635, -0.18477343022823334),
             tl.where(low == 6, -0.09105003625154495, 0.0))),
         tl.where(low < 12,
           tl.where(low < 10,
             tl.where(low == 8, 0.07958029955625534, 0.16093020141124725),
             tl.where(low == 10, 0.24611230194568634, 0.33791524171829224)),
           tl.where(low < 14,
             tl.where(low == 12, 0.44070982933044434, 0.5626170039176941),
             tl.where(low == 14, 0.7229568362236023, 1.0))))
    
    # High nibbles
    hv = tl.where(high < 8,
         tl.where(high < 4,
           tl.where(high < 2,
             tl.where(high == 0, -1.0, -0.6961928009986877),
             tl.where(high == 2, -0.5250730514526367, -0.39491748809814453)),
           tl.where(high < 6,
             tl.where(high == 4, -0.28444138169288635, -0.18477343022823334),
             tl.where(high == 6, -0.09105003625154495, 0.0))),
         tl.where(high < 12,
           tl.where(high < 10,
             tl.where(high == 8, 0.07958029955625534, 0.16093020141124725),
             tl.where(high == 10, 0.24611230194568634, 0.33791524171829224)),
           tl.where(high < 14,
             tl.where(high == 12, 0.44070982933044434, 0.5626170039176941),
             tl.where(high == 14, 0.7229568362236023, 1.0))))
    
    # Apply scale
    lv = lv * scale
    hv = hv * scale
    
    # Vectorized interleaved store using advanced indexing
    # Store low values at even positions
    even_offs = offs * 2
    even_mask = (col_start + even_offs) < n
    tl.store(output_ptr + obase + even_offs, lv, mask=even_mask)
    
    # Store high values at odd positions
    odd_offs = offs * 2 + 1
    odd_mask = (col_start + odd_offs) < n
    tl.store(output_ptr + obase + odd_offs, hv, mask=odd_mask)


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
    
    _nf4_dequantize_kernel[(total_blocks,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        m, n,
        blocks_per_row,
        absmax32_per_row,
        num_warps=8,  # More warps for better occupancy
        num_stages=1,  # Minimal stages for less overhead
    )
    
    return output