"""
Optimized NF4 Triton Dequantization Kernel
High-performance implementation with autotuning for Tesla T4
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=['m', 'n'],
)
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
    """Autotuned NF4 dequantization kernel."""
    
    pid = tl.program_id(0)
    
    # Each block processes one 64-element NF4 block
    row = pid // blocks_per_row
    block_in_row = pid % blocks_per_row
    
    if row >= m:
        return
    
    col_start = block_in_row * 64
    if col_start >= n:
        return
    
    # Load scale factors
    absmax_idx = row * blocks_per_row + block_in_row
    absmax32_idx = row * absmax32_per_row + (block_in_row >> 2)
    
    absmax_val = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
    absmax32_val = tl.load(absmax32_ptr + absmax32_idx).to(tl.float32)
    scale = absmax_val * 0.00787401574803149606 * absmax32_val
    
    # Base addresses
    qweight_base = row * (n >> 1) + (col_start >> 1)
    output_base = row * n + col_start
    
    # Load all 32 bytes at once
    offsets = tl.arange(0, 32)
    packed = tl.load(qweight_ptr + qweight_base + offsets)
    
    # Extract nibbles
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    
    # NF4 lookup table
    low_vals = tl.where(low == 0, -1.0,
               tl.where(low == 1, -0.6961928009986877,
               tl.where(low == 2, -0.5250730514526367,
               tl.where(low == 3, -0.39491748809814453,
               tl.where(low == 4, -0.28444138169288635,
               tl.where(low == 5, -0.18477343022823334,
               tl.where(low == 6, -0.09105003625154495,
               tl.where(low == 7, 0.0,
               tl.where(low == 8, 0.07958029955625534,
               tl.where(low == 9, 0.16093020141124725,
               tl.where(low == 10, 0.24611230194568634,
               tl.where(low == 11, 0.33791524171829224,
               tl.where(low == 12, 0.44070982933044434,
               tl.where(low == 13, 0.5626170039176941,
               tl.where(low == 14, 0.7229568362236023, 1.0)))))))))))))))
    
    high_vals = tl.where(high == 0, -1.0,
                tl.where(high == 1, -0.6961928009986877,
                tl.where(high == 2, -0.5250730514526367,
                tl.where(high == 3, -0.39491748809814453,
                tl.where(high == 4, -0.28444138169288635,
                tl.where(high == 5, -0.18477343022823334,
                tl.where(high == 6, -0.09105003625154495,
                tl.where(high == 7, 0.0,
                tl.where(high == 8, 0.07958029955625534,
                tl.where(high == 9, 0.16093020141124725,
                tl.where(high == 10, 0.24611230194568634,
                tl.where(high == 11, 0.33791524171829224,
                tl.where(high == 12, 0.44070982933044434,
                tl.where(high == 13, 0.5626170039176941,
                tl.where(high == 14, 0.7229568362236023, 1.0)))))))))))))))
    
    # Apply scale
    low_scaled = low_vals * scale
    high_scaled = high_vals * scale
    
    # Store interleaved results
    for i in tl.static_range(32):
        idx = i * 2
        if col_start + idx < n:
            tl.store(output_ptr + output_base + idx, low_scaled[i])
        if col_start + idx + 1 < n:
            tl.store(output_ptr + output_base + idx + 1, high_scaled[i])


def triton_dequantize_nf4(module):
    """Main NF4 dequantization function with autotuning."""
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
    
    # Ensure contiguous memory
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
    )
    
    return output