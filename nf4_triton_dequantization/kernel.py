"""
Optimized NF4 Triton Dequantization Kernel
Single file containing the complete optimized implementation
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def _nf4_dequantize_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Optimized NF4 dequantization kernel with double dequant in single pass."""
    
    pid = tl.program_id(0)
    
    # Decode position
    row = pid // blocks_per_row
    block_in_row = pid % blocks_per_row
    
    if row >= M:
        return
    
    col_start = block_in_row * 64
    if col_start >= N:
        return
    
    # Double dequantization - load both scale factors
    absmax_idx = row * blocks_per_row + block_in_row
    absmax_val = tl.load(absmax_ptr + absmax_idx, cache_modifier=".ca").to(tl.float32)
    
    absmax32_idx = row * absmax32_per_row + (block_in_row >> 2)
    absmax32_val = tl.load(absmax32_ptr + absmax32_idx, cache_modifier=".ca").to(tl.float32)
    
    # Combined scale (1/127 = 0.00787401574803149606)
    scale = absmax_val * 0.00787401574803149606 * absmax32_val
    
    # Calculate addresses
    qweight_base = row * (N >> 1) + (col_start >> 1)
    output_base = row * N + col_start
    
    # Load 32 packed bytes (64 nibbles)
    packed = tl.load(qweight_ptr + qweight_base + tl.arange(0, 32), cache_modifier=".ca")
    
    # Extract nibbles
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    
    # Optimized NF4 lookup using balanced binary tree
    low_vals = tl.where(low < 8,
        tl.where(low < 4,
            tl.where(low < 2,
                tl.where(low == 0, -1.0, -0.6961928009986877),
                tl.where(low == 2, -0.5250730514526367, -0.39491748809814453)
            ),
            tl.where(low < 6,
                tl.where(low == 4, -0.28444138169288635, -0.18477343022823334),
                tl.where(low == 6, -0.09105003625154495, 0.0)
            )
        ),
        tl.where(low < 12,
            tl.where(low < 10,
                tl.where(low == 8, 0.07958029955625534, 0.16093020141124725),
                tl.where(low == 10, 0.24611230194568634, 0.33791524171829224)
            ),
            tl.where(low < 14,
                tl.where(low == 12, 0.44070982933044434, 0.5626170039176941),
                tl.where(low == 14, 0.7229568362236023, 1.0)
            )
        )
    )
    
    high_vals = tl.where(high < 8,
        tl.where(high < 4,
            tl.where(high < 2,
                tl.where(high == 0, -1.0, -0.6961928009986877),
                tl.where(high == 2, -0.5250730514526367, -0.39491748809814453)
            ),
            tl.where(high < 6,
                tl.where(high == 4, -0.28444138169288635, -0.18477343022823334),
                tl.where(high == 6, -0.09105003625154495, 0.0)
            )
        ),
        tl.where(high < 12,
            tl.where(high < 10,
                tl.where(high == 8, 0.07958029955625534, 0.16093020141124725),
                tl.where(high == 10, 0.24611230194568634, 0.33791524171829224)
            ),
            tl.where(high < 14,
                tl.where(high == 12, 0.44070982933044434, 0.5626170039176941),
                tl.where(high == 14, 0.7229568362236023, 1.0)
            )
        )
    )
    
    # Apply scale
    low_scaled = low_vals * scale
    high_scaled = high_vals * scale
    
    # Store with unrolling for better performance
    for i in tl.static_range(32):
        idx = i * 2
        if col_start + idx < N:
            tl.store(output_ptr + output_base + idx, low_scaled[i], cache_modifier=".wb")
        if col_start + idx + 1 < N:
            tl.store(output_ptr + output_base + idx + 1, high_scaled[i], cache_modifier=".wb")


def triton_dequantize_nf4(module):
    """Main NF4 dequantization function."""
    weight = module.weight
    quant_state = weight.quant_state
    
    qweight = weight.data
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    dtype = quant_state.dtype
    device = qweight.device
    
    M = module.out_features
    N = module.in_features
    
    blocks_per_row = (N + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Handle tensor shapes
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        elif absmax.numel() == M * blocks_per_row:
            absmax = absmax.view(M, blocks_per_row)
    
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch kernel
    total_blocks = M * blocks_per_row
    _nf4_dequantize_kernel[(total_blocks,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
    )
    
    return output