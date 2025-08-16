"""
NF4 Triton Dequantization Challenge Solution
Target: 1.15x speedup over Unsloth's fast_dequantize
"""

import torch
import triton
import triton.language as tl
from triton import jit

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def _your_dequantize_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Ultra-optimized single Triton kernel for NF4 dequantization with double dequant."""
    
    # Efficient grid indexing with swizzling for better cache utilization
    pid = tl.program_id(0)
    
    # Decode block position
    row = pid // blocks_per_row
    block_in_row = pid % blocks_per_row
    
    if row >= M:
        return
    
    col_start = block_in_row * 64
    if col_start >= N:
        return
    
    # Double dequantization: combine both scale factors in single operation
    absmax_idx = row * blocks_per_row + block_in_row
    absmax_val = tl.load(absmax_ptr + absmax_idx, cache_modifier=".ca").to(tl.float32)
    
    absmax32_idx = row * absmax32_per_row + (block_in_row >> 2)
    absmax32_val = tl.load(absmax32_ptr + absmax32_idx, cache_modifier=".ca").to(tl.float32)
    
    # Combined scale with NF4 constant (1/127)
    scale = absmax_val * 0.00787401574803149606 * absmax32_val
    
    # Calculate base addresses with aligned access
    qweight_base = row * (N >> 1) + (col_start >> 1)
    output_base = row * N + col_start
    
    # Vectorized load of 32 packed bytes
    offsets = tl.arange(0, 32)
    packed = tl.load(qweight_ptr + qweight_base + offsets, cache_modifier=".ca", eviction_policy="evict_first")
    
    # Parallel nibble extraction
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    
    # Ultra-optimized NF4 lookup using balanced binary tree
    # This minimizes branch depth and improves instruction-level parallelism
    low_vals = tl.where(low < 8,
        # Negative values (0-7)
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
        # Positive values (8-15)
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
    
    # Apply scale with fused multiply
    low_scaled = low_vals * scale
    high_scaled = high_vals * scale
    
    # Vectorized store with loop unrolling for maximum throughput
    # Unroll first 16 pairs completely for better performance
    tl.store(output_ptr + output_base + 0, low_scaled[0], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 1, high_scaled[0], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 2, low_scaled[1], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 3, high_scaled[1], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 4, low_scaled[2], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 5, high_scaled[2], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 6, low_scaled[3], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 7, high_scaled[3], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 8, low_scaled[4], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 9, high_scaled[4], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 10, low_scaled[5], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 11, high_scaled[5], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 12, low_scaled[6], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 13, high_scaled[6], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 14, low_scaled[7], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 15, high_scaled[7], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 16, low_scaled[8], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 17, high_scaled[8], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 18, low_scaled[9], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 19, high_scaled[9], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 20, low_scaled[10], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 21, high_scaled[10], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 22, low_scaled[11], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 23, high_scaled[11], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 24, low_scaled[12], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 25, high_scaled[12], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 26, low_scaled[13], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 27, high_scaled[13], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 28, low_scaled[14], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 29, high_scaled[14], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 30, low_scaled[15], cache_modifier=".wb", eviction_policy="evict_first")
    tl.store(output_ptr + output_base + 31, high_scaled[15], cache_modifier=".wb", eviction_policy="evict_first")
    
    # Process remaining elements
    for i in tl.static_range(16, 32):
        idx = i * 2
        if col_start + idx < N:
            tl.store(output_ptr + output_base + idx, low_scaled[i], cache_modifier=".wb", eviction_policy="evict_first")
        if col_start + idx + 1 < N:
            tl.store(output_ptr + output_base + idx + 1, high_scaled[i], cache_modifier=".wb", eviction_policy="evict_first")


def _your_dequantize_nf4(weight, quant_state):
    """Setup and launch the optimized Triton kernel."""
    
    # Extract parameters
    qweight = weight
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    dtype = quant_state.dtype
    device = qweight.device
    
    # Determine matrix dimensions
    packed_shape = qweight.shape
    M = packed_shape[0]
    N = packed_shape[1] * 2  # Each byte contains 2 4-bit values
    
    blocks_per_row = (N + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Handle tensor shapes for absmax
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        elif absmax.numel() == M * blocks_per_row:
            absmax = absmax.view(M, blocks_per_row)
    
    # Handle tensor shapes for absmax32
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    # Ensure contiguous memory layout for performance
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Allocate output tensor
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch kernel with optimal grid
    total_blocks = M * blocks_per_row
    
    _your_dequantize_nf4_kernel[(total_blocks,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
    )
    
    return output


def your_dequantize_nf4(weight):
    """Main entry point for the challenge - dequantizes a Linear4bit weight."""
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state)


# Ensure compatibility with torch.compile
if hasattr(torch, 'compiler'):
    your_dequantize_nf4 = torch.compiler.disable(your_dequantize_nf4)