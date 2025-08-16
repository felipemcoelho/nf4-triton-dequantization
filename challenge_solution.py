"""
NF4 Triton Dequantization Challenge Solution
Target: 1.15x speedup over Unsloth's fast_dequantize
"""

import torch
import triton
import triton.language as tl

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
    """Optimized single Triton kernel for NF4 dequantization with double dequant."""
    
    pid = tl.program_id(0)
    
    # Decode position
    row = pid // blocks_per_row
    block_in_row = pid % blocks_per_row
    
    if row >= M:
        return
    
    col_start = block_in_row * 64
    if col_start >= N:
        return
    
    # Double dequantization: combine both scale factors
    absmax_idx = row * blocks_per_row + block_in_row
    absmax_val = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
    
    absmax32_idx = row * absmax32_per_row + (block_in_row >> 2)
    absmax32_val = tl.load(absmax32_ptr + absmax32_idx).to(tl.float32)
    
    # Combined scale with NF4 constant (1/127)
    scale = absmax_val * 0.00787401574803149606 * absmax32_val
    
    # Calculate base addresses
    qweight_base = row * (N >> 1) + (col_start >> 1)
    output_base = row * N + col_start
    
    # Vectorized load of 32 packed bytes
    packed = tl.load(qweight_ptr + qweight_base + tl.arange(0, 32))
    
    # Extract nibbles
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    
    # NF4 lookup - standard approach
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
    
    # Vectorized store
    for i in range(32):
        idx = i * 2
        if col_start + idx < N:
            tl.store(output_ptr + output_base + idx, low_scaled[i])
        if col_start + idx + 1 < N:
            tl.store(output_ptr + output_base + idx + 1, high_scaled[i])


def _your_dequantize_nf4(weight, quant_state):
    """Setup and launch the optimized Triton kernel."""
    
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
    
    # Ensure contiguous memory layout
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Allocate output tensor
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch kernel
    total_blocks = M * blocks_per_row
    
    _your_dequantize_nf4_kernel[(total_blocks,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        num_warps=2,
        num_stages=2,
    )
    
    return output


def your_dequantize_nf4(weight):
    """Main entry point for the challenge - dequantizes a Linear4bit weight."""
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state)