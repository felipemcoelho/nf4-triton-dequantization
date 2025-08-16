"""
Optimized NF4 Triton Dequantization Kernel
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
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Optimized NF4 dequantization kernel."""
    
    pid = tl.program_id(0)
    
    row = pid // blocks_per_row
    block_idx = pid % blocks_per_row
    
    if row >= M:
        return
        
    col_start = block_idx * 64
    if col_start >= N:
        return
    
    # Load scale factors once
    absmax_idx = row * blocks_per_row + block_idx
    absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
    
    absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx).to(tl.float32)
    
    scale = absmax * 0.00787401574803149606 * absmax32
    
    # Base pointers
    qweight_base = row * (N >> 1) + (col_start >> 1)
    output_base = row * N + col_start
    
    # Load 32 bytes at once
    offsets = tl.arange(0, 32)
    packed = tl.load(qweight_ptr + qweight_base + offsets)
    
    # Process all 32 bytes in parallel
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    
    # NF4 lookup - use select for better performance
    # Process low nibbles
    low_neg = low < 8
    low_vals = tl.where(low_neg,
        tl.where(low == 0, -1.0,
        tl.where(low == 1, -0.6961928009986877,
        tl.where(low == 2, -0.5250730514526367,
        tl.where(low == 3, -0.39491748809814453,
        tl.where(low == 4, -0.28444138169288635,
        tl.where(low == 5, -0.18477343022823334,
        tl.where(low == 6, -0.09105003625154495, 0.0))))))),
        tl.where(low == 8, 0.07958029955625534,
        tl.where(low == 9, 0.16093020141124725,
        tl.where(low == 10, 0.24611230194568634,
        tl.where(low == 11, 0.33791524171829224,
        tl.where(low == 12, 0.44070982933044434,
        tl.where(low == 13, 0.5626170039176941,
        tl.where(low == 14, 0.7229568362236023, 1.0)))))))
    )
    
    # Process high nibbles
    high_neg = high < 8
    high_vals = tl.where(high_neg,
        tl.where(high == 0, -1.0,
        tl.where(high == 1, -0.6961928009986877,
        tl.where(high == 2, -0.5250730514526367,
        tl.where(high == 3, -0.39491748809814453,
        tl.where(high == 4, -0.28444138169288635,
        tl.where(high == 5, -0.18477343022823334,
        tl.where(high == 6, -0.09105003625154495, 0.0))))))),
        tl.where(high == 8, 0.07958029955625534,
        tl.where(high == 9, 0.16093020141124725,
        tl.where(high == 10, 0.24611230194568634,
        tl.where(high == 11, 0.33791524171829224,
        tl.where(high == 12, 0.44070982933044434,
        tl.where(high == 13, 0.5626170039176941,
        tl.where(high == 14, 0.7229568362236023, 1.0)))))))
    )
    
    # Scale values
    low_scaled = low_vals * scale
    high_scaled = high_vals * scale
    
    # Store results - unroll for better performance
    # First 16 elements
    if col_start < N:
        tl.store(output_ptr + output_base + 0, low_scaled[0])
        tl.store(output_ptr + output_base + 1, high_scaled[0])
    if col_start + 2 < N:
        tl.store(output_ptr + output_base + 2, low_scaled[1])
        tl.store(output_ptr + output_base + 3, high_scaled[1])
    if col_start + 4 < N:
        tl.store(output_ptr + output_base + 4, low_scaled[2])
        tl.store(output_ptr + output_base + 5, high_scaled[2])
    if col_start + 6 < N:
        tl.store(output_ptr + output_base + 6, low_scaled[3])
        tl.store(output_ptr + output_base + 7, high_scaled[3])
    if col_start + 8 < N:
        tl.store(output_ptr + output_base + 8, low_scaled[4])
        tl.store(output_ptr + output_base + 9, high_scaled[4])
    if col_start + 10 < N:
        tl.store(output_ptr + output_base + 10, low_scaled[5])
        tl.store(output_ptr + output_base + 11, high_scaled[5])
    if col_start + 12 < N:
        tl.store(output_ptr + output_base + 12, low_scaled[6])
        tl.store(output_ptr + output_base + 13, high_scaled[6])
    if col_start + 14 < N:
        tl.store(output_ptr + output_base + 14, low_scaled[7])
        tl.store(output_ptr + output_base + 15, high_scaled[7])
    if col_start + 16 < N:
        tl.store(output_ptr + output_base + 16, low_scaled[8])
        tl.store(output_ptr + output_base + 17, high_scaled[8])
    if col_start + 18 < N:
        tl.store(output_ptr + output_base + 18, low_scaled[9])
        tl.store(output_ptr + output_base + 19, high_scaled[9])
    if col_start + 20 < N:
        tl.store(output_ptr + output_base + 20, low_scaled[10])
        tl.store(output_ptr + output_base + 21, high_scaled[10])
    if col_start + 22 < N:
        tl.store(output_ptr + output_base + 22, low_scaled[11])
        tl.store(output_ptr + output_base + 23, high_scaled[11])
    if col_start + 24 < N:
        tl.store(output_ptr + output_base + 24, low_scaled[12])
        tl.store(output_ptr + output_base + 25, high_scaled[12])
    if col_start + 26 < N:
        tl.store(output_ptr + output_base + 26, low_scaled[13])
        tl.store(output_ptr + output_base + 27, high_scaled[13])
    if col_start + 28 < N:
        tl.store(output_ptr + output_base + 28, low_scaled[14])
        tl.store(output_ptr + output_base + 29, high_scaled[14])
    if col_start + 30 < N:
        tl.store(output_ptr + output_base + 30, low_scaled[15])
        tl.store(output_ptr + output_base + 31, high_scaled[15])
    
    # Second 16 elements
    if col_start + 32 < N:
        tl.store(output_ptr + output_base + 32, low_scaled[16])
        tl.store(output_ptr + output_base + 33, high_scaled[16])
    if col_start + 34 < N:
        tl.store(output_ptr + output_base + 34, low_scaled[17])
        tl.store(output_ptr + output_base + 35, high_scaled[17])
    if col_start + 36 < N:
        tl.store(output_ptr + output_base + 36, low_scaled[18])
        tl.store(output_ptr + output_base + 37, high_scaled[18])
    if col_start + 38 < N:
        tl.store(output_ptr + output_base + 38, low_scaled[19])
        tl.store(output_ptr + output_base + 39, high_scaled[19])
    if col_start + 40 < N:
        tl.store(output_ptr + output_base + 40, low_scaled[20])
        tl.store(output_ptr + output_base + 41, high_scaled[20])
    if col_start + 42 < N:
        tl.store(output_ptr + output_base + 42, low_scaled[21])
        tl.store(output_ptr + output_base + 43, high_scaled[21])
    if col_start + 44 < N:
        tl.store(output_ptr + output_base + 44, low_scaled[22])
        tl.store(output_ptr + output_base + 45, high_scaled[22])
    if col_start + 46 < N:
        tl.store(output_ptr + output_base + 46, low_scaled[23])
        tl.store(output_ptr + output_base + 47, high_scaled[23])
    if col_start + 48 < N:
        tl.store(output_ptr + output_base + 48, low_scaled[24])
        tl.store(output_ptr + output_base + 49, high_scaled[24])
    if col_start + 50 < N:
        tl.store(output_ptr + output_base + 50, low_scaled[25])
        tl.store(output_ptr + output_base + 51, high_scaled[25])
    if col_start + 52 < N:
        tl.store(output_ptr + output_base + 52, low_scaled[26])
        tl.store(output_ptr + output_base + 53, high_scaled[26])
    if col_start + 54 < N:
        tl.store(output_ptr + output_base + 54, low_scaled[27])
        tl.store(output_ptr + output_base + 55, high_scaled[27])
    if col_start + 56 < N:
        tl.store(output_ptr + output_base + 56, low_scaled[28])
        tl.store(output_ptr + output_base + 57, high_scaled[28])
    if col_start + 58 < N:
        tl.store(output_ptr + output_base + 58, low_scaled[29])
        tl.store(output_ptr + output_base + 59, high_scaled[29])
    if col_start + 60 < N:
        tl.store(output_ptr + output_base + 60, low_scaled[30])
        tl.store(output_ptr + output_base + 61, high_scaled[30])
    if col_start + 62 < N:
        tl.store(output_ptr + output_base + 62, low_scaled[31])
        tl.store(output_ptr + output_base + 63, high_scaled[31])


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
        num_warps=2,
        num_stages=2,
    )
    
    return output