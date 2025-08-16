"""
Optimized NF4 Triton Dequantization Kernel
Ultra-optimized implementation using Triton's advanced features
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _nf4_lookup(nibble):
    """Optimized NF4 lookup using single expression."""
    return tl.where(nibble == 0, -1.0,
           tl.where(nibble == 1, -0.6961928009986877,
           tl.where(nibble == 2, -0.5250730514526367,
           tl.where(nibble == 3, -0.39491748809814453,
           tl.where(nibble == 4, -0.28444138169288635,
           tl.where(nibble == 5, -0.18477343022823334,
           tl.where(nibble == 6, -0.09105003625154495,
           tl.where(nibble == 7, 0.0,
           tl.where(nibble == 8, 0.07958029955625534,
           tl.where(nibble == 9, 0.16093020141124725,
           tl.where(nibble == 10, 0.24611230194568634,
           tl.where(nibble == 11, 0.33791524171829224,
           tl.where(nibble == 12, 0.44070982933044434,
           tl.where(nibble == 13, 0.5626170039176941,
           tl.where(nibble == 14, 0.7229568362236023, 1.0)))))))))))))))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 256}, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_M': 2, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 2, 'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 256}, num_warps=8),
    ],
    key=['m', 'n'],
)
@triton.jit
def _nf4_dequant_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    stride_qw_m, stride_qw_n,
    stride_am_m, stride_am_n,
    stride_a32_m, stride_a32_n,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Ultra-optimized NF4 dequantization with 2D tiling."""
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate block boundaries
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create 2D mask
    rm = tl.max_contiguous(tl.multiple_of(rm % m, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn % n, BLOCK_N), BLOCK_N)
    
    mask_m = rm < m
    mask_n = rn < n
    
    # Process each row in the block
    for row_idx in range(BLOCK_M):
        row = pid_m * BLOCK_M + row_idx
        if row >= m:
            break
            
        # Process columns in chunks
        for col_idx in range(0, BLOCK_N, 64):
            col_start = pid_n * BLOCK_N + col_idx
            if col_start >= n:
                break
                
            # Load scale factors for this 64-element block
            block_idx = col_start // 64
            absmax_idx = row * stride_am_m + block_idx * stride_am_n
            absmax32_idx = row * stride_a32_m + (block_idx // 4) * stride_a32_n
            
            absmax_val = tl.load(absmax_ptr + absmax_idx)
            absmax32_val = tl.load(absmax32_ptr + absmax32_idx)
            scale = absmax_val * 0.00787401574803149606 * absmax32_val
            
            # Process 32 packed bytes
            packed_base = row * stride_qw_m + (col_start // 2) * stride_qw_n
            packed_offsets = tl.arange(0, 32)
            packed_mask = (col_start + packed_offsets * 2) < n
            
            packed_data = tl.load(
                qweight_ptr + packed_base + packed_offsets,
                mask=packed_mask,
                other=0
            )
            
            # Extract and dequantize
            low = _nf4_lookup(packed_data & 0xF) * scale
            high = _nf4_lookup((packed_data >> 4) & 0xF) * scale
            
            # Store results
            output_base = row * stride_out_m + col_start * stride_out_n
            
            # Interleaved store
            even_idx = packed_offsets * 2
            odd_idx = even_idx + 1
            
            even_mask = (col_start + even_idx) < n
            odd_mask = (col_start + odd_idx) < n
            
            tl.store(
                output_ptr + output_base + even_idx,
                low,
                mask=even_mask
            )
            tl.store(
                output_ptr + output_base + odd_idx,
                high,
                mask=odd_mask
            )


def triton_dequantize_nf4(module):
    """Main NF4 dequantization function with optimized launch configuration."""
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
    
    # Get strides
    stride_qw_m, stride_qw_n = qweight.stride()
    stride_am_m, stride_am_n = absmax.stride()
    stride_a32_m, stride_a32_n = absmax32.stride()
    stride_out_m, stride_out_n = output.stride()
    
    # Launch with 2D grid
    def grid(meta):
        return (
            triton.cdiv(m, meta['BLOCK_M']),
            triton.cdiv(n, meta['BLOCK_N']),
        )
    
    _nf4_dequant_kernel[grid](
        qweight,
        absmax,
        absmax32,
        output,
        m, n,
        stride_qw_m, stride_qw_n,
        stride_am_m, stride_am_n,
        stride_a32_m, stride_a32_n,
        stride_out_m, stride_out_n,
    )
    
    return output