import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _final_ultra_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Final ultra-optimized kernel processing multiple rows at once."""
    
    # 2D grid - process multiple rows and columns
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Process BLOCK_M rows and BLOCK_N columns
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    
    if row_start >= M or col_start >= N:
        return
    
    # Process multiple 64-element blocks at once
    for row_offset in range(min(BLOCK_M, M - row_start)):
        row = row_start + row_offset
        
        for col_offset in range(0, min(BLOCK_N, N - col_start), 64):
            col_base = col_start + col_offset
            block_idx = col_base // 64
            
            if col_base >= N:
                continue
            
            # Load scales for this block
            absmax_idx = row * blocks_per_row + block_idx
            absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            
            absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx)
            
            scale = absmax * 0.00787401574803149606 * absmax32
            
            # Base offset for this block
            base_offset = row * N + col_base
            
            # Process 64 elements with maximum vectorization
            idx = base_offset + tl.arange(0, 64)
            mask = (col_base + tl.arange(0, 64)) < N
            
            # Load all packed data at once
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            
            # Extract nibbles using vectorized operations
            is_odd = idx & 1
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Direct computation of NF4 values
            # Use polynomial approximation for middle values
            x = nibbles.to(tl.float32)
            
            # Base linear interpolation
            y = (x - 7.5) * 0.2857142857
            
            # Apply corrections for exact values
            y = tl.where(x == 0, -1.0,
                tl.where(x == 1, -0.6961928009986877,
                tl.where(x == 2, -0.5250730514526367,
                tl.where(x == 3, -0.39491748809814453,
                tl.where(x == 4, -0.28444138169288635,
                tl.where(x == 5, -0.18477343022823334,
                tl.where(x == 6, -0.09105003625154495,
                tl.where(x == 7, 0.0,
                tl.where(x == 8, 0.07958029955625534,
                tl.where(x == 9, 0.16093020141124725,
                tl.where(x == 10, 0.24611230194568634,
                tl.where(x == 11, 0.33791524171829224,
                tl.where(x == 12, 0.44070982933044434,
                tl.where(x == 13, 0.5626170039176941,
                tl.where(x == 14, 0.7229568362236023,
                tl.where(x == 15, 1.0, y))))))))))))))))
            
            # Apply scale and store
            output = (y * scale).to(dtype)
            tl.store(output_ptr + idx, output, mask=mask)

def final_ultra_dequantize_nf4(module):
    """Final ultra-optimized NF4 dequantization."""
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
    
    # Prepare absmax
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        elif absmax.numel() == M * blocks_per_row:
            absmax = absmax.view(M, blocks_per_row)
    
    if absmax.shape != (M, blocks_per_row):
        if fast_dequantize is not None:
            return fast_dequantize(weight, quant_state)
        else:
            raise ValueError("Invalid absmax shape")
    
    # Prepare absmax32
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    if absmax32.shape != (M, absmax32_per_row):
        if fast_dequantize is not None:
            return fast_dequantize(weight, quant_state)
        else:
            raise ValueError("Invalid absmax32 shape")
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # 2D grid configuration for better parallelism
    BLOCK_M = 4  # Process 4 rows at a time
    BLOCK_N = 256  # Process 256 columns (4 blocks) at a time
    
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )
    
    _final_ultra_kernel[grid](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        dtype,
        BLOCK_M,
        BLOCK_N,
        num_warps=2,
        num_stages=1,
    )
    
    return output

# Export
triton_dequantize_nf4 = final_ultra_dequantize_nf4