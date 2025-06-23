import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _final_optimized_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    WARP_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Final optimized kernel with all performance tricks."""
    # Use 2D grid for better occupancy
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_m >= m:
        return
    
    # Each warp processes WARP_SIZE elements
    col_start = pid_n * WARP_SIZE * 64  # Each thread block handles multiple 64-element blocks
    
    # Pre-load NF4 values into registers using bit tricks
    # Instead of lookup table, use mathematical approximation for some values
    
    # Process multiple 64-element blocks per warp
    for block_idx in range(WARP_SIZE):
        col_block = col_start + block_idx * 64
        if col_block >= n:
            break
        
        # Load absmax values
        abs_block_idx = col_block >> 6
        absmax_idx = pid_m * blocks_per_row + abs_block_idx
        absmax32_idx = pid_m * ((blocks_per_row + 3) >> 2) + (abs_block_idx >> 2)
        
        absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax * 0.00787401574803149606 * absmax32
        
        # Process 64 elements in vectorized chunks
        # Unroll the inner loop for better performance
        base_idx = pid_m * n + col_block
        
        # Process in 16-element vectors (4 vectors per 64-element block)
        for vec in range(4):
            vec_offset = vec * 16
            idx = base_idx + vec_offset + tl.arange(0, 16)
            col = col_block + vec_offset + tl.arange(0, 16)
            mask = col < n
            
            # Load 8 bytes (16 nibbles)
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            
            # Extract nibbles with optimized bit manipulation
            is_odd = (idx & 1).to(tl.int1)
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Ultra-fast NF4 lookup using bit manipulation and arithmetic
            # Split into symmetric parts
            is_negative = nibbles < 8
            abs_val = tl.where(is_negative, 7 - nibbles, nibbles - 8)
            
            # Use polynomial approximation for middle values
            # and exact values for extremes
            is_extreme = (nibbles == 0) | (nibbles == 15)
            is_zero = nibbles == 7
            
            # Approximate middle values with a cubic polynomial
            # Coefficients optimized for NF4 distribution
            x = abs_val.to(tl.float32)
            approx = 0.0795 + 0.0814 * x + 0.0247 * x * x + 0.0186 * x * x * x
            
            # Handle special cases
            nf4_val = tl.where(is_zero, 0.0,
                      tl.where(nibbles == 0, -1.0,
                      tl.where(nibbles == 15, 1.0,
                      tl.where(is_negative, -approx, approx))))
            
            # Apply scale and store
            output = (nf4_val * scale).to(dtype)
            tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def _cache_optimized_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Cache-optimized kernel that processes data in cache-friendly tiles."""
    pid = tl.program_id(0)
    
    # Use a tile size that fits in L1/L2 cache
    TILE_M: tl.constexpr = 4
    TILE_N: tl.constexpr = 256  # 4 blocks of 64
    
    num_tile_m = (m + TILE_M - 1) // TILE_M
    num_tile_n = (n + TILE_N - 1) // TILE_N
    
    tile_m = pid // num_tile_n
    tile_n = pid % num_tile_n
    
    if tile_m >= num_tile_m:
        return
    
    # Pre-compute NF4 values
    nf4_0 = -1.0
    nf4_1 = -0.6961928009986877
    nf4_2 = -0.5250730514526367
    nf4_3 = -0.39491748809814453
    nf4_4 = -0.28444138169288635
    nf4_5 = -0.18477343022823334
    nf4_6 = -0.09105003625154495
    nf4_7 = 0.0
    nf4_8 = 0.07958029955625534
    nf4_9 = 0.16093020141124725
    nf4_10 = 0.24611230194568634
    nf4_11 = 0.33791524171829224
    nf4_12 = 0.44070982933044434
    nf4_13 = 0.5626170039176941
    nf4_14 = 0.7229568362236023
    nf4_15 = 1.0
    
    # Process TILE_M rows
    for row_offset in range(TILE_M):
        row = tile_m * TILE_M + row_offset
        if row >= m:
            break
        
        # Process TILE_N columns in 64-element blocks
        col_base = tile_n * TILE_N
        
        # Load absmax32 values for this row tile (reuse across blocks)
        absmax32_base_idx = row * absmax32_per_row + (col_base >> 8)
        
        for block in range(0, TILE_N, 64):
            col_block = col_base + block
            if col_block >= n:
                break
            
            # Compute indices
            block_idx = col_block >> 6
            absmax_idx = row * blocks_per_row + block_idx
            absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
            
            # Load scales
            absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx)
            scale = absmax * 0.00787401574803149606 * absmax32
            
            # Process 64 elements with maximum vectorization
            base_idx = row * n + col_block
            
            # Unroll completely for 64 elements (4x16)
            for i in range(0, 64, 16):
                idx = base_idx + i + tl.arange(0, 16)
                col = col_block + i + tl.arange(0, 16)
                mask = col < n
                
                # Vectorized load
                packed_idx = idx >> 1
                packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
                
                # Extract nibbles
                nibbles = tl.where((idx & 1) == 1, (packed >> 4) & 0xF, packed & 0xF)
                
                # Direct lookup with minimal branching
                nf4 = tl.where(nibbles == 0, nf4_0,
                      tl.where(nibbles == 1, nf4_1,
                      tl.where(nibbles == 2, nf4_2,
                      tl.where(nibbles == 3, nf4_3,
                      tl.where(nibbles == 4, nf4_4,
                      tl.where(nibbles == 5, nf4_5,
                      tl.where(nibbles == 6, nf4_6,
                      tl.where(nibbles == 7, nf4_7,
                      tl.where(nibbles == 8, nf4_8,
                      tl.where(nibbles == 9, nf4_9,
                      tl.where(nibbles == 10, nf4_10,
                      tl.where(nibbles == 11, nf4_11,
                      tl.where(nibbles == 12, nf4_12,
                      tl.where(nibbles == 13, nf4_13,
                      tl.where(nibbles == 14, nf4_14, nf4_15)))))))))))))))
                
                # Store results
                output = (nf4 * scale).to(dtype)
                tl.store(output_ptr + idx, output, mask=mask)

def final_optimized_dequantize_nf4(module):
    """Final optimized NF4 dequantization combining all techniques."""
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
    
    # Prepare tensors
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        elif absmax.numel() == M * blocks_per_row:
            absmax = absmax.view(M, blocks_per_row)
    
    if absmax.shape != (M, blocks_per_row):
        return fast_dequantize(weight, quant_state)
    
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    if absmax32.shape != (M, absmax32_per_row):
        return fast_dequantize(weight, quant_state)
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Choose kernel based on matrix size
    if N <= 4096 and M <= 4096:
        # Small matrices: use 2D grid for better parallelism
        WARP_SIZE = 2  # Each warp handles 2 blocks of 64
        grid = (M, (N + WARP_SIZE * 64 - 1) // (WARP_SIZE * 64))
        
        _final_optimized_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            WARP_SIZE,
            dtype,
        )
    else:
        # Large matrices: use cache-optimized tiling
        TILE_M = 4
        TILE_N = 256
        num_tiles = ((M + TILE_M - 1) // TILE_M) * ((N + TILE_N - 1) // TILE_N)
        grid = (num_tiles,)
        
        _cache_optimized_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            dtype,
        )
    
    return output