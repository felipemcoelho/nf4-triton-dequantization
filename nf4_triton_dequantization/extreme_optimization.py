import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _extreme_nf4_kernel(
    qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Extreme optimization with register tiling and software pipelining."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Early exit for out-of-bounds blocks
    if pid_m * BLOCK_M >= M or pid_n * BLOCK_N >= N:
        return
    
    # NF4 lookup table in registers (constant propagation)
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
    
    # Scaling constant
    scale_const = 0.00787401574803149606  # 1/127
    
    # Process blocks with register tiling
    for m in range(BLOCK_M):
        row = pid_m * BLOCK_M + m
        if row >= M:
            break
            
        # Process in aligned 64-element chunks for optimal memory access
        for n_start in range(pid_n * BLOCK_N, min((pid_n + 1) * BLOCK_N, N), 64):
            # Calculate absmax indices
            block_idx = n_start >> 6
            absmax_idx = row * blocks_per_row + block_idx
            absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
            
            # Load and fuse scaling factors
            absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx)
            fused_scale = absmax * scale_const * absmax32
            
            # Process 64 elements with vectorized operations
            # Unroll by 2 for better ILP
            for i in range(0, 64, 32):
                n = n_start + i + tl.arange(0, 32)
                mask = n < N
                
                # Calculate indices
                linear_idx = row * N + n
                packed_idx = linear_idx >> 1
                
                # Load packed weights (16 bytes = 32 nibbles)
                packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
                
                # Extract nibbles with optimized bit manipulation
                is_odd = linear_idx & 1
                nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
                
                # Optimized NF4 lookup using nested ternary for better GPU utilization
                # Split into two parts for better instruction scheduling
                nf4_low = tl.where(nibbles < 8,
                    tl.where(nibbles < 4,
                        tl.where(nibbles < 2,
                            tl.where(nibbles == 0, nf4_0, nf4_1),
                            tl.where(nibbles == 2, nf4_2, nf4_3)),
                        tl.where(nibbles < 6,
                            tl.where(nibbles == 4, nf4_4, nf4_5),
                            tl.where(nibbles == 6, nf4_6, nf4_7))),
                    0.0)
                
                nf4_high = tl.where(nibbles >= 8,
                    tl.where(nibbles < 12,
                        tl.where(nibbles < 10,
                            tl.where(nibbles == 8, nf4_8, nf4_9),
                            tl.where(nibbles == 10, nf4_10, nf4_11)),
                        tl.where(nibbles < 14,
                            tl.where(nibbles == 12, nf4_12, nf4_13),
                            tl.where(nibbles == 14, nf4_14, nf4_15))),
                    0.0)
                
                nf4_vals = nf4_low + nf4_high
                
                # Apply scaling and store
                output = nf4_vals * fused_scale
                output_idx = row * N + n
                tl.store(output_ptr + output_idx, output, mask=mask)

@triton.jit
def _extreme_nf4_kernel_v2(
    qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Version 2 with aggressive memory coalescing and warp-level optimizations."""
    # Use 1D grid with optimal work distribution
    pid = tl.program_id(0)
    
    # Calculate work per thread block
    ELEMENTS_PER_BLOCK = 2048  # Tuned for optimal occupancy
    total_elements = M * N
    
    # Early exit
    if pid * ELEMENTS_PER_BLOCK >= total_elements:
        return
    
    # Process elements in warps of 32 for coalesced access
    for elem_offset in range(0, ELEMENTS_PER_BLOCK, 256):  # 256 = 8 warps
        base_idx = pid * ELEMENTS_PER_BLOCK + elem_offset
        
        # Process 256 elements at once
        idx = base_idx + tl.arange(0, 256)
        mask = idx < total_elements
        
        # Convert to row/col
        row = idx // N
        col = idx % N
        row_mask = row < M
        full_mask = mask & row_mask
        
        # Batch load scaling factors
        block_idx = col >> 6
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        
        absmax = tl.load(absmax_ptr + absmax_idx, mask=full_mask, other=0).to(tl.float32)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx, mask=full_mask, other=1.0)
        scale = absmax * 0.00787401574803149606 * absmax32
        
        # Batch load packed weights
        packed_idx = idx >> 1
        packed = tl.load(qweight_ptr + packed_idx, mask=full_mask, other=0)
        
        # Extract nibbles with minimal branching
        shift = ((idx & 1) << 2).to(tl.int32)
        nibbles = (packed >> shift) & 0xF
        
        # Ultra-optimized NF4 lookup using bit manipulation tricks
        # Exploit the fact that NF4 values have a pattern
        is_zero = nibbles == 7
        is_negative = nibbles < 7
        
        # Base lookup for positive values (8-15)
        pos_base = tl.where(nibbles == 8, 0.07958029955625534,
                   tl.where(nibbles == 9, 0.16093020141124725,
                   tl.where(nibbles == 10, 0.24611230194568634,
                   tl.where(nibbles == 11, 0.33791524171829224,
                   tl.where(nibbles == 12, 0.44070982933044434,
                   tl.where(nibbles == 13, 0.5626170039176941,
                   tl.where(nibbles == 14, 0.7229568362236023,
                   1.0)))))))
        
        # Base lookup for negative values (0-6)
        neg_base = tl.where(nibbles == 0, -1.0,
                   tl.where(nibbles == 1, -0.6961928009986877,
                   tl.where(nibbles == 2, -0.5250730514526367,
                   tl.where(nibbles == 3, -0.39491748809814453,
                   tl.where(nibbles == 4, -0.28444138169288635,
                   tl.where(nibbles == 5, -0.18477343022823334,
                   -0.09105003625154495))))))
        
        # Combine results
        nf4_vals = tl.where(is_zero, 0.0, tl.where(is_negative, neg_base, pos_base))
        
        # Apply scaling and store with coalesced access
        output = nf4_vals * scale
        tl.store(output_ptr + idx, output, mask=full_mask)

def extreme_triton_dequantize_nf4(module):
    """Extreme performance NF4 dequantization targeting 1.15x+ speedup."""
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
    
    # Tensor preparation (same as before)
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
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Choose kernel based on dimensions and GPU architecture
    total_elements = M * N
    
    if N % 64 == 0 and M < 8192:  # Well-aligned case
        # Use 2D grid for better cache utilization
        BLOCK_M = min(8, M)
        BLOCK_N = 256  # Process 4 absmax blocks at once
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        _extreme_nf4_kernel[grid](
            qweight.contiguous(),
            absmax.contiguous(),
            absmax32.contiguous(),
            output,
            M, N,
            blocks_per_row,
            absmax32_per_row,
            BLOCK_M, BLOCK_N,
        )
    else:  # General case with 1D grid
        ELEMENTS_PER_BLOCK = 2048
        grid = (triton.cdiv(total_elements, ELEMENTS_PER_BLOCK),)
        
        _extreme_nf4_kernel_v2[grid](
            qweight.contiguous().view(-1),
            absmax.contiguous().view(-1),
            absmax32.contiguous().view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
        )
    
    return output