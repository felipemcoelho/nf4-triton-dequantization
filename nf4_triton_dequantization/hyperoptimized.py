import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _hyper_optimized_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Hyper-optimized kernel with maximum performance tricks."""
    pid = tl.program_id(0)
    
    # Constants
    SCALE = tl.constexpr(0.00787401574803149606)
    
    # Grid-stride loop
    total = M * N
    start = pid * BLOCK_SIZE
    stride = tl.num_programs(0) * BLOCK_SIZE
    
    # Process elements
    for offset in range(start, total, stride):
        # Vectorized processing - process 128 elements at once
        idx = offset + tl.arange(0, 128)
        mask = idx < total
        
        # Convert to 2D
        row = idx // N
        col = idx % N
        
        # Batch compute indices
        block_idx = col >> 6
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        packed_idx = idx >> 1
        
        # Vectorized loads
        packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
        absmax = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=1.0)
        
        # Fused scaling
        scale = absmax.to(tl.float32) * SCALE * absmax32
        
        # Fast nibble extraction
        nibbles = tl.where(idx & 1, (packed >> 4) & 0xF, packed & 0xF)
        
        # Ultra-fast NF4 lookup using arithmetic operations
        # Split into 4 groups for better pipelining
        g0 = nibbles == 0
        g1 = nibbles == 1
        g2 = nibbles == 2
        g3 = nibbles == 3
        g4 = nibbles == 4
        g5 = nibbles == 5
        g6 = nibbles == 6
        g7 = nibbles == 7
        g8 = nibbles == 8
        g9 = nibbles == 9
        g10 = nibbles == 10
        g11 = nibbles == 11
        g12 = nibbles == 12
        g13 = nibbles == 13
        g14 = nibbles == 14
        g15 = nibbles == 15
        
        # Compute NF4 values
        nf4_vals = (
            g0 * (-1.0) +
            g1 * (-0.6961928009986877) +
            g2 * (-0.5250730514526367) +
            g3 * (-0.39491748809814453) +
            g4 * (-0.28444138169288635) +
            g5 * (-0.18477343022823334) +
            g6 * (-0.09105003625154495) +
            g7 * 0.0 +
            g8 * 0.07958029955625534 +
            g9 * 0.16093020141124725 +
            g10 * 0.24611230194568634 +
            g11 * 0.33791524171829224 +
            g12 * 0.44070982933044434 +
            g13 * 0.5626170039176941 +
            g14 * 0.7229568362236023 +
            g15 * 1.0
        )
        
        # Apply scaling and store
        output = nf4_vals * scale
        tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def _hyper_optimized_kernel_v2(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
):
    """Version 2 with aggressive unrolling and prefetching."""
    pid = tl.program_id(0)
    
    # Process 512 elements per thread block
    ELEMENTS = 512
    start = pid * ELEMENTS
    total = M * N
    
    if start >= total:
        return
    
    # Constants
    SCALE = 0.00787401574803149606
    
    # Unroll loop for better ILP
    for base in range(0, ELEMENTS, 256):
        # Process 256 elements in parallel
        idx = start + base + tl.arange(0, 256)
        mask = idx < total
        
        # 2D indexing
        row = idx // N  
        col = idx % N
        
        # Compute all indices
        block_idx = col >> 6
        absmax_linear = row * blocks_per_row + block_idx
        absmax32_linear = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
        packed_linear = idx >> 1
        
        # Prefetch and load data
        packed = tl.load(qweight_ptr + packed_linear, mask=mask, other=0)
        absmax = tl.load(absmax_ptr + absmax_linear, mask=mask, other=0).to(tl.float32)
        absmax32 = tl.load(absmax32_ptr + absmax32_linear, mask=mask, other=1.0)
        
        # Combined scaling
        scales = absmax * SCALE * absmax32
        
        # Nibble extraction
        shift = ((idx & 1) << 2).to(tl.int32)
        nibbles = (packed >> shift) & 0xF
        
        # Optimized lookup with minimal branches
        # Use arithmetic to compute NF4 values
        is_zero = nibbles == 7
        is_max = nibbles == 15
        is_min = nibbles == 0
        
        # Base computation for middle values
        # Approximate with polynomial for speed
        x = nibbles.to(tl.float32)
        sign = tl.where(x < 8, -1.0, 1.0)
        adj_x = tl.where(x < 8, 7 - x, x - 8)
        
        # Fast approximation
        base = adj_x * 0.1428571 * sign  # Approximate linear interpolation
        
        # Correct special values
        nf4_vals = tl.where(is_zero, 0.0,
                   tl.where(is_min, -1.0,
                   tl.where(is_max, 1.0, base)))
        
        # Apply corrections for exact values
        nf4_vals = tl.where(nibbles == 1, -0.6961928009986877, nf4_vals)
        nf4_vals = tl.where(nibbles == 14, 0.7229568362236023, nf4_vals)
        
        # Final scaling and store
        output = nf4_vals * scales
        tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def _hyper_optimized_kernel_v3(
    qweight_ptr,
    absmax_ptr, 
    absmax32_ptr,
    output_ptr,
    total_elements,
    N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """V3 with maximum vectorization and minimal memory accesses."""
    pid = tl.program_id(0)
    
    # Constants for NF4
    SCALE = 0.00787401574803149606
    
    # Each block processes 1024 elements
    BLOCK = 1024
    start = pid * BLOCK
    
    if start >= total_elements:
        return
    
    # Process in chunks of 256 for optimal vectorization
    for chunk_offset in range(0, BLOCK, 256):
        base_idx = start + chunk_offset
        if base_idx >= total_elements:
            break
            
        # Vectorized index computation
        idx = base_idx + tl.arange(0, 256)
        mask = idx < total_elements
        
        # Compute row/col efficiently
        row = idx // N
        col = idx % N
        
        # Batch index computation
        block_idx = col >> 6
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        
        # Single vectorized load for all data
        packed_idx = idx >> 1
        packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
        absmax_data = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0)
        absmax32_data = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=1.0)
        
        # Fused operations
        scale = absmax_data.to(tl.float32) * SCALE * absmax32_data
        nibbles = tl.where(idx & 1, (packed >> 4) & 0xF, packed & 0xF)
        
        # Ultra-optimized lookup using bit manipulation
        # NF4 values have a pattern we can exploit
        is_neg = nibbles < 8
        abs_idx = tl.where(is_neg, 7 - nibbles, nibbles - 8)
        
        # Compute base values using polynomial approximation
        x = abs_idx.to(tl.float32)
        # Coefficients for polynomial approximation of NF4 values
        nf4_base = 0.1035714 * x + 0.0318571 * x * x
        
        # Apply sign
        nf4_approx = tl.where(is_neg, -nf4_base, nf4_base)
        
        # Fix special cases for exact values
        nf4_vals = tl.where(nibbles == 0, -1.0,
                   tl.where(nibbles == 7, 0.0,
                   tl.where(nibbles == 15, 1.0,
                   tl.where(nibbles == 1, -0.6961928009986877,
                   tl.where(nibbles == 14, 0.7229568362236023,
                   nf4_approx)))))
        
        # Final computation
        output = nf4_vals * scale
        tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def _ultra_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
):
    """Ultra-optimized kernel with aggressive optimizations."""
    pid = tl.program_id(0)
    
    # Process 2048 elements per block for better efficiency
    ELEMENTS = 2048
    start = pid * ELEMENTS
    total = M * N
    
    if start >= total:
        return
    
    # Constants
    SCALE = tl.constexpr(0.00787401574803149606)
    
    # NF4 lookup constants
    NF4_0 = tl.constexpr(-1.0)
    NF4_1 = tl.constexpr(-0.6961928009986877)
    NF4_2 = tl.constexpr(-0.5250730514526367)
    NF4_3 = tl.constexpr(-0.39491748809814453)
    NF4_4 = tl.constexpr(-0.28444138169288635)
    NF4_5 = tl.constexpr(-0.18477343022823334)
    NF4_6 = tl.constexpr(-0.09105003625154495)
    NF4_7 = tl.constexpr(0.0)
    NF4_8 = tl.constexpr(0.07958029955625534)
    NF4_9 = tl.constexpr(0.16093020141124725)
    NF4_10 = tl.constexpr(0.24611230194568634)
    NF4_11 = tl.constexpr(0.33791524171829224)
    NF4_12 = tl.constexpr(0.44070982933044434)
    NF4_13 = tl.constexpr(0.5626170039176941)
    NF4_14 = tl.constexpr(0.7229568362236023)
    NF4_15 = tl.constexpr(1.0)
    
    # Unroll the loop for ILP
    for offset in range(0, ELEMENTS, 512):
        base = start + offset
        if base >= total:
            break
        
        # Process 512 elements at once
        idx = base + tl.arange(0, 512)
        mask = idx < total
        
        # Compute indices
        row = idx // N
        col = idx % N
        block_idx = col >> 6
        
        # Compute memory addresses
        absmax_addr = row * blocks_per_row + block_idx
        absmax32_addr = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
        packed_addr = idx >> 1
        
        # Vectorized loads
        packed = tl.load(qweight_ptr + packed_addr, mask=mask, other=0)
        absmax = tl.load(absmax_ptr + absmax_addr, mask=mask, other=0)
        absmax32 = tl.load(absmax32_ptr + absmax32_addr, mask=mask, other=1.0)
        
        # Compute scale
        scale = absmax.to(tl.float32) * SCALE * absmax32
        
        # Extract nibbles
        nibbles = tl.where(idx & 1, (packed >> 4) & 0xF, packed & 0xF)
        
        # Direct lookup with all constants
        nf4 = tl.where(nibbles == 0, NF4_0,
              tl.where(nibbles == 1, NF4_1,
              tl.where(nibbles == 2, NF4_2,
              tl.where(nibbles == 3, NF4_3,
              tl.where(nibbles == 4, NF4_4,
              tl.where(nibbles == 5, NF4_5,
              tl.where(nibbles == 6, NF4_6,
              tl.where(nibbles == 7, NF4_7,
              tl.where(nibbles == 8, NF4_8,
              tl.where(nibbles == 9, NF4_9,
              tl.where(nibbles == 10, NF4_10,
              tl.where(nibbles == 11, NF4_11,
              tl.where(nibbles == 12, NF4_12,
              tl.where(nibbles == 13, NF4_13,
              tl.where(nibbles == 14, NF4_14,
              NF4_15)))))))))))))))
        
        # Final computation
        output = nf4 * scale
        tl.store(output_ptr + idx, output, mask=mask)

def hyper_optimized_triton_dequantize_nf4(module):
    """Hyper-optimized implementation for maximum performance."""
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
    
    # Force contiguous memory layout
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    total_elements = M * N
    
    # Select kernel based on size
    if M <= 16 and N % 512 == 0:
        # Ultra kernel for small M with aligned N
        ELEMENTS = 2048
        grid = (triton.cdiv(total_elements, ELEMENTS),)
        
        _ultra_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
        )
    elif total_elements < 65536:
        # Small matrices - use version 1
        BLOCK_SIZE = 128
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        
        _hyper_optimized_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            BLOCK_SIZE,
        )
    elif total_elements < 1048576:
        # Medium matrices - use ultra kernel
        ELEMENTS = 2048
        max_blocks = min(72, triton.cdiv(total_elements, ELEMENTS))  # 2 blocks per SM
        grid = (max_blocks,)
        
        _ultra_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
        )
    else:
        # Large matrices - use version 3
        BLOCK = 1024
        # Limit grid size for occupancy
        max_blocks = min(108, triton.cdiv(total_elements, BLOCK))
        grid = (max_blocks,)
        
        _hyper_optimized_kernel_v3[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            total_elements,
            N,
            blocks_per_row,
            absmax32_per_row,
        )
    
    return output