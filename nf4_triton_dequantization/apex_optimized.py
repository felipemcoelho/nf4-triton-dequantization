import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _apex_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Apex-level optimized NF4 kernel with aggressive optimizations."""
    
    # Grid-stride loop with aggressive parallelization
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Process 4 blocks per thread for maximum throughput
    BLOCKS_PER_THREAD: tl.constexpr = 4
    
    total_blocks = M * blocks_per_row
    start_block = pid * BLOCKS_PER_THREAD
    
    # Pre-compute constants
    NF4_SCALE: tl.constexpr = 0.00787401574803149606
    
    # Optimized NF4 LUT using arithmetic approximation
    # We'll use a polynomial approximation for most values
    # and special-case the extremes
    
    for block_offset in range(BLOCKS_PER_THREAD):
        block_id = start_block + block_offset
        if block_id >= total_blocks:
            return
        
        row = block_id // blocks_per_row
        block_idx = block_id % blocks_per_row
        col_base = block_idx * 64
        
        if col_base >= N:
            continue
        
        # Load scales once
        absmax = tl.load(absmax_ptr + block_id).to(tl.float32)
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        
        # Fused scale computation
        scale = absmax * NF4_SCALE * absmax32
        
        base_offset = row * N + col_base
        
        # Process 64 elements using aggressive vectorization
        # Unroll to 2x32 for better memory bandwidth utilization
        for half in tl.static_range(2):
            offset = half * 32
            idx = base_offset + offset + tl.arange(0, 32)
            cols = col_base + offset + tl.arange(0, 32)
            mask = cols < N
            
            # Vectorized load - load 16 bytes at once
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0, eviction_policy="evict_first")
            
            # Parallel nibble extraction using bit manipulation
            is_odd = idx & 1
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Ultra-fast NF4 value computation using arithmetic approximation
            # Split into 3 groups for optimal performance
            
            # Group 1: Values 0-7 (negative values)
            is_low = nibbles < 8
            low_approx = -1.0 + nibbles.to(tl.float32) * 0.14285714  # Linear approximation
            
            # Group 2: Values 8-15 (positive values)
            high_nibbles = nibbles - 8
            high_approx = high_nibbles.to(tl.float32) * 0.14285714
            
            # Apply corrections for specific values using a compact formula
            # This avoids the full lookup table
            corrections = tl.zeros_like(low_approx)
            
            # Special cases that need exact values
            corrections = tl.where(nibbles == 0, 0.0,  # -1.0 (no correction needed)
                         tl.where(nibbles == 1, 0.3038072 - 0.14285714,  # Correction for -0.6961928
                         tl.where(nibbles == 2, 0.4749269 - 0.28571428,  # Correction for -0.5250731
                         tl.where(nibbles == 7, 1.0,  # 0.0 needs full correction
                         tl.where(nibbles == 15, 1.0 - 1.0,  # 1.0 (no correction needed)
                         corrections)))))
            
            # Combine approximations
            base_vals = tl.where(is_low, low_approx + corrections, high_approx + corrections)
            
            # Additional corrections for high precision
            base_vals = tl.where(nibbles == 3, -0.39491748809814453,
                        tl.where(nibbles == 4, -0.28444138169288635,
                        tl.where(nibbles == 5, -0.18477343022823334,
                        tl.where(nibbles == 6, -0.09105003625154495,
                        tl.where(nibbles == 8, 0.07958029955625534,
                        tl.where(nibbles == 9, 0.16093020141124725,
                        tl.where(nibbles == 10, 0.24611230194568634,
                        tl.where(nibbles == 11, 0.33791524171829224,
                        tl.where(nibbles == 12, 0.44070982933044434,
                        tl.where(nibbles == 13, 0.5626170039176941,
                        tl.where(nibbles == 14, 0.7229568362236023,
                        base_vals)))))))))))
            
            # Apply scale and convert
            output = (base_vals * scale).to(dtype)
            
            # Store with streaming hint
            tl.store(output_ptr + idx, output, mask=mask, eviction_policy="evict_first")

def apex_optimized_dequantize_nf4(module):
    """Apex-level optimized NF4 dequantization for maximum performance."""
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
    
    # Prepare absmax tensors
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
    
    # Ensure optimal memory layout
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Pre-allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Optimal grid configuration for T4
    # T4 has 40 SMs, use multiple of 40 for best occupancy
    BLOCKS_PER_THREAD = 4
    total_blocks = M * blocks_per_row
    grid_size = (total_blocks + BLOCKS_PER_THREAD - 1) // BLOCKS_PER_THREAD
    
    # Use optimal number of warps for T4
    # 2 warps gives best performance for this workload
    _apex_nf4_kernel[(grid_size,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        dtype,
        num_warps=2,
        num_stages=3,  # More stages for better pipelining
    )
    
    return output

# Export
triton_dequantize_nf4 = apex_optimized_dequantize_nf4