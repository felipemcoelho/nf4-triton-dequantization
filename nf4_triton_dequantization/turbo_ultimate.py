import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _turbo_ultimate_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    total_blocks,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Turbo Ultimate kernel - the fastest possible implementation."""
    
    # Super-aggressive parallelization - 8 blocks per thread
    pid = tl.program_id(0)
    BLOCKS_PER_THREAD: tl.constexpr = 8
    
    # Pre-compute all constants
    NF4_SCALE: tl.constexpr = 0.00787401574803149606
    
    # Ultra-optimized NF4 LUT split into two halves
    nf4_low_0 = tl.inline_const_val(-1.0)
    nf4_low_1 = tl.inline_const_val(-0.6961928009986877)
    nf4_low_2 = tl.inline_const_val(-0.5250730514526367)
    nf4_low_3 = tl.inline_const_val(-0.39491748809814453)
    nf4_low_4 = tl.inline_const_val(-0.28444138169288635)
    nf4_low_5 = tl.inline_const_val(-0.18477343022823334)
    nf4_low_6 = tl.inline_const_val(-0.09105003625154495)
    nf4_low_7 = tl.inline_const_val(0.0)
    
    nf4_high_0 = tl.inline_const_val(0.07958029955625534)
    nf4_high_1 = tl.inline_const_val(0.16093020141124725)
    nf4_high_2 = tl.inline_const_val(0.24611230194568634)
    nf4_high_3 = tl.inline_const_val(0.33791524171829224)
    nf4_high_4 = tl.inline_const_val(0.44070982933044434)
    nf4_high_5 = tl.inline_const_val(0.5626170039176941)
    nf4_high_6 = tl.inline_const_val(0.7229568362236023)
    nf4_high_7 = tl.inline_const_val(1.0)
    
    start_block = pid * BLOCKS_PER_THREAD
    
    # Process 8 blocks per thread
    for block_offset in tl.static_range(BLOCKS_PER_THREAD):
        block_id = start_block + block_offset
        if block_id >= total_blocks:
            return
        
        # Decode position
        row = block_id // blocks_per_row
        block_idx = block_id % blocks_per_row
        col_base = block_idx * 64
        
        if col_base >= N:
            continue
        
        # Load and compute scale
        absmax = tl.load(absmax_ptr + block_id).to(tl.float32)
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax * NF4_SCALE * absmax32
        
        base_offset = row * N + col_base
        
        # Process 64 elements in one shot using maximum vectorization
        # Load all 32 packed bytes at once
        packed_base = (base_offset >> 1) + tl.arange(0, 32)
        packed = tl.load(qweight_ptr + packed_base, eviction_policy="evict_first")
        
        # Extract all 64 nibbles in parallel
        # Even indices
        nibbles_even = packed & 0xF
        # Odd indices  
        nibbles_odd = (packed >> 4) & 0xF
        
        # Process even nibbles (0, 2, 4, ..., 62)
        vals_even = tl.where(nibbles_even == 0, nf4_low_0,
                    tl.where(nibbles_even == 1, nf4_low_1,
                    tl.where(nibbles_even == 2, nf4_low_2,
                    tl.where(nibbles_even == 3, nf4_low_3,
                    tl.where(nibbles_even == 4, nf4_low_4,
                    tl.where(nibbles_even == 5, nf4_low_5,
                    tl.where(nibbles_even == 6, nf4_low_6,
                    tl.where(nibbles_even == 7, nf4_low_7,
                    tl.where(nibbles_even == 8, nf4_high_0,
                    tl.where(nibbles_even == 9, nf4_high_1,
                    tl.where(nibbles_even == 10, nf4_high_2,
                    tl.where(nibbles_even == 11, nf4_high_3,
                    tl.where(nibbles_even == 12, nf4_high_4,
                    tl.where(nibbles_even == 13, nf4_high_5,
                    tl.where(nibbles_even == 14, nf4_high_6,
                    nf4_high_7)))))))))))))))
        
        # Process odd nibbles (1, 3, 5, ..., 63)
        vals_odd = tl.where(nibbles_odd == 0, nf4_low_0,
                   tl.where(nibbles_odd == 1, nf4_low_1,
                   tl.where(nibbles_odd == 2, nf4_low_2,
                   tl.where(nibbles_odd == 3, nf4_low_3,
                   tl.where(nibbles_odd == 4, nf4_low_4,
                   tl.where(nibbles_odd == 5, nf4_low_5,
                   tl.where(nibbles_odd == 6, nf4_low_6,
                   tl.where(nibbles_odd == 7, nf4_low_7,
                   tl.where(nibbles_odd == 8, nf4_high_0,
                   tl.where(nibbles_odd == 9, nf4_high_1,
                   tl.where(nibbles_odd == 10, nf4_high_2,
                   tl.where(nibbles_odd == 11, nf4_high_3,
                   tl.where(nibbles_odd == 12, nf4_high_4,
                   tl.where(nibbles_odd == 13, nf4_high_5,
                   tl.where(nibbles_odd == 14, nf4_high_6,
                   nf4_high_7)))))))))))))))
        
        # Scale and convert
        out_even = (vals_even * scale).to(dtype)
        out_odd = (vals_odd * scale).to(dtype)
        
        # Store interleaved results
        # Check bounds for the last block
        if col_base + 63 < N:
            # Fast path - no bounds checking needed
            for i in tl.static_range(32):
                tl.store(output_ptr + base_offset + i * 2, out_even[i], eviction_policy="evict_first")
                tl.store(output_ptr + base_offset + i * 2 + 1, out_odd[i], eviction_policy="evict_first")
        else:
            # Slow path - need bounds checking
            for i in tl.static_range(32):
                if col_base + i * 2 < N:
                    tl.store(output_ptr + base_offset + i * 2, out_even[i])
                if col_base + i * 2 + 1 < N:
                    tl.store(output_ptr + base_offset + i * 2 + 1, out_odd[i])

def turbo_ultimate_dequantize_nf4(module):
    """Turbo Ultimate NF4 dequantization - absolutely the fastest."""
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
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Pre-allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch with optimal configuration
    total_blocks = M * blocks_per_row
    BLOCKS_PER_THREAD = 8
    grid_size = (total_blocks + BLOCKS_PER_THREAD - 1) // BLOCKS_PER_THREAD
    
    # Optimal configuration for T4 GPU
    _turbo_ultimate_kernel[(grid_size,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        total_blocks,
        M, N,
        blocks_per_row,
        absmax32_per_row,
        dtype,
        num_warps=4,  # More warps for this complex kernel
        num_stages=4,  # Maximum pipelining
    )
    
    return output

# Export
triton_dequantize_nf4 = turbo_ultimate_dequantize_nf4