import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _warp_speed_kernel(
    qweight_ptr,
    absmax_ptr, 
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Warp-speed NF4 kernel with extreme optimizations."""
    
    # Grid-stride loop with aggressive parallelization
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Constants
    SCALE: tl.constexpr = 0.00787401574803149606
    BLOCKS_PER_ITER: tl.constexpr = 16  # Process 16 blocks per iteration
    
    total_blocks = M * blocks_per_row
    
    # Grid-stride loop
    for iter_start in range(pid * BLOCKS_PER_ITER, total_blocks, num_programs * BLOCKS_PER_ITER):
        
        # Process 16 blocks in this iteration
        for block_offset in range(BLOCKS_PER_ITER):
            block_id = iter_start + block_offset
            if block_id >= total_blocks:
                return
            
            row = block_id // blocks_per_row
            block_idx = block_id % blocks_per_row
            col_base = block_idx * 64
            
            if col_base >= N:
                continue
            
            # Load scales
            absmax = tl.load(absmax_ptr + block_id).to(tl.float32)
            absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx)
            scale = absmax * SCALE * absmax32
            
            base_offset = row * N + col_base
            
            # Process all 64 elements using maximum vectorization
            # Load 32 bytes (64 nibbles) in 4 vector loads of 8 bytes each
            for vec in range(4):
                vec_offset = vec * 16
                if col_base + vec_offset >= N:
                    break
                
                # Load 8 packed bytes at once
                packed_idx = (base_offset + vec_offset) >> 1
                packed_vec = packed_idx + tl.arange(0, 8)
                packed = tl.load(qweight_ptr + packed_vec, eviction_policy="evict_first")
                
                # Extract all nibbles at once
                nibbles_even = packed & 0xF
                nibbles_odd = (packed >> 4) & 0xF
                
                # Ultra-fast lookup using arithmetic
                # For values 0-7: linear interpolation with corrections
                # For values 8-15: linear interpolation with corrections
                
                # Even nibbles
                base_even = tl.where(
                    nibbles_even < 8,
                    -1.0 + nibbles_even.to(tl.float32) * 0.142857,  # Approximate for negative
                    (nibbles_even.to(tl.float32) - 7.5) * 0.133333  # Approximate for positive
                )
                
                # Apply exact corrections for critical values
                vals_even = tl.where(nibbles_even == 0, -1.0,
                           tl.where(nibbles_even == 1, -0.6961928,
                           tl.where(nibbles_even == 2, -0.5250731,
                           tl.where(nibbles_even == 3, -0.3949175,
                           tl.where(nibbles_even == 4, -0.2844414,
                           tl.where(nibbles_even == 5, -0.1847734,
                           tl.where(nibbles_even == 6, -0.09105004,
                           tl.where(nibbles_even == 7, 0.0,
                           tl.where(nibbles_even == 8, 0.0795803,
                           tl.where(nibbles_even == 9, 0.1609302,
                           tl.where(nibbles_even == 10, 0.2461123,
                           tl.where(nibbles_even == 11, 0.3379152,
                           tl.where(nibbles_even == 12, 0.4407098,
                           tl.where(nibbles_even == 13, 0.562617,
                           tl.where(nibbles_even == 14, 0.7229568,
                           1.0)))))))))))))))
                
                # Odd nibbles - same approach
                vals_odd = tl.where(nibbles_odd == 0, -1.0,
                          tl.where(nibbles_odd == 1, -0.6961928,
                          tl.where(nibbles_odd == 2, -0.5250731,
                          tl.where(nibbles_odd == 3, -0.3949175,
                          tl.where(nibbles_odd == 4, -0.2844414,
                          tl.where(nibbles_odd == 5, -0.1847734,
                          tl.where(nibbles_odd == 6, -0.09105004,
                          tl.where(nibbles_odd == 7, 0.0,
                          tl.where(nibbles_odd == 8, 0.0795803,
                          tl.where(nibbles_odd == 9, 0.1609302,
                          tl.where(nibbles_odd == 10, 0.2461123,
                          tl.where(nibbles_odd == 11, 0.3379152,
                          tl.where(nibbles_odd == 12, 0.4407098,
                          tl.where(nibbles_odd == 13, 0.562617,
                          tl.where(nibbles_odd == 14, 0.7229568,
                          1.0)))))))))))))))
                
                # Scale and store
                out_even = vals_even * scale
                out_odd = vals_odd * scale
                
                # Interleaved store with bounds checking
                out_base = base_offset + vec_offset
                
                for i in range(8):
                    idx_even = out_base + i * 2
                    idx_odd = out_base + i * 2 + 1
                    
                    if idx_even < row * N + N:
                        tl.store(output_ptr + idx_even, out_even[i], eviction_policy="evict_first")
                    if idx_odd < row * N + N:
                        tl.store(output_ptr + idx_odd, out_odd[i], eviction_policy="evict_first")

def warp_speed_dequantize_nf4(module):
    """Warp-speed NF4 dequantization."""
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
    
    # Optimal grid for T4
    total_blocks = M * blocks_per_row
    BLOCKS_PER_ITER = 16
    grid_size = min((total_blocks + BLOCKS_PER_ITER - 1) // BLOCKS_PER_ITER, 320)
    
    _warp_speed_kernel[(grid_size,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        num_warps=8,  # Maximum warps
        num_stages=2,
    )
    
    return output

# Export
triton_dequantize_nf4 = warp_speed_dequantize_nf4