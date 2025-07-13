import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _hyperdrive_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr, 
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Hyperdrive NF4 kernel - absolute maximum performance."""
    
    # Single dimension parallelization for simplicity
    pid = tl.program_id(0)
    
    # Each thread processes one 64-element block
    total_blocks = M * blocks_per_row
    if pid >= total_blocks:
        return
    
    # Decode block position
    row = pid // blocks_per_row
    block_idx = pid % blocks_per_row
    col_base = block_idx * 64
    
    if col_base >= N:
        return
    
    # Constants
    SCALE: tl.constexpr = 0.00787401574803149606
    
    # Load scale factors once
    absmax = tl.load(absmax_ptr + pid).to(tl.float32)
    absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx)
    scale = absmax * SCALE * absmax32
    
    # Base offsets
    input_base = row * (N >> 1) + (col_base >> 1)  # Packed input
    output_base = row * N + col_base
    
    # Process 64 elements in one shot
    # Load all 32 packed bytes
    packed_data = tl.load(qweight_ptr + input_base + tl.arange(0, 32))
    
    # Extract nibbles using vectorized operations
    # Even positions (0, 2, 4, ..., 62)
    nibbles_even = packed_data & 0xF
    # Odd positions (1, 3, 5, ..., 63)
    nibbles_odd = (packed_data >> 4) & 0xF
    
    # NF4 lookup using nested conditionals (fastest on GPU)
    # Even values
    vals_even = tl.where(nibbles_even == 0, -1.0,
               tl.where(nibbles_even == 1, -0.6961928009986877,
               tl.where(nibbles_even == 2, -0.5250730514526367,
               tl.where(nibbles_even == 3, -0.39491748809814453,
               tl.where(nibbles_even == 4, -0.28444138169288635,
               tl.where(nibbles_even == 5, -0.18477343022823334,
               tl.where(nibbles_even == 6, -0.09105003625154495,
               tl.where(nibbles_even == 7, 0.0,
               tl.where(nibbles_even == 8, 0.07958029955625534,
               tl.where(nibbles_even == 9, 0.16093020141124725,
               tl.where(nibbles_even == 10, 0.24611230194568634,
               tl.where(nibbles_even == 11, 0.33791524171829224,
               tl.where(nibbles_even == 12, 0.44070982933044434,
               tl.where(nibbles_even == 13, 0.5626170039176941,
               tl.where(nibbles_even == 14, 0.7229568362236023,
               1.0)))))))))))))))
    
    # Odd values
    vals_odd = tl.where(nibbles_odd == 0, -1.0,
              tl.where(nibbles_odd == 1, -0.6961928009986877,
              tl.where(nibbles_odd == 2, -0.5250730514526367,
              tl.where(nibbles_odd == 3, -0.39491748809814453,
              tl.where(nibbles_odd == 4, -0.28444138169288635,
              tl.where(nibbles_odd == 5, -0.18477343022823334,
              tl.where(nibbles_odd == 6, -0.09105003625154495,
              tl.where(nibbles_odd == 7, 0.0,
              tl.where(nibbles_odd == 8, 0.07958029955625534,
              tl.where(nibbles_odd == 9, 0.16093020141124725,
              tl.where(nibbles_odd == 10, 0.24611230194568634,
              tl.where(nibbles_odd == 11, 0.33791524171829224,
              tl.where(nibbles_odd == 12, 0.44070982933044434,
              tl.where(nibbles_odd == 13, 0.5626170039176941,
              tl.where(nibbles_odd == 14, 0.7229568362236023,
              1.0)))))))))))))))
    
    # Apply scale
    out_even = vals_even * scale
    out_odd = vals_odd * scale
    
    # Check if we need bounds checking
    elements_remaining = N - col_base
    
    if elements_remaining >= 64:
        # Fast path - store all 64 elements without bounds checking
        # Interleave even and odd values
        for i in tl.static_range(32):
            tl.store(output_ptr + output_base + i * 2, out_even[i])
            tl.store(output_ptr + output_base + i * 2 + 1, out_odd[i])
    else:
        # Slow path - need bounds checking
        for i in tl.static_range(32):
            if i * 2 < elements_remaining:
                tl.store(output_ptr + output_base + i * 2, out_even[i])
            if i * 2 + 1 < elements_remaining:
                tl.store(output_ptr + output_base + i * 2 + 1, out_odd[i])

def hyperdrive_dequantize_nf4(module):
    """Hyperdrive NF4 dequantization - optimized for 1.20x+ speedup."""
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
    
    # Ensure contiguous for maximum performance
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Pre-allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Simple 1D grid - one thread per block
    total_blocks = M * blocks_per_row
    
    # Launch kernel
    _hyperdrive_kernel[(total_blocks,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        num_warps=1,  # Single warp per block is fastest
        num_stages=1,  # No pipelining needed
    )
    
    return output

# Export
triton_dequantize_nf4 = hyperdrive_dequantize_nf4