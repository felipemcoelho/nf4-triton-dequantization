import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _turbo_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Turbo-optimized kernel with aggressive unrolling."""
    pid = tl.program_id(0)
    
    # Early exit
    total_blocks = m * blocks_per_row
    if pid >= total_blocks:
        return
    
    # Decode block position
    row = pid // blocks_per_row
    block_idx = pid % blocks_per_row
    col_base = block_idx << 6  # * 64
    
    # Boundary check
    if col_base >= n:
        return
    
    # Load scales once
    absmax = tl.load(absmax_ptr + pid).to(tl.float32)
    absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx)
    scale = absmax * 0.00787401574803149606 * absmax32
    
    # Base offset
    base_offset = row * n + col_base
    
    # NF4 lookup - use conditional approach for speed
    # Process all 64 elements with full unrolling
    cols = col_base + tl.arange(0, 64)
    mask = cols < n
    
    # Calculate all indices at once
    idx = base_offset + tl.arange(0, 64)
    packed_idx = idx >> 1
    
    # Load all packed data
    packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
    
    # Extract all nibbles
    is_odd = idx & 1
    nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
    
    # Ultra-fast conditional lookup without gather
    # Each condition evaluates to 0 or the value
    nf4_vals = tl.where(nibbles == 0, -1.0,
               tl.where(nibbles == 1, -0.6961928009986877,
               tl.where(nibbles == 2, -0.5250730514526367,
               tl.where(nibbles == 3, -0.39491748809814453,
               tl.where(nibbles == 4, -0.28444138169288635,
               tl.where(nibbles == 5, -0.18477343022823334,
               tl.where(nibbles == 6, -0.09105003625154495,
               tl.where(nibbles == 7, 0.0,
               tl.where(nibbles == 8, 0.07958029955625534,
               tl.where(nibbles == 9, 0.16093020141124725,
               tl.where(nibbles == 10, 0.24611230194568634,
               tl.where(nibbles == 11, 0.33791524171829224,
               tl.where(nibbles == 12, 0.44070982933044434,
               tl.where(nibbles == 13, 0.5626170039176941,
               tl.where(nibbles == 14, 0.7229568362236023,
               1.0)))))))))))))))
    
    # Apply scale and store all at once
    output = (nf4_vals * scale).to(dtype)
    tl.store(output_ptr + idx, output, mask=mask)


@triton.jit
def _turbo_split_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Split processing for better cache usage."""
    pid = tl.program_id(0)
    
    total_blocks = m * blocks_per_row
    if pid >= total_blocks:
        return
    
    row = pid // blocks_per_row
    block_idx = pid % blocks_per_row
    col_base = block_idx * 64
    
    if col_base >= n:
        return
    
    # Load scales
    absmax = tl.load(absmax_ptr + pid).to(tl.float32)
    absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx)
    scale = absmax * 0.00787401574803149606 * absmax32
    
    base_offset = row * n + col_base
    
    # NF4 LUT split for cache efficiency
    nf4_0_7 = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0
    ])
    nf4_8_15 = tl.inline_const_array([
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Process in two 32-element halves
    for half in range(2):
        offset = half * 32
        cols = col_base + offset + tl.arange(0, 32)
        mask = cols < n
        
        idx = base_offset + offset + tl.arange(0, 32)
        packed_idx = idx >> 1
        
        packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
        
        is_odd = idx & 1
        nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
        
        # Split lookup
        is_low = nibbles < 8
        low_idx = nibbles
        high_idx = nibbles - 8
        
        low_vals = tl.gather(nf4_0_7, low_idx)
        high_vals = tl.gather(nf4_8_15, high_idx)
        
        nf4_vals = tl.where(is_low, low_vals, high_vals)
        
        output = (nf4_vals * scale).to(dtype)
        tl.store(output_ptr + idx, output, mask=mask)


def turbo_dequantize_nf4(module):
    """Turbo-optimized NF4 dequantization."""
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
    
    total_blocks = M * blocks_per_row
    grid = (total_blocks,)
    
    # Choose kernel based on size
    if N <= 2048:
        # Small matrices - use conditional lookup
        _turbo_nf4_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            dtype,
            num_warps=1,
        )
    else:
        # Large matrices - use split processing
        _turbo_split_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            dtype,
            num_warps=2,
        )
    
    return output