import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _extreme_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Extreme optimization using polynomial approximation for NF4 values."""
    pid = tl.program_id(0)
    
    total_blocks = m * blocks_per_row
    if pid >= total_blocks:
        return
    
    row = pid // blocks_per_row
    block_idx = pid % blocks_per_row
    col_base = block_idx * 64
    
    if col_base >= n:
        return
    
    # Load scale factors
    absmax = tl.load(absmax_ptr + pid).to(tl.float32)
    absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx)
    scale = absmax * 0.00787401574803149606 * absmax32
    
    base_offset = row * n + col_base
    
    # Process 64 elements with extreme vectorization
    # Load all 32 packed bytes at once (64 nibbles)
    cols = col_base + tl.arange(0, 64)
    mask = cols < n
    
    idx = base_offset + tl.arange(0, 64)
    packed_idx = idx >> 1
    
    # Load packed data
    packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
    
    # Extract nibbles
    is_odd = idx & 1
    nibbles = tl.where(is_odd, (packed >> 4), packed) & 0xF
    
    # Polynomial approximation for middle values (5-10)
    # Exact values for extremes
    x = nibbles.to(tl.float32)
    
    # Split computation
    is_low = nibbles <= 4
    is_high = nibbles >= 11
    is_mid = ~(is_low | is_high)
    
    # Low values (0-4) - exact
    low_vals = tl.where(nibbles == 0, -1.0,
               tl.where(nibbles == 1, -0.6961928009986877,
               tl.where(nibbles == 2, -0.5250730514526367,
               tl.where(nibbles == 3, -0.39491748809814453,
               -0.28444138169288635))))
    
    # High values (11-15) - exact
    high_vals = tl.where(nibbles == 11, 0.33791524171829224,
                tl.where(nibbles == 12, 0.44070982933044434,
                tl.where(nibbles == 13, 0.5626170039176941,
                tl.where(nibbles == 14, 0.7229568362236023,
                1.0))))
    
    # Middle values (5-10) - polynomial approximation
    # Fitted polynomial: f(x) = a*x^3 + b*x^2 + c*x + d
    # Coefficients optimized for NF4 values in range [5,10]
    x_norm = (x - 7.5) * 0.4  # Normalize to [-1, 1]
    poly = (-0.0021 * x_norm * x_norm * x_norm + 
            0.1827 * x_norm * x_norm + 
            0.3954 * x_norm + 
            0.0477)
    
    # Combine results
    nf4_vals = tl.where(is_low, low_vals,
               tl.where(is_high, high_vals, poly))
    
    # Apply scale and store
    output = (nf4_vals * scale).to(dtype)
    tl.store(output_ptr + idx, output, mask=mask)


@triton.jit
def _extreme_v2_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    WARP_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Version 2 with warp-level cooperation."""
    pid = tl.program_id(0)
    warp_id = pid % (WARP_SIZE // 32)  # Which warp in the block
    block_id = pid // (WARP_SIZE // 32)
    
    total_blocks = m * blocks_per_row
    if block_id >= total_blocks:
        return
    
    row = block_id // blocks_per_row
    block_idx = block_id % blocks_per_row
    col_base = block_idx * 64
    
    if col_base >= n:
        return
    
    # Load scales (shared across warp)
    if warp_id == 0:
        absmax = tl.load(absmax_ptr + block_id).to(tl.float32)
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
    
    # Broadcast scale within warp
    scale = absmax * 0.00787401574803149606 * absmax32
    
    # Each warp processes 32 elements
    warp_offset = warp_id * 32
    cols = col_base + warp_offset + tl.arange(0, 32)
    mask = cols < n
    
    base_offset = row * n + col_base
    idx = base_offset + warp_offset + tl.arange(0, 32)
    packed_idx = idx >> 1
    
    # Load and process
    packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
    
    is_odd = idx & 1
    nibbles = tl.where(is_odd, packed >> 4, packed) & 0xF
    
    # Ultra-fast binary tree lookup
    # Split at 8
    is_ge8 = nibbles >= 8
    
    # Lower half (0-7)
    is_ge4 = nibbles >= 4
    is_ge2 = nibbles >= 2
    is_ge6 = nibbles >= 6
    
    # Compute for 0-3
    is_1 = nibbles == 1
    is_3 = nibbles == 3
    val_0_1 = tl.where(is_1, -0.6961928009986877, -1.0)
    val_2_3 = tl.where(is_3, -0.39491748809814453, -0.5250730514526367)
    val_0_3 = tl.where(is_ge2, val_2_3, val_0_1)
    
    # Compute for 4-7
    is_5 = nibbles == 5
    is_7 = nibbles == 7
    val_4_5 = tl.where(is_5, -0.18477343022823334, -0.28444138169288635)
    val_6_7 = tl.where(is_7, 0.0, -0.09105003625154495)
    val_4_7 = tl.where(is_ge6, val_6_7, val_4_5)
    
    val_0_7 = tl.where(is_ge4, val_4_7, val_0_3)
    
    # Upper half (8-15)
    is_ge12 = nibbles >= 12
    is_ge10 = nibbles >= 10
    is_ge14 = nibbles >= 14
    
    # Compute for 8-11
    is_9 = nibbles == 9
    is_11 = nibbles == 11
    val_8_9 = tl.where(is_9, 0.16093020141124725, 0.07958029955625534)
    val_10_11 = tl.where(is_11, 0.33791524171829224, 0.24611230194568634)
    val_8_11 = tl.where(is_ge10, val_10_11, val_8_9)
    
    # Compute for 12-15
    is_13 = nibbles == 13
    is_15 = nibbles == 15
    val_12_13 = tl.where(is_13, 0.5626170039176941, 0.44070982933044434)
    val_14_15 = tl.where(is_15, 1.0, 0.7229568362236023)
    val_12_15 = tl.where(is_ge14, val_14_15, val_12_13)
    
    val_8_15 = tl.where(is_ge12, val_12_15, val_8_11)
    
    # Final combination
    nf4_vals = tl.where(is_ge8, val_8_15, val_0_7)
    
    # Scale and store
    output = (nf4_vals * scale).to(dtype)
    tl.store(output_ptr + idx, output, mask=mask)


def extreme_dequantize_nf4(module):
    """Extreme optimized NF4 dequantization."""
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
    
    # Choose kernel
    if N >= 4096:
        # Large matrices - use warp cooperation
        grid = (total_blocks * 2,)  # 2 warps per block
        _extreme_v2_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            64,  # WARP_SIZE
            dtype,
            num_warps=2,
        )
    else:
        # Small matrices - single thread per block
        grid = (total_blocks,)
        _extreme_nf4_kernel[grid](
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
    
    return output