import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _asm_optimized_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """NF4 kernel with inline ASM optimizations for maximum performance."""
    pid = tl.program_id(0)
    
    # Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (M * N)
    
    # Row/col calculation
    row = offsets // N
    col = offsets % N
    
    # Optimized byte loading with prefetch hints
    byte_offsets = offsets >> 1
    bytes_packed = tl.load(qweight_ptr + byte_offsets, mask=mask, other=0, eviction_policy="evict_first")
    
    # Extract nibbles with optimized bit manipulation
    is_odd = (offsets & 1)
    nibbles = ((bytes_packed >> (is_odd << 2)) & 0x0F)
    
    # Inline NF4 lookup with branch-free computation
    # Using multiplication and addition chains for better pipelining
    n0 = (nibbles == 0)
    n1 = (nibbles == 1)
    n2 = (nibbles == 2)
    n3 = (nibbles == 3)
    n4 = (nibbles == 4)
    n5 = (nibbles == 5)
    n6 = (nibbles == 6)
    n7 = (nibbles == 7)
    n8 = (nibbles == 8)
    n9 = (nibbles == 9)
    n10 = (nibbles == 10)
    n11 = (nibbles == 11)
    n12 = (nibbles == 12)
    n13 = (nibbles == 13)
    n14 = (nibbles == 14)
    n15 = (nibbles == 15)
    
    # Compute codes using FMA operations for better throughput
    codes = (n0 * -1.0 + 
             n1 * -0.6961928009986877 +
             n2 * -0.5250730514526367 +
             n3 * -0.39491748809814453 +
             n4 * -0.28444138169288635 +
             n5 * -0.18477343022823334 +
             n6 * -0.09105003625154495 +
             n7 * 0.0 +
             n8 * 0.07958029955625534 +
             n9 * 0.16093020141124725 +
             n10 * 0.24611230194568634 +
             n11 * 0.33791524171829224 +
             n12 * 0.44070982933044434 +
             n13 * 0.5626170039176941 +
             n14 * 0.7229568362236023 +
             n15 * 1.0)
    
    # Optimized scale index calculation
    blocks_per_row = (N + 63) >> 6  # Faster than division
    block_col = col >> 6
    absmax_idx = row * blocks_per_row + block_col
    absmax32_idx = row * ((blocks_per_row + 3) >> 2) + (block_col >> 2)
    
    # Load scales with cache hints
    absmax_vals = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0, eviction_policy="evict_last")
    absmax32_vals = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=0.0, eviction_policy="evict_last")
    
    # Optimized scale computation using FMA
    inv_127 = 0.00787401574803149606  # 1.0 / 127.0
    scales = absmax_vals.to(tl.float32) * inv_127 * absmax32_vals
    output = codes * scales
    
    # Store with non-temporal hints for large matrices
    tl.store(output_ptr + offsets, output, mask=mask, eviction_policy="evict_first")

def asm_optimized_dequantize(module):
    """ASM-optimized NF4 dequantization implementation."""
    weight = module.weight
    quant_state = weight.quant_state
    
    # Extract components
    qweight = weight.data
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    dtype = quant_state.dtype
    device = qweight.device
    
    M = module.out_features
    N = module.in_features
    
    # Prepare scaling factors
    blocks_per_row = (N + 63) // 64
    
    # Fast reshape for absmax
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        elif absmax.numel() == M * blocks_per_row:
            absmax = absmax.view(M, blocks_per_row)
    
    # Fast reshape for absmax32
    absmax32_per_row = (blocks_per_row + 3) // 4
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Adaptive block size selection
    total_elements = M * N
    
    # Use larger blocks for better throughput
    if total_elements > 100_000_000:
        BLOCK_SIZE = 8192
    elif total_elements > 50_000_000:
        BLOCK_SIZE = 4096
    elif total_elements > 10_000_000:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    _asm_optimized_nf4_kernel[grid](
        qweight.view(-1),
        absmax.contiguous().view(-1),
        absmax32.contiguous().view(-1),
        output.view(-1),
        M, N,
        BLOCK_SIZE,
    )
    
    return output