import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

# Precomputed NF4 lookup table as constants
_NF4_TO_FP = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]

@triton.jit
def _optimized_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    total_elements,
    cols,
    blocks_per_row: tl.constexpr,
    block_size: tl.constexpr,
):
    """Ultra-optimized NF4 kernel with coalesced memory access."""
    pid = tl.program_id(0)
    
    # Process block_size elements with better memory coalescing
    base = pid * block_size
    offsets = base + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Fused index computation with better cache utilization
    rows = offsets // cols
    cols_idx = offsets % cols
    
    # Coalesced packed weight loading - optimize for even/odd access pattern
    byte_offsets = offsets >> 1
    packed = tl.load(qweight_ptr + byte_offsets, mask=mask, other=0)
    
    # Vectorized nibble extraction
    is_odd = offsets & 1
    nibbles = tl.where(is_odd, packed >> 4, packed) & 0x0F
    
    # Ultra-optimized branchless NF4 lookup with FMA operations
    # Compute all comparisons in parallel
    eq0 = nibbles == 0
    eq1 = nibbles == 1
    eq2 = nibbles == 2
    eq3 = nibbles == 3
    eq4 = nibbles == 4
    eq5 = nibbles == 5
    eq6 = nibbles == 6
    eq7 = nibbles == 7
    eq8 = nibbles == 8
    eq9 = nibbles == 9
    eq10 = nibbles == 10
    eq11 = nibbles == 11
    eq12 = nibbles == 12
    eq13 = nibbles == 13
    eq14 = nibbles == 14
    eq15 = nibbles == 15
    
    # Single FMA chain for NF4 values
    nf4_vals = (\n        eq0 * -1.0 +\n        eq1 * -0.6961928009986877 +\n        eq2 * -0.5250730514526367 +\n        eq3 * -0.39491748809814453 +\n        eq4 * -0.28444138169288635 +\n        eq5 * -0.18477343022823334 +\n        eq6 * -0.09105003625154495 +\n        eq7 * 0.0 +\n        eq8 * 0.07958029955625534 +\n        eq9 * 0.16093020141124725 +\n        eq10 * 0.24611230194568634 +\n        eq11 * 0.33791524171829224 +\n        eq12 * 0.44070982933044434 +\n        eq13 * 0.5626170039176941 +\n        eq14 * 0.7229568362236023 +\n        eq15 * 1.0\n    )
    
    # Optimized scale index calculation
    block_col = cols_idx >> 6
    absmax_idx = rows * blocks_per_row + block_col
    
    # Coalesced scale loading with better cache utilization
    absmax = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0)
    
    # Calculate absmax32 index and load
    absmax32_blocks_per_row = (blocks_per_row + 3) >> 2
    absmax32_idx = rows * absmax32_blocks_per_row + (block_col >> 2)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=0.0)
    
    # Optimized final computation with reduced operations
    # Precompute constant: 1/127 = 0.00787401574803149606
    scale = absmax.to(tl.float32) * 0.00787401574803149606
    output = nf4_vals * scale * absmax32
    
    # Coalesced store
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['total_elements', 'cols'],
)
@triton.jit
def _autotuned_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    total_elements,
    cols,
    blocks_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned kernel for maximum performance across different sizes."""
    pid = tl.program_id(0)
    
    # Process BLOCK_SIZE elements
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute indices
    rows = offsets // cols
    cols_idx = offsets % cols
    
    # Load packed weights
    byte_idx = offsets >> 1
    packed = tl.load(qweight_ptr + byte_idx, mask=mask, other=0)
    
    # Extract nibbles efficiently
    shift = (offsets & 1) << 2
    nibbles = (packed >> shift) & 0x0F
    
    # Super-optimized branchless lookup
    # Use FMA operations to reduce instruction count
    n = nibbles
    
    # Create masks
    m0 = n == 0
    m1 = n == 1
    m2 = n == 2
    m3 = n == 3
    m4 = n == 4
    m5 = n == 5
    m6 = n == 6
    m7 = n == 7
    m8 = n == 8
    m9 = n == 9
    m10 = n == 10
    m11 = n == 11
    m12 = n == 12
    m13 = n == 13
    m14 = n == 14
    m15 = n == 15
    
    # Use FMA to combine operations
    nf4_vals = tl.zeros_like(nibbles, dtype=tl.float32)
    nf4_vals = tl.where(m0, -1.0, nf4_vals)
    nf4_vals = tl.where(m1, -0.6961928009986877, nf4_vals)
    nf4_vals = tl.where(m2, -0.5250730514526367, nf4_vals)
    nf4_vals = tl.where(m3, -0.39491748809814453, nf4_vals)
    nf4_vals = tl.where(m4, -0.28444138169288635, nf4_vals)
    nf4_vals = tl.where(m5, -0.18477343022823334, nf4_vals)
    nf4_vals = tl.where(m6, -0.09105003625154495, nf4_vals)
    nf4_vals = tl.where(m7, 0.0, nf4_vals)
    nf4_vals = tl.where(m8, 0.07958029955625534, nf4_vals)
    nf4_vals = tl.where(m9, 0.16093020141124725, nf4_vals)
    nf4_vals = tl.where(m10, 0.24611230194568634, nf4_vals)
    nf4_vals = tl.where(m11, 0.33791524171829224, nf4_vals)
    nf4_vals = tl.where(m12, 0.44070982933044434, nf4_vals)
    nf4_vals = tl.where(m13, 0.5626170039176941, nf4_vals)
    nf4_vals = tl.where(m14, 0.7229568362236023, nf4_vals)
    nf4_vals = tl.where(m15, 1.0, nf4_vals)
    
    # Calculate scale indices
    block_col = cols_idx >> 6
    absmax_idx = rows * blocks_per_row + block_col
    absmax32_idx = rows * ((blocks_per_row + 3) >> 2) + (block_col >> 2)
    
    # Load scales
    absmax = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=0.0)
    
    # Optimized final computation
    output = nf4_vals * (absmax.to(tl.float32) * 0.00787401574803149606 * absmax32)
    
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_dequantize_nf4(module):
    """Optimized NF4 dequantization using Triton."""
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
    total_elements = M * N
    
    # Calculate blocks
    blocks_per_row = (N + 63) // 64
    
    # Prepare absmax tensor
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        elif absmax.numel() == M * blocks_per_row:
            absmax = absmax.view(M, blocks_per_row)
    
    if absmax.shape != (M, blocks_per_row):
        return fast_dequantize(weight, quant_state)
    
    # Prepare absmax32 tensor
    absmax32_per_row = (blocks_per_row + 3) // 4
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    if absmax32.shape != (M, absmax32_per_row):
        return fast_dequantize(weight, quant_state)
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Choose kernel strategy based on matrix characteristics
    if total_elements > 10_000_000:
        # Use autotuned kernel for large matrices
        grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
        _autotuned_nf4_kernel[grid](
            qweight.view(-1),
            absmax.contiguous().view(-1),
            absmax32.contiguous().view(-1),
            output.view(-1),
            total_elements,
            N,
            blocks_per_row,
        )
    else:
        # Use fixed optimized kernel for smaller matrices
        # Choose block size based on matrix dimensions
        if total_elements < 100_000:
            BLOCK_SIZE = 256
        elif total_elements < 1_000_000:
            BLOCK_SIZE = 512
        else:
            BLOCK_SIZE = 1024
            
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        _optimized_nf4_kernel[grid](
            qweight.view(-1),
            absmax.contiguous().view(-1),
            absmax32.contiguous().view(-1),
            output.view(-1),
            total_elements,
            N,
            blocks_per_row,
            BLOCK_SIZE,
        )
    
    return output

def optimized_triton_dequantize_nf4(module):
    """Alias for compatibility."""
    return triton_dequantize_nf4(module)

def benchmark_fast_dequantize(module):
    """Benchmark entry point."""
    return triton_dequantize_nf4(module)

def reset_triton_dequantize_state():
    """Reset state - no-op to avoid errors."""
    pass