import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize
from .asm_optimized import asm_optimized_dequantize

@triton.jit
def _ultra_fast_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """Ultra-optimized NF4 kernel with inline lookup and vectorization."""
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for bounds checking
    mask = offsets < (M * N)
    
    # Calculate row/col from linear index
    row = offsets // N
    col = offsets % N
    
    # Vectorized byte loading with cache eviction hint
    byte_offsets = offsets >> 1
    bytes_packed = tl.load(qweight_ptr + byte_offsets, mask=mask, other=0, eviction_policy="evict_first")
    
    # Extract nibbles using bitwise ops
    is_odd = (offsets & 1) == 1
    nibbles = tl.where(is_odd, (bytes_packed >> 4), bytes_packed) & 0x0F
    
    # Inline NF4 lookup table - hardcoded for maximum performance
    codes = tl.where(nibbles == 0, -1.0,
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
            tl.where(nibbles == 14, 0.7229568362236023, 1.0)))))))))))))))
    
    # Calculate scale indices
    blocks_per_row = (N + 63) // 64
    block_col = col // 64
    absmax_idx = row * blocks_per_row + block_col
    absmax32_idx = row * ((blocks_per_row + 3) // 4) + (block_col // 4)
    
    # Load scales with coalescing and cache hints
    absmax_vals = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0, eviction_policy="evict_last")
    absmax32_vals = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=0.0, eviction_policy="evict_last")
    
    # Compute final values - optimized multiplication
    scales = (absmax_vals.to(tl.float32) * (1.0 / 127.0)) * absmax32_vals
    output = codes * scales
    
    # Store with coalescing
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def _memory_efficient_nf4_kernel(
    qweight_ptr,
    absmax_ptr, 
    absmax32_ptr,
    output_ptr,
    total_elements,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Memory-efficient version with prefetching."""
    pid = tl.program_id(0)
    
    # Base offset for this block
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute indices
    rows = offsets // cols
    cols_idx = offsets % cols
    
    # Prefetch packed weights
    byte_idx = offsets >> 1
    packed = tl.load(qweight_ptr + byte_idx, mask=mask, other=0)
    
    # Extract nibbles efficiently
    is_high = (offsets & 1) == 1
    nibbles = tl.where(is_high, packed >> 4, packed) & 0x0F
    
    # Direct computation of NF4 values
    # Split into groups for better instruction scheduling
    is_neg = nibbles < 8
    abs_nibbles = tl.where(is_neg, 7 - nibbles, nibbles - 8)
    
    # Compute base values
    base_vals = tl.where(abs_nibbles == 0, 1.0,
                tl.where(abs_nibbles == 1, 0.7229568362236023,
                tl.where(abs_nibbles == 2, 0.5626170039176941,
                tl.where(abs_nibbles == 3, 0.44070982933044434,
                tl.where(abs_nibbles == 4, 0.33791524171829224,
                tl.where(abs_nibbles == 5, 0.24611230194568634,
                tl.where(abs_nibbles == 6, 0.16093020141124725,
                0.07958029955625534)))))))
    
    # Handle special cases
    base_vals = tl.where(nibbles == 7, 0.0, base_vals)
    base_vals = tl.where(nibbles == 6, -0.09105003625154495, base_vals)
    base_vals = tl.where(nibbles == 5, -0.18477343022823334, base_vals)
    base_vals = tl.where(nibbles == 4, -0.28444138169288635, base_vals)
    base_vals = tl.where(nibbles == 3, -0.39491748809814453, base_vals)
    base_vals = tl.where(nibbles == 2, -0.5250730514526367, base_vals)
    base_vals = tl.where(nibbles == 1, -0.6961928009986877, base_vals)
    base_vals = tl.where(nibbles == 0, -1.0, base_vals)
    
    # Compute block indices
    blocks_per_row = (cols + 63) // 64
    block_idx = cols_idx // 64
    absmax_idx = rows * blocks_per_row + block_idx
    absmax32_idx = rows * ((blocks_per_row + 3) // 4) + (block_idx >> 2)
    
    # Load and apply scales
    absmax = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=0.0)
    
    # Final computation
    scale = (absmax.to(tl.float32) * (1.0 / 127.0)) * absmax32
    output = base_vals * scale
    
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_dequantize_nf4(module):
    """Ultra-optimized NF4 dequantization using Triton."""
    # Try ASM-optimized version first for maximum performance
    try:
        return asm_optimized_dequantize(module)
    except:
        pass
    
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
    
    # Reshape absmax efficiently
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        elif absmax.numel() == M * blocks_per_row:
            absmax = absmax.view(M, blocks_per_row)
    
    if absmax.shape != (M, blocks_per_row):
        return fast_dequantize(weight, quant_state)
    
    # Reshape absmax32
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
    
    # Choose optimal block size based on matrix size
    total_elements = M * N
    if total_elements > 50_000_000:  # Very large matrices
        BLOCK_SIZE = 4096
    elif total_elements > 10_000_000:  # Large matrices  
        BLOCK_SIZE = 2048
    elif total_elements > 1_000_000:   # Medium matrices
        BLOCK_SIZE = 1024
    else:                              # Small matrices
        BLOCK_SIZE = 512
    
    # Ensure block size doesn't exceed total elements
    BLOCK_SIZE = min(BLOCK_SIZE, total_elements)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Use the ultra-fast kernel for best performance
    _ultra_fast_nf4_kernel[grid](
        qweight.view(-1),
        absmax.contiguous().view(-1),
        absmax32.contiguous().view(-1),
        output.view(-1),
        M, N,
        BLOCK_SIZE,
    )
    
    return output

def optimized_triton_dequantize_nf4(module):
    """Alias for compatibility."""
    return triton_dequantize_nf4(module)

def benchmark_fast_dequantize(module):
    """Benchmark entry point using the fastest implementation."""
    return triton_dequantize_nf4(module)

def reset_triton_dequantize_state():
    """Reset any cached state."""
    # Clear Triton's compilation cache for fresh benchmarking
    if hasattr(triton, 'runtime'):
        if hasattr(triton.runtime, 'cache'):
            triton.runtime.cache.clear()
    pass