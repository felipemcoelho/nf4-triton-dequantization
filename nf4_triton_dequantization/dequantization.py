import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _nf4_dequantize_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    n_elements,
    n_cols,
    blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Ultra-optimized NF4 dequantization kernel."""
    pid = tl.program_id(0)
    
    # Process BLOCK_SIZE elements per program
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute row/col indices
    row = offsets // n_cols
    col = offsets % n_cols
    
    # Load packed weights efficiently
    byte_idx = offsets >> 1
    packed = tl.load(qweight_ptr + byte_idx, mask=mask, other=0)
    
    # Extract nibbles with optimized bit manipulation
    is_odd = offsets & 1
    nibbles = (packed >> (is_odd << 2)) & 0x0F
    
    # Inline NF4 lookup table for maximum performance
    codes = (
        tl.where(nibbles == 0, -1.0,
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
    )
    
    # Calculate scale indices efficiently
    block_col = col >> 6
    absmax_idx = row * blocks_per_row + block_col
    absmax32_idx = row * ((blocks_per_row + 3) >> 2) + (block_col >> 2)
    
    # Load scales
    absmax_val = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0)
    absmax32_val = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=0.0)
    
    # Final dequantization with FMA
    scale = absmax_val.to(tl.float32) * (1.0 / 127.0) * absmax32_val
    output = codes * scale
    
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_dequantize_nf4(module):
    """Optimized NF4 dequantization using Triton.
    
    Args:
        module: A Linear4bit module with NF4 quantized weights
        
    Returns:
        Dequantized weight tensor in fp16/bf16 format
    """
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
    n_elements = M * N
    
    # Calculate blocks per row
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
    
    # Allocate output tensor
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Choose optimal block size based on matrix dimensions
    if n_elements < 500_000:
        BLOCK_SIZE = 256
    elif n_elements < 5_000_000:
        BLOCK_SIZE = 1024
    elif n_elements < 50_000_000:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _nf4_dequantize_kernel[grid](
        qweight.view(-1),
        absmax.contiguous().view(-1),
        absmax32.contiguous().view(-1),
        output.view(-1),
        n_elements,
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