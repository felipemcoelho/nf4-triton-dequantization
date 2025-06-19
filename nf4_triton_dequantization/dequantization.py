import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _ultra_fast_nf4_dequant(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr, 
    output_ptr,
    total_elements,
    cols,
    blocks_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Ultra-optimized NF4 dequantization kernel."""
    pid = tl.program_id(0)
    
    # Calculate elements to process
    base_idx = pid * BLOCK_SIZE
    offsets = base_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute row/col from linear index
    rows = offsets // cols
    cols_idx = offsets % cols
    
    # Load packed bytes
    byte_offsets = offsets >> 1
    packed_bytes = tl.load(qweight_ptr + byte_offsets, mask=mask, other=0)
    
    # Extract nibbles with bit manipulation
    is_odd = offsets & 1
    nibbles = tl.where(is_odd, packed_bytes >> 4, packed_bytes) & 0x0F
    
    # Ultra-fast NF4 lookup using parallel selection
    # Split into 4 groups of 4 values each for better parallelism
    n0 = nibbles == 0
    n1 = nibbles == 1
    n2 = nibbles == 2
    n3 = nibbles == 3
    n4 = nibbles == 4
    n5 = nibbles == 5
    n6 = nibbles == 6
    n7 = nibbles == 7
    n8 = nibbles == 8
    n9 = nibbles == 9
    n10 = nibbles == 10
    n11 = nibbles == 11
    n12 = nibbles == 12
    n13 = nibbles == 13
    n14 = nibbles == 14
    n15 = nibbles == 15
    
    # Compute NF4 values using multiplication (branchless)
    nf4_vals = (
        n0 * -1.0 +
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
        n15 * 1.0
    )
    
    # Calculate scale indices
    block_idx = cols_idx >> 6
    absmax_idx = rows * blocks_per_row + block_idx
    absmax32_idx = rows * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
    
    # Load scales
    absmax_vals = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0)
    absmax32_vals = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=0.0)
    
    # Final computation with FMA
    scales = absmax_vals.to(tl.float32) * (1.0 / 127.0) * absmax32_vals
    output_vals = nf4_vals * scales
    
    # Store results
    tl.store(output_ptr + offsets, output_vals, mask=mask)

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
    
    # Use optimized block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    _ultra_fast_nf4_dequant[grid](
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