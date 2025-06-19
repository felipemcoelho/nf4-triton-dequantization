import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

# NF4 lookup table - precomputed values
NF4_QUANT_TABLE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]

@triton.jit
def _fast_nf4_kernel(
    # Inputs
    qweight_ptr,      # Packed weights
    absmax_ptr,       # Block scales (uint8)
    absmax32_ptr,     # Second-level scales (float32)
    # Outputs
    output_ptr,
    # Dimensions
    M, N,             # Matrix dimensions
    # Constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Ultra-optimized NF4 dequantization kernel."""
    # Program ID
    pid = tl.program_id(0)
    
    # Number of blocks in each dimension
    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)
    
    # Swizzle for better memory access pattern
    pid_m = pid // num_block_n
    pid_n = pid % num_block_n
    
    # Block boundaries
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks
    mask_m = rm < M
    mask_n = rn < N
    
    # Precompute NF4 lookup table in registers
    nf4_0 = -1.0
    nf4_1 = -0.6961928009986877
    nf4_2 = -0.5250730514526367
    nf4_3 = -0.39491748809814453
    nf4_4 = -0.28444138169288635
    nf4_5 = -0.18477343022823334
    nf4_6 = -0.09105003625154495
    nf4_7 = 0.0
    nf4_8 = 0.07958029955625534
    nf4_9 = 0.16093020141124725
    nf4_10 = 0.24611230194568634
    nf4_11 = 0.33791524171829224
    nf4_12 = 0.44070982933044434
    nf4_13 = 0.5626170039176941
    nf4_14 = 0.7229568362236023
    nf4_15 = 1.0
    
    # Process blocks
    for m in range(BLOCK_M):
        if rm[m] < M:
            for n in range(BLOCK_N):
                if rn[n] < N:
                    # Calculate element index
                    elem_idx = rm[m] * N + rn[n]
                    
                    # Load packed byte
                    byte_idx = elem_idx >> 1
                    byte_val = tl.load(qweight_ptr + byte_idx)
                    
                    # Extract nibble
                    is_high = (elem_idx & 1) == 1
                    nibble = tl.where(is_high, (byte_val >> 4) & 0x0F, byte_val & 0x0F)
                    
                    # Fast lookup using conditionals (faster than memory load)
                    code_val = tl.where(nibble == 0, nf4_0,
                              tl.where(nibble == 1, nf4_1,
                              tl.where(nibble == 2, nf4_2,
                              tl.where(nibble == 3, nf4_3,
                              tl.where(nibble == 4, nf4_4,
                              tl.where(nibble == 5, nf4_5,
                              tl.where(nibble == 6, nf4_6,
                              tl.where(nibble == 7, nf4_7,
                              tl.where(nibble == 8, nf4_8,
                              tl.where(nibble == 9, nf4_9,
                              tl.where(nibble == 10, nf4_10,
                              tl.where(nibble == 11, nf4_11,
                              tl.where(nibble == 12, nf4_12,
                              tl.where(nibble == 13, nf4_13,
                              tl.where(nibble == 14, nf4_14, nf4_15)))))))))))))))
                    
                    # Calculate block indices
                    block_idx = rn[n] // 64
                    absmax_idx = rm[m] * ((N + 63) // 64) + block_idx
                    absmax32_idx = rm[m] * ((N + 255) // 256) + (block_idx // 4)
                    
                    # Load scales
                    absmax_val = tl.load(absmax_ptr + absmax_idx)
                    absmax32_val = tl.load(absmax32_ptr + absmax32_idx)
                    
                    # Dequantize
                    scale = (absmax_val.to(tl.float32) / 127.0) * absmax32_val
                    dequant_val = code_val * scale
                    
                    # Store result
                    tl.store(output_ptr + elem_idx, dequant_val)

@triton.jit
def _ultra_fast_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """Most optimized NF4 kernel with vectorization and prefetching."""
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for bounds checking
    mask = offsets < (M * N)
    
    # Calculate row/col from linear index
    row = offsets // N
    col = offsets % N
    
    # Vectorized byte loading
    byte_offsets = offsets >> 1
    bytes_packed = tl.load(qweight_ptr + byte_offsets, mask=mask, other=0)
    
    # Extract nibbles using bitwise ops
    is_odd = (offsets & 1) == 1
    nibbles = tl.where(is_odd, (bytes_packed >> 4), bytes_packed) & 0x0F
    
    # Inline NF4 lookup (unrolled for performance)
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
    
    # Load scales with coalescing
    absmax_vals = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0)
    absmax32_vals = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=0.0)
    
    # Compute final values
    scales = (absmax_vals.to(tl.float32) * (1.0 / 127.0)) * absmax32_vals
    output = codes * scales
    
    # Store with coalescing
    tl.store(output_ptr + offsets, output, mask=mask)

def ultra_optimized_dequantize(module):
    """Ultra-optimized NF4 dequantization implementation."""
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
    
    # Reshape absmax
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        else:
            absmax = absmax.view(M, blocks_per_row)
    
    # Reshape absmax32
    absmax32_per_row = (blocks_per_row + 3) // 4
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        else:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch optimized kernel
    total_elements = M * N
    BLOCK_SIZE = 2048  # Larger blocks for better throughput
    
    grid = lambda meta: (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    _ultra_fast_nf4_kernel[grid](
        qweight.view(-1),
        absmax.contiguous().view(-1),
        absmax32.contiguous().view(-1),
        output.view(-1),
        M, N,
        BLOCK_SIZE,
    )
    
    return output

# Export the optimized function
def triton_dequantize_nf4(module):
    """Main entry point for optimized dequantization."""
    return ultra_optimized_dequantize(module)

def optimized_triton_dequantize_nf4(module):
    """Alias for compatibility."""
    return ultra_optimized_dequantize(module)

def benchmark_fast_dequantize(module):
    """Benchmark entry point."""
    return ultra_optimized_dequantize(module)

def reset_triton_dequantize_state():
    """Reset state if needed."""
    pass