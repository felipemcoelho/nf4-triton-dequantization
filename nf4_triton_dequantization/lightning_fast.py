import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _lightning_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Lightning-fast NF4 kernel optimized for T4 GPUs."""
    
    # 2D grid for better parallelization
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Each thread block processes BLOCK_SIZE_M x BLOCK_SIZE_N elements
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    # Bounds checking
    if row_start >= M or col_start >= N:
        return
    
    # NF4 LUT - using const array for better performance
    nf4_low = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0
    ])
    nf4_high = tl.inline_const_array([
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Process multiple rows in parallel
    for m_offset in range(BLOCK_SIZE_M):
        row = row_start + m_offset
        if row >= M:
            break
            
        # Process blocks of 64 elements (NF4 block size)
        for block_offset in range(0, BLOCK_SIZE_N, 64):
            col_base = col_start + block_offset
            if col_base >= N:
                break
                
            block_idx = col_base // 64
            
            # Load scales with minimal memory transactions
            absmax_idx = row * blocks_per_row + block_idx
            absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            
            absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx)
            
            # Precompute scale
            scale = absmax * 0.00787401574803149606 * absmax32
            
            base_idx = row * N + col_base
            
            # Process 64 elements in 4 chunks of 16 for optimal vectorization
            for chunk in tl.static_range(4):
                chunk_offset = chunk * 16
                cols = col_base + chunk_offset + tl.arange(0, 16)
                mask = cols < N
                
                idx = base_idx + chunk_offset + tl.arange(0, 16)
                
                # Vectorized load of packed data
                packed_idx = idx >> 1
                packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
                
                # Extract nibbles using optimized bit operations
                is_odd = idx & 1
                nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
                
                # Split-lookup approach for better performance
                is_high = nibbles >= 8
                low_idx = nibbles & 7
                high_idx = (nibbles - 8) & 7
                
                # Parallel lookups
                low_vals = tl.load(nf4_low + low_idx, mask=mask, other=0.0)
                high_vals = tl.load(nf4_high + high_idx, mask=mask, other=0.0)
                
                # Select and scale
                nf4_vals = tl.where(is_high, high_vals, low_vals)
                output = (nf4_vals * scale).to(dtype)
                
                # Store with cache eviction hint
                tl.store(output_ptr + idx, output, mask=mask, eviction_policy="evict_first")

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def _lightning_nf4_kernel_autotuned(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Autotuned version of the lightning kernel."""
    _lightning_nf4_kernel(
        qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
        M, N, blocks_per_row, absmax32_per_row, dtype,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )

def lightning_fast_dequantize_nf4(module):
    """Lightning-fast NF4 dequantization optimized for 1.15x+ speedup."""
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
    
    # Prepare absmax tensors
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
    
    # Ensure contiguous memory layout
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Use autotuned kernel for optimal performance
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )
    
    _lightning_nf4_kernel_autotuned[grid](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        dtype
    )
    
    return output

# Export
triton_dequantize_nf4 = lightning_fast_dequantize_nf4