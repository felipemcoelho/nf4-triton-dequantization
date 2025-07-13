import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _hyper_optimized_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    """Hyper-optimized kernel that processes multiple rows and columns per thread."""
    # 2D grid with better work distribution
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    if pid_m >= num_pid_m:
        return
    
    # Row and column ranges
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Masks
    rm_mask = rm < m
    rn_mask = rn < n
    
    # Pre-computed NF4 values as hex for faster loading
    nf4_hex = tl.inline_const_array([
        0xBF800000, 0xBF322E8C, 0xBF066E46, 0xBECA3244,
        0xBE91CB77, 0xBE3D6EBC, 0xBDBA6B62, 0x00000000,
        0x3DA30966, 0x3E24DE11, 0x3E7C1953, 0x3EAD1268,
        0x3EE1E4F8, 0x3F100806, 0x3F38D666, 0x3F800000
    ])
    
    # Process each block
    for i in range(BLOCK_M):
        row = rm[i]
        if not rm_mask[i]:
            continue
            
        for j in range(BLOCK_N):
            col = rn[j]
            if not rn_mask[j]:
                continue
                
            # Calculate block index for this column
            block_idx = col >> 6
            col_in_block = col & 63
            
            # Skip if we're processing elements outside block boundary
            if block_idx >= blocks_per_row:
                continue
            
            # Load absmax values
            absmax_idx = row * blocks_per_row + block_idx
            absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
            
            absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx)
            scale = absmax * 0.00787401574803149606 * absmax32
            
            # Calculate index
            idx = row * n + col
            packed_idx = idx >> 1
            
            # Load packed byte
            packed = tl.load(qweight_ptr + packed_idx)
            
            # Extract nibble
            is_odd = idx & 1
            nibble = (packed >> (is_odd << 2)) & 0xF
            
            # Get NF4 value using hex lookup
            hex_val = tl.gather(nf4_hex, nibble)
            nf4_val = tl.bit_cast(hex_val, tl.float32)
            
            # Scale and store
            output = (nf4_val * scale).to(dtype)
            tl.store(output_ptr + idx, output)


@triton.jit
def _vector_optimized_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Vectorized kernel that processes full 64-element blocks."""
    pid = tl.program_id(0)
    
    # Each thread processes one full 64-element block
    total_blocks = m * blocks_per_row
    if pid >= total_blocks:
        return
    
    row = pid // blocks_per_row
    block_in_row = pid % blocks_per_row
    col_start = block_in_row << 6
    
    if col_start >= n:
        return
    
    # Load absmax values once
    absmax_idx = pid
    absmax32_idx = row * absmax32_per_row + (block_in_row >> 2)
    
    absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx)
    scale = absmax * 0.00787401574803149606 * absmax32
    
    # NF4 lookup table
    nf4_vals = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Process all 64 elements with maximum vectorization
    base_idx = row * n + col_start
    
    # Unroll processing for better performance
    for i in range(0, 64, 16):
        if col_start + i >= n:
            break
            
        # Process 16 elements at once
        cols = col_start + i + tl.arange(0, 16)
        mask = cols < n
        
        # Calculate indices
        idx = base_idx + i + tl.arange(0, 16)
        packed_idx = idx >> 1
        
        # Vectorized load
        packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
        
        # Extract nibbles
        is_odd = idx & 1
        nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
        
        # Vectorized lookup
        nf4 = tl.gather(nf4_vals, nibbles)
        
        # Scale and store
        output = (nf4 * scale).to(dtype)
        tl.store(output_ptr + idx, output, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 64}, num_warps=1),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 128}, num_warps=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_N': 64}, num_warps=2),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 64}, num_warps=4),
    ],
    key=['m', 'n'],
)
@triton.jit
def _autotuned_hyper_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    """Autotuned wrapper for finding optimal configuration."""
    _hyper_optimized_nf4_kernel(
        qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
        m, n, blocks_per_row, absmax32_per_row,
        BLOCK_M, BLOCK_N, dtype
    )


def hyper_optimized_dequantize_nf4(module):
    """Hyper-optimized NF4 dequantization for maximum performance."""
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
    
    # Handle absmax shapes
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
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Choose kernel based on problem size
    if M * N < 512 * 512:
        # Small matrices - use simple vectorized kernel
        total_blocks = M * blocks_per_row
        grid = (total_blocks,)
        
        _vector_optimized_kernel[grid](
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
    else:
        # Large matrices - use autotuned kernel
        grid = lambda META: (tl.cdiv(M, META['BLOCK_M']) * tl.cdiv(N, META['BLOCK_N']),)
        
        _autotuned_hyper_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            dtype=dtype,
        )
    
    return output