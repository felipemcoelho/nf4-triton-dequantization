import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _challenge_optimized_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    stride_qw,
    stride_am, stride_am_k,
    stride_am32, stride_am32_k,
    stride_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    dtype: tl.constexpr,
):
    """Highly optimized NF4 dequantization kernel for 1.15x+ speedup."""
    
    # Program ID for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Early exit bounds check
    if pid_m >= tl.cdiv(m, BLOCK_M) or pid_n >= tl.cdiv(n, GROUP_N):
        return
    
    # Calculate row offset
    row_start = pid_m * BLOCK_M
    row_end = tl.minimum(row_start + BLOCK_M, m)
    
    # Calculate column offset (process GROUP_N columns)
    col_start = pid_n * GROUP_N
    col_end = tl.minimum(col_start + GROUP_N, n)
    
    # Pre-compute NF4 values in registers (avoid gather operations)
    nf4_neg = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0
    ])
    nf4_pos = tl.inline_const_array([
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Magic constant for scaling
    scale_factor = 0.00787401574803149606  # 1/127
    
    # Process blocks within the group
    for block_col in range(col_start, col_end, 64):
        if block_col >= n:
            break
            
        # Calculate indices for absmax values
        block_idx = block_col >> 6
        absmax32_idx = block_idx >> 2
        
        # Process each row in the block
        for row in range(row_start, row_end):
            if row >= m:
                break
                
            # Load absmax values once per 64-element block
            absmax_offset = row * stride_am + block_idx * stride_am_k
            absmax = tl.load(absmax_ptr + absmax_offset).to(tl.float32)
            
            absmax32_offset = row * stride_am32 + absmax32_idx * stride_am32_k
            absmax32 = tl.load(absmax32_ptr + absmax32_offset)
            
            # Compute final scale
            scale = absmax * scale_factor * absmax32
            
            # Process BLOCK_N elements at a time within the 64-element block
            for inner_col in range(0, 64, BLOCK_N):
                col = block_col + inner_col + tl.arange(0, BLOCK_N)
                
                # Bounds check
                mask = col < n
                
                # Calculate linear indices
                idx = row * n + col
                
                # Load packed weights (2 nibbles per byte)
                packed_idx = idx >> 1
                packed_offset = packed_idx
                packed = tl.load(qweight_ptr + packed_offset, mask=mask, other=0)
                
                # Extract nibbles efficiently
                is_odd = (idx & 1).to(tl.int1)
                nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
                
                # Ultra-fast lookup using conditional selection
                # Split at 8 to use two separate lookup arrays
                is_neg = nibbles < 8
                idx_neg = nibbles
                idx_pos = nibbles - 8
                
                # Perform lookups
                val_neg = tl.gather(nf4_neg, idx_neg)
                val_pos = tl.gather(nf4_pos, idx_pos)
                
                # Select final value
                nf4_val = tl.where(is_neg, val_neg, val_pos)
                
                # Apply scale and convert to output dtype
                output = (nf4_val * scale).to(dtype)
                
                # Store result
                output_offset = row * stride_out + col
                tl.store(output_ptr + output_offset, output, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 32, 'GROUP_N': 256}, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 64, 'GROUP_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 2, 'BLOCK_N': 32, 'GROUP_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 32, 'GROUP_N': 512}, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 64, 'GROUP_N': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 32, 'GROUP_N': 1024}, num_warps=4),
    ],
    key=['m', 'n'],
)
@triton.jit
def _challenge_autotuned_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    stride_qw,
    stride_am, stride_am_k,
    stride_am32, stride_am32_k,
    stride_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    dtype: tl.constexpr,
):
    """Autotuned wrapper for the optimized kernel."""
    _challenge_optimized_nf4_kernel(
        qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
        m, n,
        stride_qw,
        stride_am, stride_am_k,
        stride_am32, stride_am32_k,
        stride_out,
        BLOCK_M, BLOCK_N, GROUP_N,
        dtype
    )

def challenge_optimized_dequantize_nf4(module):
    """Challenge-optimized NF4 dequantization targeting 1.15x+ speedup."""
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
    
    # Handle absmax tensor shapes
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
    
    # Ensure contiguous memory layout for optimal performance
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Pre-allocate output tensor
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Calculate strides
    stride_qw = qweight.stride(0) if qweight.dim() > 1 else qweight.shape[0]
    stride_am = absmax.stride(0)
    stride_am_k = absmax.stride(1)
    stride_am32 = absmax32.stride(0)
    stride_am32_k = absmax32.stride(1)
    stride_out = output.stride(0)
    
    # Choose kernel based on matrix size
    if M * N < 1024 * 1024:  # Small matrices
        # Use fixed configuration for small matrices
        BLOCK_M = 1
        BLOCK_N = 32
        GROUP_N = 256
        
        grid = (tl.cdiv(M, BLOCK_M), tl.cdiv(N, GROUP_N))
        
        _challenge_optimized_nf4_kernel[grid](
            qweight.data_ptr(),
            absmax.data_ptr(),
            absmax32.data_ptr(),
            output.data_ptr(),
            M, N,
            stride_qw,
            stride_am, stride_am_k,
            stride_am32, stride_am32_k,
            stride_out,
            BLOCK_M, BLOCK_N, GROUP_N,
            dtype,
        )
    else:
        # Use autotuned kernel for larger matrices
        grid = lambda META: (tl.cdiv(M, META['BLOCK_M']), tl.cdiv(N, META['GROUP_N']))
        
        _challenge_autotuned_kernel[grid](
            qweight.data_ptr(),
            absmax.data_ptr(),
            absmax32.data_ptr(),
            output.data_ptr(),
            M, N,
            stride_qw,
            stride_am, stride_am_k,
            stride_am32, stride_am32_k,
            stride_out,
            dtype=dtype,
        )
    
    return output

# Additional optimized kernel with cache eviction hints
@triton.jit
def _cache_aware_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Cache-aware kernel with eviction hints for maximum performance."""
    pid = tl.program_id(0)
    
    # Each thread processes one 64-element block
    total_blocks = m * blocks_per_row
    if pid >= total_blocks:
        return
    
    # Calculate row and block within row
    row = pid // blocks_per_row
    block_in_row = pid % blocks_per_row
    
    # Column range for this block
    col_start = block_in_row * 64
    if col_start >= n:
        return
    
    # Load absmax values
    absmax_idx = pid
    absmax32_idx = row * absmax32_per_row + (block_in_row >> 2)
    
    absmax = tl.load(absmax_ptr + absmax_idx, eviction_policy="evict_first").to(tl.float32)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx, eviction_policy="evict_first")
    scale = absmax * 0.00787401574803149606 * absmax32
    
    # NF4 lookup values in constants
    nf4_vals = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Process 64 elements in vectorized chunks
    for i in range(0, 64, 32):
        col = col_start + i + tl.arange(0, 32)
        mask = col < n
        
        idx = row * n + col
        packed_idx = idx >> 1
        
        # Load with cache eviction hint
        packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0, eviction_policy="evict_first")
        
        # Extract nibbles
        is_odd = (idx & 1).to(tl.int1)
        nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
        
        # Lookup and scale
        nf4 = tl.gather(nf4_vals, nibbles)
        output = (nf4 * scale).to(dtype)
        
        # Store with streaming hint
        tl.store(output_ptr + idx, output, mask=mask, eviction_policy="evict_first")

def cache_aware_dequantize_nf4(module):
    """Cache-aware implementation with eviction hints."""
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
    
    # Prepare tensors
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        elif absmax.numel() == M * blocks_per_row:
            absmax = absmax.view(M, blocks_per_row)
    
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch kernel
    grid = (M * blocks_per_row,)
    
    _cache_aware_nf4_kernel[grid](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        dtype,
    )
    
    return output