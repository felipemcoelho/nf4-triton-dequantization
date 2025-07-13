import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _ultra_fast_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Ultra-fast kernel with maximum vectorization and minimal overhead."""
    
    # Each program handles one 64-element block
    pid = tl.program_id(0)
    
    total_blocks = M * blocks_per_row
    if pid >= total_blocks:
        return
    
    # Decode position
    row = pid // blocks_per_row
    block_idx = pid % blocks_per_row
    col_base = block_idx * 64
    
    if col_base >= N:
        return
    
    # Load scales once
    absmax = tl.load(absmax_ptr + pid).to(tl.float32)
    absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx)
    
    # Fuse scale computation
    scale = absmax * (0.00787401574803149606 * absmax32)
    
    base_offset = row * N + col_base
    
    # Process all 64 elements in one shot with maximum vectorization
    # Use larger vector size for better memory throughput
    idx = base_offset + tl.arange(0, 64)
    
    # Single vectorized load of all packed data
    packed_idx = idx >> 1
    packed = tl.load(qweight_ptr + packed_idx)
    
    # Optimized nibble extraction using bit manipulation
    # Process even and odd indices separately for better vectorization
    even_idx = tl.arange(0, 32) * 2
    odd_idx = even_idx + 1
    
    even_nibbles = packed[tl.arange(0, 32)] & 0xF
    odd_nibbles = (packed[tl.arange(0, 32)] >> 4) & 0xF
    
    # Interleave results
    nibbles = tl.zeros((64,), dtype=tl.int32)
    nibbles[even_idx] = even_nibbles
    nibbles[odd_idx] = odd_nibbles
    
    # Use direct lookup for NF4 values
    nf4_lut = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Vectorized lookup
    nf4_vals = tl.load(nf4_lut + nibbles)
    
    # Apply scale and store in one operation
    output = (nf4_vals * scale).to(dtype)
    
    # Check bounds and store
    mask = (col_base + tl.arange(0, 64)) < N
    tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def _ultra_fast_kernel_v2(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Alternative ultra-fast implementation."""
    
    pid = tl.program_id(0)
    
    # Calculate block position
    total_blocks = M * blocks_per_row
    if pid >= total_blocks:
        return
        
    row = pid // blocks_per_row
    block_idx = pid % blocks_per_row
    col_base = block_idx << 6  # * 64
    
    if col_base >= N:
        return
    
    # Optimized scale loading and computation
    absmax_val = tl.load(absmax_ptr + pid).to(tl.float32)
    absmax32_val = tl.load(absmax32_ptr + row * absmax32_per_row + (block_idx >> 2))
    scale = absmax_val * 0.00787401574803149606 * absmax32_val
    
    # Base indices
    base_idx = row * N + col_base
    
    # Pre-computed NF4 values
    nf4_vals_low = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0
    ])
    nf4_vals_high = tl.inline_const_array([
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Process in two halves for better cache usage
    for half in tl.static_range(2):
        offset = half * 32
        indices = base_idx + offset + tl.arange(0, 32)
        cols = col_base + offset + tl.arange(0, 32)
        
        # Bounds check
        mask = cols < N
        
        # Load packed data
        packed_indices = indices >> 1
        packed_data = tl.load(qweight_ptr + packed_indices, mask=mask, other=0)
        
        # Extract nibbles
        is_odd = indices & 1
        nibbles = tl.where(is_odd, (packed_data >> 4) & 0xF, packed_data & 0xF)
        
        # Fast lookup using split tables
        is_high = nibbles >= 8
        low_idx = nibbles
        high_idx = nibbles - 8
        
        low_vals = tl.load(nf4_vals_low + low_idx, mask=mask & ~is_high, other=0.0)
        high_vals = tl.load(nf4_vals_high + high_idx, mask=mask & is_high, other=0.0)
        
        nf4_values = tl.where(is_high, high_vals, low_vals)
        
        # Scale and store
        output = (nf4_values * scale).to(dtype)
        tl.store(output_ptr + indices, output, mask=mask)

def ultra_fast_dequantize_nf4_v2(module):
    """Ultra-fast NF4 dequantization implementation."""
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
    
    # Prepare absmax
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
    
    # Prepare absmax32
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
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch kernel - one thread per block for simplicity
    total_blocks = M * blocks_per_row
    
    _ultra_fast_kernel_v2[(total_blocks,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        dtype,
        num_warps=1,  # Minimal warps for small workload
        num_stages=1,  # No pipelining needed
    )
    
    return output

# Export
triton_dequantize_nf4 = ultra_fast_dequantize_nf4_v2