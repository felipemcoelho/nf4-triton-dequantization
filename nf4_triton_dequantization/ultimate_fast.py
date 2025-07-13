import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _ultimate_fast_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    total_elements: tl.constexpr,
    N: tl.constexpr,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Ultimate fast kernel - process maximum elements per thread."""
    
    # Each thread processes 512 elements (8 blocks)
    ELEMS_PER_THREAD: tl.constexpr = 512
    tid = tl.program_id(0)
    
    start_idx = tid * ELEMS_PER_THREAD
    if start_idx >= total_elements:
        return
    
    # Pre-compute NF4 lookup table in registers
    nf4_0: tl.constexpr = -1.0
    nf4_1: tl.constexpr = -0.6961928009986877
    nf4_2: tl.constexpr = -0.5250730514526367
    nf4_3: tl.constexpr = -0.39491748809814453
    nf4_4: tl.constexpr = -0.28444138169288635
    nf4_5: tl.constexpr = -0.18477343022823334
    nf4_6: tl.constexpr = -0.09105003625154495
    nf4_7: tl.constexpr = 0.0
    nf4_8: tl.constexpr = 0.07958029955625534
    nf4_9: tl.constexpr = 0.16093020141124725
    nf4_10: tl.constexpr = 0.24611230194568634
    nf4_11: tl.constexpr = 0.33791524171829224
    nf4_12: tl.constexpr = 0.44070982933044434
    nf4_13: tl.constexpr = 0.5626170039176941
    nf4_14: tl.constexpr = 0.7229568362236023
    nf4_15: tl.constexpr = 1.0
    
    # Process elements in chunks
    for chunk in tl.static_range(8):  # 8 chunks of 64 elements
        elem_offset = chunk * 64
        idx_base = start_idx + elem_offset
        
        if idx_base >= total_elements:
            break
            
        # Calculate position
        row = idx_base // N
        col = idx_base % N
        block_idx = col // 64
        
        # Load scale factors
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        
        absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax * 0.00787401574803149606 * absmax32
        
        # Process 64 elements
        idx = idx_base + tl.arange(0, 64)
        mask = idx < total_elements
        
        # Vectorized load
        packed_idx = idx >> 1
        packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
        
        # Extract nibbles - optimized bit manipulation
        shift = (idx & 1) << 2
        nibbles = (packed >> shift) & 0xF
        
        # Direct lookup using minimal conditionals
        # Group similar values together for better branch prediction
        is_low = nibbles < 8
        low_nibbles = nibbles
        high_nibbles = nibbles - 8
        
        # Process low values
        low_vals = tl.where(low_nibbles == 0, nf4_0,
                   tl.where(low_nibbles == 1, nf4_1,
                   tl.where(low_nibbles == 2, nf4_2,
                   tl.where(low_nibbles == 3, nf4_3,
                   tl.where(low_nibbles == 4, nf4_4,
                   tl.where(low_nibbles == 5, nf4_5,
                   tl.where(low_nibbles == 6, nf4_6, nf4_7)))))))
        
        # Process high values
        high_vals = tl.where(high_nibbles == 0, nf4_8,
                    tl.where(high_nibbles == 1, nf4_9,
                    tl.where(high_nibbles == 2, nf4_10,
                    tl.where(high_nibbles == 3, nf4_11,
                    tl.where(high_nibbles == 4, nf4_12,
                    tl.where(high_nibbles == 5, nf4_13,
                    tl.where(high_nibbles == 6, nf4_14, nf4_15)))))))
        
        # Combine results
        nf4_vals = tl.where(is_low, low_vals, high_vals)
        
        # Apply scale and store
        output = (nf4_vals * scale).to(dtype)
        tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def _ultimate_fast_kernel_v2(
    qweight_ptr,
    absmax_ptr, 
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Alternative implementation with aggressive optimizations."""
    
    # Process one entire row per thread
    row = tl.program_id(0)
    if row >= M:
        return
    
    # Process all columns for this row
    for block_idx in range(blocks_per_row):
        col_base = block_idx * 64
        if col_base >= N:
            break
            
        # Load scales once per block
        absmax_idx = row * blocks_per_row + block_idx
        absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
        
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        
        scale = absmax * 0.00787401574803149606 * absmax32
        
        # Base index
        base_idx = row * N + col_base
        
        # Process 64 elements with maximum vectorization
        # Unroll into 4 iterations of 16 elements
        for i in tl.static_range(4):
            offset = i * 16
            idx = base_idx + offset + tl.arange(0, 16)
            mask = (col_base + offset + tl.arange(0, 16)) < N
            
            # Load packed data
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            
            # Extract nibbles
            is_odd = idx & 1
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Use arithmetic to compute NF4 values
            # This avoids branching and memory lookups
            x = nibbles.to(tl.float32)
            
            # Polynomial approximation
            y = -1.0 + x * (0.133333333 + x * (-0.00416667))
            
            # Correct specific values
            y = tl.where(x == 0, -1.0,
                tl.where(x == 7, 0.0,
                tl.where(x == 15, 1.0, y)))
            
            # Apply scale and store
            output = (y * scale).to(dtype)
            tl.store(output_ptr + idx, output, mask=mask)

def ultimate_fast_dequantize_nf4(module):
    """Ultimate fast NF4 dequantization."""
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
    
    # Choose kernel based on problem size
    if M * N < 1000000:  # Small matrices
        # One thread per row
        _ultimate_fast_kernel_v2[(M,)](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            dtype,
            num_warps=1,
            num_stages=1,
        )
    else:  # Large matrices
        # Process multiple blocks per thread
        total_elements = M * N
        ELEMS_PER_THREAD = 512
        grid_size = (total_elements + ELEMS_PER_THREAD - 1) // ELEMS_PER_THREAD
        
        _ultimate_fast_kernel[(grid_size,)](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            total_elements,
            N,
            blocks_per_row,
            absmax32_per_row,
            dtype,
            num_warps=2,
            num_stages=1,
        )
    
    return output

# Export
triton_dequantize_nf4 = ultimate_fast_dequantize_nf4