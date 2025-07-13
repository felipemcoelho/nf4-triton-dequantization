import torch
import triton
import triton.language as tl
from triton import jit

@triton.jit
def _your_dequantize_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    total_blocks,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Ultra-optimized single Triton kernel for NF4 dequantization."""
    
    # Process multiple blocks per thread for maximum throughput
    pid = tl.program_id(0)
    BLOCKS_PER_THREAD: tl.constexpr = 8
    
    # NF4 scale constant
    NF4_SCALE: tl.constexpr = 0.00787401574803149606
    
    # Inline NF4 lookup values for maximum performance
    nf4_vals = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    start_block = pid * BLOCKS_PER_THREAD
    
    # Process 8 blocks per thread
    for block_offset in tl.static_range(BLOCKS_PER_THREAD):
        block_id = start_block + block_offset
        if block_id >= total_blocks:
            return
        
        # Decode position
        row = block_id // blocks_per_row
        block_idx = block_id % blocks_per_row
        col_base = block_idx * 64
        
        if col_base >= N:
            continue
        
        # Load and compute scale
        absmax = tl.load(absmax_ptr + block_id).to(tl.float32)
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax * NF4_SCALE * absmax32
        
        base_offset = row * N + col_base
        
        # Process 64 elements with maximum vectorization
        # Load all 32 packed bytes at once
        packed_base = (base_offset >> 1) + tl.arange(0, 32)
        packed = tl.load(qweight_ptr + packed_base, eviction_policy="evict_first")
        
        # Extract nibbles for even and odd positions
        nibbles_even = packed & 0xF
        nibbles_odd = (packed >> 4) & 0xF
        
        # Lookup values
        vals_even = tl.load(nf4_vals + nibbles_even)
        vals_odd = tl.load(nf4_vals + nibbles_odd)
        
        # Scale and convert
        out_even = (vals_even * scale).to(dtype)
        out_odd = (vals_odd * scale).to(dtype)
        
        # Store interleaved results
        if col_base + 63 < N:
            # Fast path - no bounds checking
            for i in tl.static_range(32):
                tl.store(output_ptr + base_offset + i * 2, out_even[i], eviction_policy="evict_first")
                tl.store(output_ptr + base_offset + i * 2 + 1, out_odd[i], eviction_policy="evict_first")
        else:
            # Bounds checking for edge case
            for i in tl.static_range(32):
                if col_base + i * 2 < N:
                    tl.store(output_ptr + base_offset + i * 2, out_even[i])
                if col_base + i * 2 + 1 < N:
                    tl.store(output_ptr + base_offset + i * 2 + 1, out_odd[i])

def _your_dequantize_nf4(weight, quant_state):
    """Setup and launch the Triton kernel."""
    qweight = weight
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    dtype = quant_state.dtype
    device = qweight.device
    
    # Get dimensions from weight shape
    packed_shape = qweight.shape
    M = packed_shape[0]
    N = packed_shape[1] * 2  # Each byte contains 2 4-bit values
    
    blocks_per_row = (N + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Prepare absmax tensors
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
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch kernel
    total_blocks = M * blocks_per_row
    BLOCKS_PER_THREAD = 8
    grid_size = (total_blocks + BLOCKS_PER_THREAD - 1) // BLOCKS_PER_THREAD
    
    _your_dequantize_nf4_kernel[(grid_size,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        total_blocks,
        M, N,
        blocks_per_row,
        absmax32_per_row,
        dtype,
        num_warps=4,
        num_stages=4,
    )
    
    return output

def your_dequantize_nf4(weight):
    """Main entry point for the challenge."""
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state)