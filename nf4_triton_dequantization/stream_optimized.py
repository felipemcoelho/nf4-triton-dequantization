import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _stream_optimized_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Highly optimized kernel for stream execution."""
    
    # Grid-stride loop for better GPU utilization
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    total_blocks = M * blocks_per_row
    
    # Each program processes multiple blocks with grid-stride
    for block_id in range(pid, total_blocks, num_programs):
        row = block_id // blocks_per_row
        block_idx = block_id % blocks_per_row
        col_base = block_idx * 64
        
        if col_base >= N:
            continue
        
        # Load scales
        absmax = tl.load(absmax_ptr + block_id).to(tl.float32)
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax * 0.00787401574803149606 * absmax32
        
        base_offset = row * N + col_base
        
        # Process 64 elements with aggressive unrolling
        # Unroll into 8 iterations of 8 elements for maximum ILP
        for i in tl.static_range(8):
            offset = i * 8
            idx = base_offset + offset + tl.arange(0, 8)
            cols = col_base + offset + tl.arange(0, 8)
            mask = cols < N
            
            # Vectorized load
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            
            # Optimized nibble extraction
            is_odd = idx & 1
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Direct NF4 mapping with minimal branches
            # Split into two groups for better performance
            is_special = (nibbles == 0) | (nibbles == 7) | (nibbles == 15)
            
            # Regular values use linear approximation
            regular_vals = (nibbles.to(tl.float32) - 7.5) * 0.2666666667
            
            # Special values
            special_vals = tl.where(nibbles == 0, -1.0,
                          tl.where(nibbles == 7, 0.0, 1.0))
            
            # Combine
            base_vals = tl.where(is_special, special_vals, regular_vals)
            
            # Apply corrections for other specific values
            nf4_vals = tl.where(nibbles == 1, -0.6961928009986877,
                       tl.where(nibbles == 2, -0.5250730514526367,
                       tl.where(nibbles == 3, -0.39491748809814453,
                       tl.where(nibbles == 4, -0.28444138169288635,
                       tl.where(nibbles == 5, -0.18477343022823334,
                       tl.where(nibbles == 6, -0.09105003625154495,
                       tl.where(nibbles == 8, 0.07958029955625534,
                       tl.where(nibbles == 9, 0.16093020141124725,
                       tl.where(nibbles == 10, 0.24611230194568634,
                       tl.where(nibbles == 11, 0.33791524171829224,
                       tl.where(nibbles == 12, 0.44070982933044434,
                       tl.where(nibbles == 13, 0.5626170039176941,
                       tl.where(nibbles == 14, 0.7229568362236023,
                       base_vals)))))))))))))
            
            # Apply scale and store
            output = (nf4_vals * scale).to(dtype)
            tl.store(output_ptr + idx, output, mask=mask)

def stream_optimized_dequantize_nf4(module):
    """Stream-optimized NF4 dequantization for parallel execution."""
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
    
    # Optimal grid size for stream execution
    # Use fewer blocks for better cache locality
    total_blocks = M * blocks_per_row
    
    # Choose grid size based on GPU architecture
    # For T4: 40 SMs, so use multiples of 40
    grid_size = min(total_blocks, 320)  # 8 blocks per SM
    
    _stream_optimized_kernel[(grid_size,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        dtype,
        num_warps=2,
        num_stages=2,
    )
    
    return output

# Export
triton_dequantize_nf4 = stream_optimized_dequantize_nf4