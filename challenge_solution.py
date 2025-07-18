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
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Ultra-optimized single Triton kernel for NF4 dequantization."""
    
    # Get block ID
    bid = tl.program_id(0)
    
    # Grid dimensions
    total_blocks = M * blocks_per_row
    if bid >= total_blocks:
        return
    
    # Decode block position
    row = bid // blocks_per_row
    block_in_row = bid % blocks_per_row
    col_start = block_in_row * 64
    
    # Skip if out of bounds
    if col_start >= N:
        return
    
    # Constants
    SCALE: tl.constexpr = 0.00787401574803149606
    
    # Load scales once
    absmax_val = tl.load(absmax_ptr + bid).to(tl.float32)
    absmax32_idx = row * absmax32_per_row + (block_in_row >> 2)
    absmax32_val = tl.load(absmax32_ptr + absmax32_idx)
    scale = absmax_val * SCALE * absmax32_val
    
    # Base addresses  
    qweight_base = row * (N >> 1) + (col_start >> 1)
    output_base = row * N + col_start
    
    # Load all 32 bytes for this block at once
    qweight_data = qweight_base + tl.arange(0, 32)
    packed = tl.load(qweight_ptr + qweight_data, eviction_policy="evict_first")
    
    # Extract low and high nibbles
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    
    # NF4 lookup using optimized binary tree
    low_vals = tl.where(low < 8,
        tl.where(low < 4,
            tl.where(low < 2,
                tl.where(low == 0, -1.0, -0.6961928009986877),
                tl.where(low == 2, -0.5250730514526367, -0.39491748809814453)
            ),
            tl.where(low < 6,
                tl.where(low == 4, -0.28444138169288635, -0.18477343022823334),
                tl.where(low == 6, -0.09105003625154495, 0.0)
            )
        ),
        tl.where(low < 12,
            tl.where(low < 10,
                tl.where(low == 8, 0.07958029955625534, 0.16093020141124725),
                tl.where(low == 10, 0.24611230194568634, 0.33791524171829224)
            ),
            tl.where(low < 14,
                tl.where(low == 12, 0.44070982933044434, 0.5626170039176941),
                tl.where(low == 14, 0.7229568362236023, 1.0)
            )
        )
    )
    
    high_vals = tl.where(high < 8,
        tl.where(high < 4,
            tl.where(high < 2,
                tl.where(high == 0, -1.0, -0.6961928009986877),
                tl.where(high == 2, -0.5250730514526367, -0.39491748809814453)
            ),
            tl.where(high < 6,
                tl.where(high == 4, -0.28444138169288635, -0.18477343022823334),
                tl.where(high == 6, -0.09105003625154495, 0.0)
            )
        ),
        tl.where(high < 12,
            tl.where(high < 10,
                tl.where(high == 8, 0.07958029955625534, 0.16093020141124725),
                tl.where(high == 10, 0.24611230194568634, 0.33791524171829224)
            ),
            tl.where(high < 14,
                tl.where(high == 12, 0.44070982933044434, 0.5626170039176941),
                tl.where(high == 14, 0.7229568362236023, 1.0)
            )
        )
    )
    
    # Apply scale
    low_scaled = low_vals * scale
    high_scaled = high_vals * scale
    
    # Store interleaved results
    for i in range(32):
        idx = i * 2
        if col_start + idx < N:
            tl.store(output_ptr + output_base + idx, low_scaled[i], eviction_policy="evict_first")
        if col_start + idx + 1 < N:
            tl.store(output_ptr + output_base + idx + 1, high_scaled[i], eviction_policy="evict_first")

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
    
    _your_dequantize_nf4_kernel[(total_blocks,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        num_warps=1,
        num_stages=1,
    )
    
    return output

def your_dequantize_nf4(weight):
    """Main entry point for the challenge."""
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state)