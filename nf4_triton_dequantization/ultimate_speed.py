import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _ultimate_speed_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Ultimate speed kernel - absolute minimal work per thread."""
    
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
    absmax_val = tl.load(absmax_ptr + bid)
    absmax32_idx = row * absmax32_per_row + (block_in_row >> 2)
    absmax32_val = tl.load(absmax32_ptr + absmax32_idx)
    scale = absmax_val * SCALE * absmax32_val
    
    # Base addresses  
    qweight_base = row * (N >> 1) + (col_start >> 1)
    output_base = row * N + col_start
    
    # Load all 32 bytes for this block at once
    qweight_data = qweight_base + tl.arange(0, 32)
    packed = tl.load(qweight_ptr + qweight_data)
    
    # Extract low and high nibbles
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    
    # NF4 lookup - use select tree for maximum performance
    # Low nibbles
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
    
    # High nibbles
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
    # Process in two halves for better performance
    for half in range(2):
        base = half * 32
        for i in range(16):
            idx = base + i * 2
            if col_start + idx < N:
                tl.store(output_ptr + output_base + idx, low_scaled[half * 16 + i])
            if col_start + idx + 1 < N:
                tl.store(output_ptr + output_base + idx + 1, high_scaled[half * 16 + i])

def ultimate_speed_dequantize_nf4(module):
    """Ultimate speed NF4 dequantization."""
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
    
    # Handle tensor shapes
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
    
    # Force contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous().to(torch.float32)
    absmax32 = absmax32.contiguous()
    
    # Pre-allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch kernel - one thread per block for simplicity
    total_blocks = M * blocks_per_row
    
    _ultimate_speed_kernel[(total_blocks,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output,
        M, N,
        blocks_per_row,
        absmax32_per_row,
        num_warps=1,
        num_stages=1,
    )
    
    return output

# Export
triton_dequantize_nf4 = ultimate_speed_dequantize_nf4