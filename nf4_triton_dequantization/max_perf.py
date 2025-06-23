import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _max_perf_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
):
    """Maximum performance kernel with all optimizations."""
    pid = tl.program_id(0)
    
    # Process one 64-element block per thread for simplicity
    block_id = pid
    total_blocks = (m * n + 63) // 64
    
    if block_id >= total_blocks:
        return
    
    # Calculate block position
    base_idx = block_id * 64
    row = base_idx // n
    col_start = base_idx % n
    
    if row >= m:
        return
    
    # Load absmax values once for the entire 64-element block
    block_idx = col_start >> 6
    absmax_idx = row * blocks_per_row + block_idx
    absmax32_idx = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
    
    absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx)
    scale = absmax * 0.00787401574803149606 * absmax32
    
    # Process 64 elements
    idx = base_idx + tl.arange(0, 64)
    col = col_start + tl.arange(0, 64)
    mask = col < n
    
    # Load packed data
    packed_idx = idx >> 1
    packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
    
    # Extract nibbles
    nibbles = tl.where(idx & 1, (packed >> 4) & 0xF, packed & 0xF)
    
    # Direct NF4 lookup
    nf4 = tl.where(nibbles == 0, -1.0,
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
          tl.where(nibbles == 14, 0.7229568362236023,
          1.0)))))))))))))))
    
    # Apply scaling and store
    output = nf4 * scale
    tl.store(output_ptr + idx, output, mask=mask)

def max_perf_triton_dequantize_nf4(module):
    """Maximum performance NF4 dequantization."""
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
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch kernel with optimal configuration
    total_blocks = (M * N + 63) // 64
    grid = (total_blocks,)
    
    _max_perf_kernel[grid](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
    )
    
    return output