import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _hyper_optimized_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Hyper-optimized kernel combining all performance techniques."""
    
    # Grid setup - each thread processes 2 blocks (128 elements)
    pid = tl.program_id(0)
    BLOCKS_PER_THREAD: tl.constexpr = 2
    
    start_block = pid * BLOCKS_PER_THREAD
    total_blocks = M * blocks_per_row
    
    # Pre-computed NF4 constants as constexpr for register allocation
    NF4_0: tl.constexpr = -1.0
    NF4_1: tl.constexpr = -0.6961928009986877
    NF4_2: tl.constexpr = -0.5250730514526367
    NF4_3: tl.constexpr = -0.39491748809814453
    NF4_4: tl.constexpr = -0.28444138169288635
    NF4_5: tl.constexpr = -0.18477343022823334
    NF4_6: tl.constexpr = -0.09105003625154495
    NF4_7: tl.constexpr = 0.0
    NF4_8: tl.constexpr = 0.07958029955625534
    NF4_9: tl.constexpr = 0.16093020141124725
    NF4_10: tl.constexpr = 0.24611230194568634
    NF4_11: tl.constexpr = 0.33791524171829224
    NF4_12: tl.constexpr = 0.44070982933044434
    NF4_13: tl.constexpr = 0.5626170039176941
    NF4_14: tl.constexpr = 0.7229568362236023
    NF4_15: tl.constexpr = 1.0
    NF4_SCALE: tl.constexpr = 0.00787401574803149606
    
    # Unroll loop over blocks
    for b in tl.static_range(BLOCKS_PER_THREAD):
        block_id = start_block + b
        if block_id >= total_blocks:
            return
            
        # Decode block position
        row = block_id // blocks_per_row
        block_idx = block_id % blocks_per_row
        col_base = block_idx << 6  # * 64
        
        if col_base >= N:
            continue
        
        # Load scales once per block
        absmax = tl.load(absmax_ptr + block_id).to(tl.float32)
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax * NF4_SCALE * absmax32
        
        base_offset = row * N + col_base
        
        # Process 64 elements in 4 chunks of 16 for maximum ILP
        for chunk in tl.static_range(4):
            offset = chunk << 4  # * 16
            idx = base_offset + offset + tl.arange(0, 16)
            cols = col_base + offset + tl.arange(0, 16)
            mask = cols < N
            
            # Vectorized load
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            
            # Optimized nibble extraction
            is_odd = idx & 1
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Direct mapping using nested conditionals (compiler optimizes this)
            nf4_vals = tl.where(nibbles == 0, NF4_0,
                       tl.where(nibbles == 1, NF4_1,
                       tl.where(nibbles == 2, NF4_2,
                       tl.where(nibbles == 3, NF4_3,
                       tl.where(nibbles == 4, NF4_4,
                       tl.where(nibbles == 5, NF4_5,
                       tl.where(nibbles == 6, NF4_6,
                       tl.where(nibbles == 7, NF4_7,
                       tl.where(nibbles == 8, NF4_8,
                       tl.where(nibbles == 9, NF4_9,
                       tl.where(nibbles == 10, NF4_10,
                       tl.where(nibbles == 11, NF4_11,
                       tl.where(nibbles == 12, NF4_12,
                       tl.where(nibbles == 13, NF4_13,
                       tl.where(nibbles == 14, NF4_14,
                       NF4_15)))))))))))))))
            
            # Apply scale and store
            output = (nf4_vals * scale).to(dtype)
            tl.store(output_ptr + idx, output, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def _hyper_autotuned_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Autotuned version for different problem sizes."""
    _hyper_optimized_kernel(
        qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
        M, N, blocks_per_row, absmax32_per_row, dtype
    )

def hyper_triton_dequantize_nf4(module):
    """Hyper-optimized NF4 dequantization targeting 1.15x+ speedup."""
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
            raise ValueError("Invalid absmax shape and fast_dequantize not available")
    
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
            raise ValueError("Invalid absmax32 shape and fast_dequantize not available")
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Grid configuration
    BLOCKS_PER_THREAD = 2
    total_blocks = M * blocks_per_row
    grid_size = (total_blocks + BLOCKS_PER_THREAD - 1) // BLOCKS_PER_THREAD
    
    # Use autotuned kernel for better performance
    if M * N > 1024 * 1024:  # Large matrices
        _hyper_autotuned_kernel[(grid_size,)](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            dtype,
        )
    else:  # Small matrices
        _hyper_optimized_kernel[(grid_size,)](
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
triton_dequantize_nf4 = hyper_triton_dequantize_nf4