import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _extreme_optimized_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Extreme optimization - process 4 blocks (256 elements) per thread."""
    
    pid = tl.program_id(0)
    BLOCKS_PER_THREAD: tl.constexpr = 4
    
    start_block = pid * BLOCKS_PER_THREAD
    total_blocks = M * blocks_per_row
    
    # NF4 constants in registers
    NF4_SCALE: tl.constexpr = 0.00787401574803149606
    
    # Process 4 blocks per thread
    for b in tl.static_range(BLOCKS_PER_THREAD):
        block_id = start_block + b
        if block_id >= total_blocks:
            return
            
        row = block_id // blocks_per_row
        block_idx = block_id % blocks_per_row
        col_base = block_idx << 6
        
        if col_base >= N:
            continue
        
        # Load scales - optimize by reusing when possible
        absmax_val = tl.load(absmax_ptr + block_id).to(tl.float32)
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        absmax32_val = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax_val * NF4_SCALE * absmax32_val
        
        base_offset = row * N + col_base
        
        # Process all 64 elements in a single vectorized operation
        idx = base_offset + tl.arange(0, 64)
        cols = col_base + tl.arange(0, 64)
        mask = cols < N
        
        # Vectorized load of packed data
        packed_idx = idx >> 1
        packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
        
        # Extract nibbles efficiently
        is_odd = idx & 1
        nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
        
        # Direct computation avoiding lookup table overhead
        # Map nibbles to NF4 values using arithmetic operations
        # This is faster than conditional lookups for modern GPUs
        
        # Split into ranges for efficient computation
        is_zero = nibbles == 0
        is_seven = nibbles == 7
        is_fifteen = nibbles == 15
        
        # Base value computation
        base = (nibbles.to(tl.float32) - 7.5) * 0.2941176471
        
        # Apply corrections for specific values
        nf4_vals = tl.where(is_zero, -1.0,
                   tl.where(is_seven, 0.0,
                   tl.where(is_fifteen, 1.0, base)))
        
        # Additional corrections for intermediate values
        is_one = nibbles == 1
        is_two = nibbles == 2
        is_three = nibbles == 3
        is_four = nibbles == 4
        is_five = nibbles == 5
        is_six = nibbles == 6
        
        nf4_vals = tl.where(is_one, -0.6961928009986877,
                   tl.where(is_two, -0.5250730514526367,
                   tl.where(is_three, -0.39491748809814453,
                   tl.where(is_four, -0.28444138169288635,
                   tl.where(is_five, -0.18477343022823334,
                   tl.where(is_six, -0.09105003625154495, nf4_vals))))))
        
        # Upper half corrections
        is_eight = nibbles == 8
        is_nine = nibbles == 9
        is_ten = nibbles == 10
        is_eleven = nibbles == 11
        is_twelve = nibbles == 12
        is_thirteen = nibbles == 13
        is_fourteen = nibbles == 14
        
        nf4_vals = tl.where(is_eight, 0.07958029955625534,
                   tl.where(is_nine, 0.16093020141124725,
                   tl.where(is_ten, 0.24611230194568634,
                   tl.where(is_eleven, 0.33791524171829224,
                   tl.where(is_twelve, 0.44070982933044434,
                   tl.where(is_thirteen, 0.5626170039176941,
                   tl.where(is_fourteen, 0.7229568362236023, nf4_vals)))))))
        
        # Apply scale and store
        output = (nf4_vals * scale).to(dtype)
        tl.store(output_ptr + idx, output, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def _extreme_autotuned_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Autotuned extreme kernel."""
    _extreme_optimized_kernel(
        qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
        M, N, blocks_per_row, absmax32_per_row, dtype
    )

def extreme_optimized_dequantize_nf4(module):
    """Extreme optimized NF4 dequantization for maximum performance."""
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
    
    # Grid configuration - fewer threads, more work per thread
    BLOCKS_PER_THREAD = 4
    total_blocks = M * blocks_per_row
    grid_size = (total_blocks + BLOCKS_PER_THREAD - 1) // BLOCKS_PER_THREAD
    
    # Use autotuned kernel
    _extreme_autotuned_kernel[(grid_size,)](
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

# Export
triton_dequantize_nf4 = extreme_optimized_dequantize_nf4