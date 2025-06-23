import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _ultra_fast_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Ultra-fast NF4 dequantization with maximum parallelism and vectorization."""
    pid = tl.program_id(0)
    
    # Calculate number of column groups
    num_col_groups = (n + GROUP_SIZE - 1) // GROUP_SIZE
    total_blocks = m * num_col_groups
    
    if pid >= total_blocks:
        return
    
    # Determine which row and column group this thread handles
    row = pid // num_col_groups
    col_group = pid % num_col_groups
    col_start = col_group * GROUP_SIZE
    
    if row >= m or col_start >= n:
        return
    
    # Pre-compute NF4 lookup values
    nf4_0 = -1.0
    nf4_1 = -0.6961928009986877
    nf4_2 = -0.5250730514526367
    nf4_3 = -0.39491748809814453
    nf4_4 = -0.28444138169288635
    nf4_5 = -0.18477343022823334
    nf4_6 = -0.09105003625154495
    nf4_7 = 0.0
    nf4_8 = 0.07958029955625534
    nf4_9 = 0.16093020141124725
    nf4_10 = 0.24611230194568634
    nf4_11 = 0.33791524171829224
    nf4_12 = 0.44070982933044434
    nf4_13 = 0.5626170039176941
    nf4_14 = 0.7229568362236023
    nf4_15 = 1.0
    
    # Process GROUP_SIZE elements, aligned to 64-element boundaries
    for block_offset in range(0, GROUP_SIZE, 64):
        col_block_start = col_start + block_offset
        if col_block_start >= n:
            break
        
        # Calculate block indices for absmax values
        block_idx = col_block_start >> 6
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        
        # Load and compute scale once per 64-element block
        absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax * 0.00787401574803149606 * absmax32
        
        # Process in BLOCK_SIZE chunks for better vectorization
        for chunk in range(0, 64, BLOCK_SIZE):
            col = col_block_start + chunk + tl.arange(0, BLOCK_SIZE)
            idx = row * n + col
            mask = col < n
            
            # Vectorized load of packed weights
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            
            # Extract nibbles using vectorized bit operations
            is_odd = (idx & 1).to(tl.int1)
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Ultra-fast lookup using direct conditionals (minimized branching)
            # Split into groups for better performance
            is_0_7 = nibbles < 8
            is_0_3 = nibbles < 4
            is_0_1 = nibbles < 2
            is_4_5 = nibbles < 6
            is_8_11 = nibbles < 12
            is_8_9 = nibbles < 10
            is_12_13 = nibbles < 14
            
            # Compute lookup values with minimal branching
            val_0_1 = tl.where(nibbles == 0, nf4_0, nf4_1)
            val_2_3 = tl.where(nibbles == 2, nf4_2, nf4_3)
            val_0_3 = tl.where(is_0_1, val_0_1, val_2_3)
            
            val_4_5 = tl.where(nibbles == 4, nf4_4, nf4_5)
            val_6_7 = tl.where(nibbles == 6, nf4_6, nf4_7)
            val_4_7 = tl.where(is_4_5, val_4_5, val_6_7)
            
            val_0_7 = tl.where(is_0_3, val_0_3, val_4_7)
            
            val_8_9 = tl.where(nibbles == 8, nf4_8, nf4_9)
            val_10_11 = tl.where(nibbles == 10, nf4_10, nf4_11)
            val_8_11 = tl.where(is_8_9, val_8_9, val_10_11)
            
            val_12_13 = tl.where(nibbles == 12, nf4_12, nf4_13)
            val_14_15 = tl.where(nibbles == 14, nf4_14, nf4_15)
            val_12_15 = tl.where(is_12_13, val_12_13, val_14_15)
            
            val_8_15 = tl.where(is_8_11, val_8_11, val_12_15)
            
            nf4_vals = tl.where(is_0_7, val_0_7, val_8_15)
            
            # Apply scaling and store with proper dtype
            output = (nf4_vals * scale).to(dtype)
            tl.store(output_ptr + idx, output, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32, 'GROUP_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64, 'GROUP_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32, 'GROUP_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64, 'GROUP_SIZE': 512}, num_warps=8),
    ],
    key=['m', 'n'],
)
@triton.jit
def _autotuned_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Auto-tuned version of the ultra-fast kernel."""
    _ultra_fast_nf4_kernel(
        qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
        m, n, blocks_per_row, absmax32_per_row,
        BLOCK_SIZE, GROUP_SIZE, dtype
    )

def ultra_fast_triton_dequantize_nf4(module):
    """Ultra-fast NF4 dequantization with all optimizations."""
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
    
    # Prepare absmax tensors
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
    
    # Ensure contiguous memory layout
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Allocate output tensor
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Choose optimal configuration based on problem size
    use_autotuned = M * N > 2 * 1024 * 1024  # Use autotuning for large matrices
    
    if use_autotuned:
        # For large matrices, use autotuned kernel
        GROUP_SIZE = 512 if N >= 512 else 256
        num_col_groups = (N + GROUP_SIZE - 1) // GROUP_SIZE
        grid = (M * num_col_groups,)
        
        _autotuned_nf4_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            dtype=dtype,
        )
    else:
        # For smaller matrices, use fixed configuration
        BLOCK_SIZE = 32
        GROUP_SIZE = 256
        num_col_groups = (N + GROUP_SIZE - 1) // GROUP_SIZE
        grid = (M * num_col_groups,)
        
        _ultra_fast_nf4_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            BLOCK_SIZE,
            GROUP_SIZE,
            dtype,
        )
    
    return output