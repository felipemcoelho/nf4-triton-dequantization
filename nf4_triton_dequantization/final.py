import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _final_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
):
    """Final optimized NF4 kernel."""
    pid = tl.program_id(0)
    
    # Each thread processes one 64-element block
    total_blocks = (m * n + 63) // 64
    if pid >= total_blocks:
        return
    
    # Calculate block position
    base_idx = pid * 64
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
    
    # Process 64 elements with maximum vectorization
    # Unroll into 4 iterations of 16 elements for better ILP
    for i in range(0, 64, 16):
        idx = base_idx + i + tl.arange(0, 16)
        col = col_start + i + tl.arange(0, 16)
        mask = col < n
        
        # Load packed data (8 bytes = 16 nibbles)
        packed_idx = idx >> 1
        packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
        
        # Extract nibbles
        is_odd = idx & 1
        nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
        
        # Optimized NF4 lookup using minimal comparisons
        # Split into groups to reduce dependencies
        is_0_7 = nibbles < 8
        is_0_3 = nibbles < 4
        is_0_1 = nibbles < 2
        is_4_5 = (nibbles >= 4) & (nibbles < 6)
        is_8_11 = (nibbles >= 8) & (nibbles < 12)
        is_8_9 = (nibbles >= 8) & (nibbles < 10)
        is_12_13 = (nibbles >= 12) & (nibbles < 14)
        
        # Compute NF4 values with reduced branching
        nf4 = tl.where(nibbles == 0, -1.0,
              tl.where(nibbles == 7, 0.0,
              tl.where(nibbles == 15, 1.0,
              tl.where(is_0_1 & (nibbles == 1), -0.6961928009986877,
              tl.where(is_0_3 & (nibbles == 2), -0.5250730514526367,
              tl.where(is_0_3 & (nibbles == 3), -0.39491748809814453,
              tl.where(is_4_5 & (nibbles == 4), -0.28444138169288635,
              tl.where(is_4_5 & (nibbles == 5), -0.18477343022823334,
              tl.where(nibbles == 6, -0.09105003625154495,
              tl.where(is_8_9 & (nibbles == 8), 0.07958029955625534,
              tl.where(is_8_9 & (nibbles == 9), 0.16093020141124725,
              tl.where(is_8_11 & (nibbles == 10), 0.24611230194568634,
              tl.where(is_8_11 & (nibbles == 11), 0.33791524171829224,
              tl.where(is_12_13 & (nibbles == 12), 0.44070982933044434,
              tl.where(is_12_13 & (nibbles == 13), 0.5626170039176941,
              0.7229568362236023)))))))))))))))
        
        # Apply scaling and store
        output = nf4 * scale
        tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def _final_nf4_kernel_v2(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
):
    """V2 optimized for memory bandwidth."""
    pid = tl.program_id(0)
    
    # Grid-stride loop for better load balancing
    ELEMENTS_PER_ITER = 1024  # Process 16 absmax blocks at once
    total = m * n
    
    for start in range(pid * ELEMENTS_PER_ITER, total, tl.num_programs(0) * ELEMENTS_PER_ITER):
        # Process in 128-element chunks for better vectorization
        for chunk_offset in range(0, ELEMENTS_PER_ITER, 128):
            base = start + chunk_offset
            if base >= total:
                break
            
            # Vectorized processing
            idx = base + tl.arange(0, 128)
            mask = idx < total
            
            # Calculate positions
            row = idx // n
            col = idx % n
            
            # Calculate absmax indices
            block_idx = col >> 6
            absmax_idx = row * blocks_per_row + block_idx
            absmax32_idx = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
            
            # Load all data
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            absmax = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=1.0)
            
            # Extract nibbles
            nibbles = tl.where(idx & 1, (packed >> 4) & 0xF, packed & 0xF)
            
            # Compute scale
            scale = absmax.to(tl.float32) * 0.00787401574803149606 * absmax32
            
            # Optimized lookup with arithmetic operations
            # Use polynomial approximation for middle values
            x = nibbles.to(tl.float32)
            
            # Special cases
            is_special = (nibbles == 0) | (nibbles == 7) | (nibbles == 15)
            special_vals = tl.where(nibbles == 0, -1.0,
                          tl.where(nibbles == 7, 0.0, 1.0))
            
            # For other values, use direct mapping
            sign = tl.where(x < 8, -1.0, 1.0)
            idx_norm = tl.where(x < 8, 7 - x, x - 7)
            
            # Direct value mapping
            mapped = tl.where(idx_norm == 1, 0.07958029955625534,
                     tl.where(idx_norm == 2, 0.16093020141124725,
                     tl.where(idx_norm == 3, 0.24611230194568634,
                     tl.where(idx_norm == 4, 0.33791524171829224,
                     tl.where(idx_norm == 5, 0.44070982933044434,
                     tl.where(idx_norm == 6, 0.5626170039176941,
                     0.7229568362236023))))))
            
            # Combine special cases and regular values
            nf4 = tl.where(is_special, special_vals, sign * mapped)
            
            # Fix specific values
            nf4 = tl.where(nibbles == 1, -0.6961928009986877, nf4)
            nf4 = tl.where(nibbles == 14, 0.7229568362236023, nf4)
            
            # Apply scaling and store
            output = nf4 * scale
            tl.store(output_ptr + idx, output, mask=mask)

def final_triton_dequantize_nf4(module):
    """Final optimized NF4 dequantization."""
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
    
    total_elements = M * N
    
    # Use different strategies based on size
    if N % 64 == 0 and total_elements < 262144:
        # Use V1 for aligned small matrices
        total_blocks = (total_elements + 63) // 64
        grid = (total_blocks,)
        
        _final_nf4_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
        )
    else:
        # Use V2 with grid-stride for better performance
        ELEMENTS_PER_ITER = 1024
        # Optimal grid size for Tesla T4
        num_blocks = min(72, triton.cdiv(total_elements, ELEMENTS_PER_ITER))
        grid = (num_blocks,)
        
        _final_nf4_kernel_v2[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
        )
    
    return output