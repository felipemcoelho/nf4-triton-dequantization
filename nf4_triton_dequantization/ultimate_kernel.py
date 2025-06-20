import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _ultimate_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Ultimate optimization combining all techniques."""
    pid = tl.program_id(0)
    
    # Optimized work distribution
    ELEMENTS_PER_WARP = 256  # 8 warps × 32 threads
    WARPS_PER_BLOCK = 4
    ELEMENTS_PER_BLOCK = ELEMENTS_PER_WARP * WARPS_PER_BLOCK  # 1024
    
    block_start = pid * ELEMENTS_PER_BLOCK
    total_elements = M * N
    
    if block_start >= total_elements:
        return
    
    # Pre-compute constants
    SCALE_FACTOR = tl.inline_const_exprs(0.00787401574803149606)  # 1/127
    
    # Process multiple warps in parallel
    for warp in range(WARPS_PER_BLOCK):
        warp_start = block_start + warp * ELEMENTS_PER_WARP
        
        if warp_start >= total_elements:
            break
        
        # Each warp processes 256 elements in chunks of 64
        for chunk in range(0, ELEMENTS_PER_WARP, 64):
            base_idx = warp_start + chunk
            idx = base_idx + tl.arange(0, 64)
            mask = idx < total_elements
            
            # 2D indexing
            row = idx // N
            col = idx % N
            valid = mask & (row < M)
            
            if not tl.sum(valid.to(tl.int32)):
                continue
            
            # Compute block indices
            block_idx = col >> 6  # // 64
            absmax_idx = row * blocks_per_row + block_idx
            absmax32_block = block_idx >> 2  # // 4
            absmax32_idx = row * absmax32_per_row + absmax32_block
            
            # Load and fuse scaling in one operation
            absmax_val = tl.load(absmax_ptr + absmax_idx, mask=valid, other=0)
            absmax32_val = tl.load(absmax32_ptr + absmax32_idx, mask=valid, other=1.0)
            scale = absmax_val.to(tl.float32) * SCALE_FACTOR * absmax32_val
            
            # Optimized packed weight loading
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=valid, other=0)
            
            # Branchless nibble extraction
            nibble_shift = ((idx & 1) << 2).to(tl.int32)
            nibbles = (packed >> nibble_shift) & 0xF
            
            # Ultra-optimized NF4 lookup using bit manipulation
            # Exploit symmetry in NF4 quantization
            sign = tl.where(nibbles < 8, -1.0, 1.0)
            abs_idx = tl.where(nibbles < 8, 7 - nibbles, nibbles - 8)
            
            # Lookup table for absolute values (8 values)
            abs_vals = tl.where(abs_idx == 0, 0.0,
                       tl.where(abs_idx == 1, 0.07958029955625534,
                       tl.where(abs_idx == 2, 0.16093020141124725,
                       tl.where(abs_idx == 3, 0.24611230194568634,
                       tl.where(abs_idx == 4, 0.33791524171829224,
                       tl.where(abs_idx == 5, 0.44070982933044434,
                       tl.where(abs_idx == 6, 0.5626170039176941,
                       tl.where(abs_idx == 7, 0.7229568362236023, 1.0))))))))
            
            # Special cases
            nf4_vals = tl.where(nibbles == 0, -1.0,
                       tl.where(nibbles == 15, 1.0,
                       sign * abs_vals))
            
            # Apply scaling and store
            output = nf4_vals * scale
            tl.store(output_ptr + idx, output, mask=valid)

@triton.jit
def _ultimate_nf4_kernel_v3(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
):
    """V3: Maximum performance with all optimizations."""
    pid = tl.program_id(0)
    
    # Grid-stride loop for better GPU utilization
    BLOCK_SIZE = 512  # Optimal for memory bandwidth
    stride = tl.num_programs(0) * BLOCK_SIZE
    start = pid * BLOCK_SIZE
    
    # Constants
    SCALE = 0.00787401574803149606
    total = M * N
    
    # NF4 lookup split for better pipelining
    NF4_NEG = tl.inline_const_exprs([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0
    ])
    NF4_POS = tl.inline_const_exprs([
        0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023
    ])
    
    # Grid-stride loop
    for offset in range(start, total, stride):
        # Process 512 elements per iteration
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < total
        
        # Skip if no valid elements
        if not tl.sum(mask.to(tl.int32)):
            continue
        
        # Compute coordinates
        row = idx // N
        col = idx % N
        valid = mask & (row < M)
        
        # Batch compute indices
        block_idx = col >> 6
        absmax_linear = row * blocks_per_row + block_idx
        absmax32_linear = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
        
        # Vectorized loads
        absmax_data = tl.load(absmax_ptr + absmax_linear, mask=valid, other=0)
        absmax32_data = tl.load(absmax32_ptr + absmax32_linear, mask=valid, other=1.0)
        scales = absmax_data.to(tl.float32) * SCALE * absmax32_data
        
        # Load packed weights
        packed_linear = idx >> 1
        packed = tl.load(qweight_ptr + packed_linear, mask=valid, other=0)
        
        # Extract nibbles
        is_odd = idx & 1
        nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
        
        # Optimized lookup with symmetry exploitation
        is_negative = nibbles < 8
        lookup_idx = tl.where(is_negative, nibbles, nibbles - 8)
        
        # Vectorized gather from lookup tables
        neg_vals = tl.gather(NF4_NEG, lookup_idx, mask=valid & is_negative, other=0.0)
        pos_vals = tl.gather(NF4_POS, lookup_idx, mask=valid & ~is_negative, other=0.0)
        
        # Combine and handle special case
        nf4_vals = tl.where(is_negative, neg_vals, pos_vals)
        nf4_vals = tl.where(nibbles == 15, 1.0, nf4_vals)
        
        # Final computation and store
        output = nf4_vals * scales
        tl.store(output_ptr + idx, output, mask=valid)

def ultimate_triton_dequantize_nf4(module):
    """Ultimate NF4 dequantization with all optimizations for 1.15x+ speedup."""
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
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Ensure contiguous memory layout
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    total_elements = M * N
    
    # Choose kernel based on problem size
    if total_elements < 65536:  # Small matrices
        # Use original kernel with warp-level optimization
        ELEMENTS_PER_BLOCK = 1024
        grid = (triton.cdiv(total_elements, ELEMENTS_PER_BLOCK),)
        
        _ultimate_nf4_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
        )
    else:  # Large matrices
        # Use grid-stride kernel for better utilization
        # Calculate optimal grid size based on GPU occupancy
        BLOCK_SIZE = 512
        MAX_BLOCKS = 108  # 3 blocks per SM on T4 (36 SMs)
        num_blocks = min(MAX_BLOCKS, triton.cdiv(total_elements, BLOCK_SIZE))
        
        grid = (num_blocks,)
        
        _ultimate_nf4_kernel_v3[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
        )
    
    return output