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
):
    """Ultimate NF4 kernel optimized for memory bandwidth."""
    pid = tl.program_id(0)
    
    # Each thread block processes multiple 64-element chunks
    CHUNKS_PER_BLOCK = 16  # Process 1024 elements per block
    total_chunks = (M * N + 63) // 64
    
    chunk_id = pid * CHUNKS_PER_BLOCK
    if chunk_id >= total_chunks:
        return
    
    # Constants
    SCALE = 0.00787401574803149606
    
    # Process multiple chunks
    for c in range(CHUNKS_PER_BLOCK):
        chunk = chunk_id + c
        if chunk >= total_chunks:
            break
        
        # Calculate position of this 64-element chunk
        base_idx = chunk * 64
        row = base_idx // N
        col_start = base_idx - row * N
        
        # Skip if row is out of bounds
        if row >= M:
            break
        
        # Load absmax values once for the entire 64-element block
        block_idx = col_start >> 6
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
        
        absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax * SCALE * absmax32
        
        # Process 64 elements in two vectorized operations
        for offset in range(0, 64, 32):
            idx = base_idx + offset + tl.arange(0, 32)
            col = col_start + offset + tl.arange(0, 32)
            mask = (row < M) & (col < N)
            
            # Load packed data
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            
            # Extract nibbles
            is_odd = idx & 1
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Optimized NF4 lookup
            # Use the fact that NF4 values are symmetric around 7.5
            is_negative = nibbles < 8
            flipped = tl.where(is_negative, 7 - nibbles, nibbles - 8)
            
            # Base lookup for positive side
            pos_vals = tl.where(flipped == 0, 0.0,
                       tl.where(flipped == 1, 0.07958029955625534,
                       tl.where(flipped == 2, 0.16093020141124725,
                       tl.where(flipped == 3, 0.24611230194568634,
                       tl.where(flipped == 4, 0.33791524171829224,
                       tl.where(flipped == 5, 0.44070982933044434,
                       tl.where(flipped == 6, 0.5626170039176941,
                       tl.where(flipped == 7, 0.7229568362236023,
                       1.0))))))))
            
            # Apply sign and handle special case
            nf4_vals = tl.where(is_negative & (flipped != 0), -pos_vals, pos_vals)
            nf4_vals = tl.where(nibbles == 0, -1.0, nf4_vals)
            
            # Apply scaling and store
            output = nf4_vals * scale
            tl.store(output_ptr + idx, output, mask=mask)

def ultimate_triton_dequantize_nf4(module):
    """Ultimate optimized NF4 dequantization."""
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
    
    # Launch kernel
    total_chunks = (M * N + 63) // 64
    CHUNKS_PER_BLOCK = 16
    grid = (triton.cdiv(total_chunks, CHUNKS_PER_BLOCK),)
    
    _ultimate_nf4_kernel[grid](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
    )
    
    return output