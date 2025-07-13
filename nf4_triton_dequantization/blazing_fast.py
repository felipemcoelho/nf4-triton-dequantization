import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def nf4_lookup(nibble):
    """Optimized NF4 lookup using binary search tree."""
    return tl.where(nibble < 8,
        tl.where(nibble < 4,
            tl.where(nibble < 2,
                tl.where(nibble == 0, -1.0, -0.6961928),
                tl.where(nibble == 2, -0.5250731, -0.3949175)
            ),
            tl.where(nibble < 6,
                tl.where(nibble == 4, -0.2844414, -0.1847734),
                tl.where(nibble == 6, -0.09105004, 0.0)
            )
        ),
        tl.where(nibble < 12,
            tl.where(nibble < 10,
                tl.where(nibble == 8, 0.0795803, 0.1609302),
                tl.where(nibble == 10, 0.2461123, 0.3379152)
            ),
            tl.where(nibble < 14,
                tl.where(nibble == 12, 0.4407098, 0.562617),
                tl.where(nibble == 14, 0.7229568, 1.0)
            )
        )
    )

@triton.jit
def _blazing_fast_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Blazing fast kernel with extreme optimizations."""
    
    # 2D blocking for better cache usage
    pid = tl.program_id(0)
    
    # Decode 2D position
    blocks_per_m = tl.cdiv(blocks_per_row, BLOCK_K)
    m_block = pid // blocks_per_m
    k_block = pid % blocks_per_m
    
    if m_block >= tl.cdiv(M, BLOCK_M):
        return
    
    # Process BLOCK_M rows and BLOCK_K blocks
    m_start = m_block * BLOCK_M
    k_start = k_block * BLOCK_K
    
    # Constants
    SCALE: tl.constexpr = 0.00787401574803149606
    
    # Process each row in the block
    for m_offset in range(min(BLOCK_M, M - m_start)):
        row = m_start + m_offset
        
        # Process BLOCK_K 64-element blocks
        for k_offset in range(min(BLOCK_K, blocks_per_row - k_start)):
            block_idx = k_start + k_offset
            col_base = block_idx * 64
            
            if col_base >= N:
                continue
            
            # Load scales
            absmax_idx = row * blocks_per_row + block_idx
            absmax_val = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            
            absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
            absmax32_val = tl.load(absmax32_ptr + absmax32_idx)
            
            scale = absmax_val * SCALE * absmax32_val
            
            # Process 64 elements in vectorized fashion
            qweight_base = row * (N >> 1) + (col_base >> 1)
            output_base = row * N + col_base
            
            # Load and process in chunks of 32 for better vectorization
            for chunk in range(2):
                offset = chunk * 16
                
                # Load 16 packed bytes
                packed_idx = qweight_base + offset + tl.arange(0, 16)
                packed = tl.load(qweight_ptr + packed_idx)
                
                # Extract nibbles
                low = packed & 0xF
                high = (packed >> 4) & 0xF
                
                # Lookup values
                low_vals = nf4_lookup(low)
                high_vals = nf4_lookup(high)
                
                # Scale
                low_scaled = low_vals * scale
                high_scaled = high_vals * scale
                
                # Store interleaved
                out_base = output_base + chunk * 32
                
                # Vectorized stores
                indices = tl.arange(0, 32)
                mask = (col_base + chunk * 32 + indices) < N
                
                # Even positions
                even_mask = mask & ((indices & 1) == 0)
                even_idx = out_base + indices
                even_vals = tl.where(even_mask, low_scaled[indices >> 1], 0.0)
                tl.store(output_ptr + even_idx, even_vals, mask=even_mask)
                
                # Odd positions
                odd_mask = mask & ((indices & 1) == 1)
                odd_idx = out_base + indices
                odd_vals = tl.where(odd_mask, high_scaled[indices >> 1], 0.0)
                tl.store(output_ptr + odd_idx, odd_vals, mask=odd_mask)

def blazing_fast_dequantize_nf4(module):
    """Blazing fast NF4 dequantization."""
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
    
    # Handle shapes
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
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # 2D blocking for cache efficiency
    BLOCK_M = 4
    BLOCK_K = 16  # Process 16 64-element blocks at once
    
    blocks_per_m = triton.cdiv(blocks_per_row, BLOCK_K)
    total_blocks = triton.cdiv(M, BLOCK_M) * blocks_per_m
    
    _blazing_fast_kernel[(total_blocks,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        BLOCK_M,
        BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    
    return output

# Export
triton_dequantize_nf4 = blazing_fast_dequantize_nf4