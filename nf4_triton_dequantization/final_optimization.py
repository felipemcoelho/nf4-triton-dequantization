import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

# Enable aggressive compiler optimizations
triton.Config.compiler_num_warps = 4
triton.Config.compiler_num_stages = 3

@triton.jit
def _final_nf4_dequant_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    total_elements,
    N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Final ultra-optimized kernel with all possible optimizations."""
    pid = tl.program_id(0)
    
    # Constants for maximum performance
    SCALE = tl.constexpr(0.00787401574803149606)  # 1/127
    
    # Grid-stride loop for perfect load balancing
    block_start = pid * BLOCK_SIZE
    grid_stride = tl.num_programs(0) * BLOCK_SIZE
    
    # NF4 lookup table split for ILP
    nf4_0_7 = tl.constexpr([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0])
    nf4_8_15 = tl.constexpr([0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                             0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
    
    # Main loop with grid-stride pattern
    for idx_base in range(block_start, total_elements, grid_stride):
        # Process BLOCK_SIZE elements
        idx = idx_base + tl.arange(0, BLOCK_SIZE)
        mask = idx < total_elements
        
        # Compute row/col from linear index
        row = idx // N
        col = idx % N
        
        # Calculate all indices at once
        block_idx = col >> 6
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        packed_idx = idx >> 1
        
        # Vectorized loads with prefetching
        packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0, cache_modifier=".ca")
        absmax = tl.load(absmax_ptr + absmax_idx, mask=mask, other=0, cache_modifier=".ca")
        absmax32 = tl.load(absmax32_ptr + absmax32_idx, mask=mask, other=1.0, cache_modifier=".ca")
        
        # Fused scaling computation
        scale = absmax.to(tl.float32) * SCALE * absmax32
        
        # Optimized nibble extraction
        nibbles = tl.where(idx & 1, (packed >> 4) & 0xF, packed & 0xF)
        
        # Ultra-fast lookup with minimal branches
        # Use bit manipulation to create lookup index
        is_high = nibbles >= 8
        lookup_idx = tl.where(is_high, nibbles - 8, nibbles)
        
        # Parallel lookup from both tables
        low_vals = tl.load(nf4_0_7 + lookup_idx, mask=mask & ~is_high, other=0.0)
        high_vals = tl.load(nf4_8_15 + lookup_idx, mask=mask & is_high, other=0.0)
        
        # Combine results
        nf4_vals = tl.where(is_high, high_vals, low_vals)
        
        # Apply scaling and store
        output = nf4_vals * scale
        tl.store(output_ptr + idx, output, mask=mask, cache_modifier=".cs")

@triton.jit
def _final_nf4_2d_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D tiled kernel for small matrices."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Constants
    SCALE = 0.00787401574803149606
    
    # Compute block boundaries
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Early exit
    if m_start >= M or n_start >= N:
        return
    
    # Process the tile
    for m in range(BLOCK_M):
        row = m_start + m
        if row >= M:
            break
        
        # Process columns in chunks of 64 (absmax block size)
        for n_offset in range(0, BLOCK_N, 64):
            col_base = n_start + n_offset
            if col_base >= N:
                break
            
            # Vectorized processing of 64 elements
            cols = col_base + tl.arange(0, 64)
            mask = cols < N
            
            # Compute indices
            block_idx = col_base >> 6
            absmax_idx = row * blocks_per_row + block_idx
            absmax32_idx = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
            
            # Load scaling factors once per block
            absmax_val = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            absmax32_val = tl.load(absmax32_ptr + absmax32_idx)
            scale = absmax_val * SCALE * absmax32_val
            
            # Process 64 elements in two vectorized operations
            for i in range(0, 64, 32):
                col_vec = cols[i:i+32]
                vec_mask = mask[i:i+32]
                
                # Linear indices
                linear_idx = row * N + col_vec
                packed_idx = linear_idx >> 1
                
                # Load packed data
                packed = tl.load(qweight_ptr + packed_idx, mask=vec_mask, other=0)
                
                # Extract nibbles
                is_odd = linear_idx & 1
                nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
                
                # Optimized NF4 lookup
                nf4_vals = (
                    (nibbles == 0) * (-1.0) +
                    (nibbles == 1) * (-0.6961928009986877) +
                    (nibbles == 2) * (-0.5250730514526367) +
                    (nibbles == 3) * (-0.39491748809814453) +
                    (nibbles == 4) * (-0.28444138169288635) +
                    (nibbles == 5) * (-0.18477343022823334) +
                    (nibbles == 6) * (-0.09105003625154495) +
                    (nibbles == 7) * 0.0 +
                    (nibbles == 8) * 0.07958029955625534 +
                    (nibbles == 9) * 0.16093020141124725 +
                    (nibbles == 10) * 0.24611230194568634 +
                    (nibbles == 11) * 0.33791524171829224 +
                    (nibbles == 12) * 0.44070982933044434 +
                    (nibbles == 13) * 0.5626170039176941 +
                    (nibbles == 14) * 0.7229568362236023 +
                    (nibbles == 15) * 1.0
                )
                
                # Apply scaling and store
                output = nf4_vals * scale
                output_idx = row * N + col_vec
                tl.store(output_ptr + output_idx, output, mask=vec_mask)

def final_triton_dequantize_nf4(module):
    """Final optimized NF4 dequantization implementation."""
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
    
    # Ensure contiguous memory
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    total_elements = M * N
    
    # Choose optimal kernel configuration
    if M <= 32 and N % 256 == 0:
        # Small M with aligned N - use 2D tiling
        BLOCK_M = min(8, M)
        BLOCK_N = 256
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        _final_nf4_2d_kernel[grid](
            qweight,
            absmax,
            absmax32,
            output,
            M, N,
            blocks_per_row,
            BLOCK_M,
            BLOCK_N,
        )
    else:
        # General case - use 1D grid with optimal block size
        # Tune block size for Tesla T4 architecture
        if total_elements < 32768:
            BLOCK_SIZE = 256
        elif total_elements < 262144:
            BLOCK_SIZE = 512
        else:
            BLOCK_SIZE = 1024
        
        # Limit grid size for better occupancy
        max_blocks = 108  # 3 blocks per SM on T4
        num_blocks = min(max_blocks, triton.cdiv(total_elements, BLOCK_SIZE))
        
        grid = (num_blocks,)
        
        _final_nf4_dequant_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            total_elements,
            N,
            blocks_per_row,
            absmax32_per_row,
            BLOCK_SIZE,
        )
    
    return output