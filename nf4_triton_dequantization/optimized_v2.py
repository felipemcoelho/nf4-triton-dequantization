import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _ultra_optimized_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Ultra-optimized NF4 kernel targeting 1.15x+ speedup.
    
    Key optimizations:
    1. Process multiple blocks per thread for better efficiency
    2. Use shared memory for NF4 lookup table
    3. Vectorized loads and stores
    4. Optimized memory access patterns
    5. Minimal register usage
    """
    # Grid setup - each program processes multiple 64-element blocks
    pid = tl.program_id(0)
    BLOCKS_PER_THREAD: tl.constexpr = 4  # Process 4 blocks per thread
    
    # Calculate which blocks this thread will process
    start_block = pid * BLOCKS_PER_THREAD
    total_blocks = M * blocks_per_row
    
    # NF4 lookup table - will be placed in shared memory by compiler
    nf4_lut = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Process multiple blocks per thread
    for block_offset in range(BLOCKS_PER_THREAD):
        block_id = start_block + block_offset
        if block_id >= total_blocks:
            return
        
        # Decode block position
        row = block_id // blocks_per_row
        block_idx = block_id % blocks_per_row
        col_base = block_idx * 64
        
        if col_base >= N:
            continue
        
        # Load scales once per block
        absmax_idx = block_id
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        
        absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax * 0.00787401574803149606 * absmax32
        
        # Process in chunks for better vectorization
        base_offset = row * N + col_base
        
        # Process 64 elements in 4 iterations of 16 elements each
        for i in range(4):
            # Load 16 elements at once
            chunk_offset = i * 16
            idx = base_offset + chunk_offset + tl.arange(0, 16)
            cols = col_base + chunk_offset + tl.arange(0, 16)
            mask = cols < N
            
            # Optimized nibble extraction
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            
            # Extract nibbles using bit manipulation
            is_odd = idx & 1
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Vectorized lookup using shared memory
            nf4_vals = tl.load(nf4_lut + nibbles, mask=mask, other=0.0)
            
            # Apply scale and store
            output = (nf4_vals * scale).to(dtype)
            tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def _fused_matmul_nf4_kernel(
    x_ptr,
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    batch_size, seq_len, in_features, out_features,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    x_dtype: tl.constexpr,
    w_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused dequantization and matrix multiplication kernel.
    
    This kernel fuses NF4 dequantization with matrix multiplication
    to reduce memory bandwidth requirements.
    """
    # Grid setup
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Output tile boundaries
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # NF4 lookup table in shared memory
    nf4_lut = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Main loop over K dimension
    for k in range(0, in_features, BLOCK_K):
        # Load activations
        x_mask = (rm[:, None] < batch_size * seq_len) & (k + tl.arange(0, BLOCK_K)[None, :] < in_features)
        x_tile = tl.load(x_ptr + rm[:, None] * in_features + (k + tl.arange(0, BLOCK_K)[None, :]), 
                        mask=x_mask, other=0.0)
        
        # Dequantize weights on-the-fly
        for n in range(BLOCK_N):
            col = rn[n]
            if col >= out_features:
                continue
                
            # Process BLOCK_K elements of this weight column
            for k_offset in range(0, BLOCK_K, 64):
                if k + k_offset >= in_features:
                    break
                    
                # Load scale factors
                block_idx = (k + k_offset) >> 6
                absmax_idx = col * blocks_per_row + block_idx
                absmax32_idx = col * absmax32_per_row + (block_idx >> 2)
                
                absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
                absmax32 = tl.load(absmax32_ptr + absmax32_idx)
                scale = absmax * 0.00787401574803149606 * absmax32
                
                # Dequantize 64 elements
                for i in range(min(64, BLOCK_K - k_offset)):
                    idx = col * in_features + k + k_offset + i
                    packed_idx = idx >> 1
                    packed = tl.load(qweight_ptr + packed_idx)
                    
                    nibble = tl.where(idx & 1, (packed >> 4) & 0xF, packed & 0xF)
                    nf4_val = tl.load(nf4_lut + nibble)
                    
                    # Accumulate
                    for m in range(BLOCK_M):
                        if rm[m] < batch_size * seq_len and k + k_offset + i < in_features:
                            acc[m, n] += x_tile[m, k_offset + i] * nf4_val * scale
    
    # Store output
    output_mask = (rm[:, None] < batch_size * seq_len) & (rn[None, :] < out_features)
    output_indices = rm[:, None] * out_features + rn[None, :]
    tl.store(output_ptr + output_indices, acc.to(w_dtype), mask=output_mask)

def optimized_v2_triton_dequantize_nf4(module):
    """Optimized NF4 dequantization targeting 1.15x+ speedup."""
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
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Optimized grid configuration
    BLOCKS_PER_THREAD = 4
    total_blocks = M * blocks_per_row
    grid_size = (total_blocks + BLOCKS_PER_THREAD - 1) // BLOCKS_PER_THREAD
    
    # Launch kernel with optimized parameters
    BLOCK_SIZE = 128  # Optimize for T4 GPU
    
    _ultra_optimized_nf4_kernel[grid_size,](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        dtype,
        BLOCK_SIZE,
        num_warps=4,  # Optimal for T4
        num_stages=2,  # Pipeline stages
    )
    
    return output

# Export the optimized function
triton_dequantize_nf4 = optimized_v2_triton_dequantize_nf4