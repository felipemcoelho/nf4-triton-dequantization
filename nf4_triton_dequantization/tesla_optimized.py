import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

# Tesla T4 specific optimizations
@triton.jit
def _tesla_t4_nf4_kernel(
    qweight_ptr,
    absmax_ptr, 
    absmax32_ptr,
    output_ptr,
    M, N,
    stride_qw_m, stride_qw_n,
    stride_am_m, stride_am_n,
    stride_am32_m, stride_am32_n,
    stride_o_m, stride_o_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tesla T4 optimized kernel with explicit memory management."""
    pid = tl.program_id(0)
    
    # 2D blocking for better cache utilization on T4
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Skip if out of bounds
    if pid_m >= tl.cdiv(M, BLOCK_M):
        return
    
    # Block pointers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Masks
    mask_m = rm < M
    mask_n = rn < N
    
    # NF4 lookup constants loaded into registers
    nf4_lut = tl.inline_const_exprs([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    scale_factor = 0.00787401574803149606  # 1/127
    
    # Process each output block
    for m in range(BLOCK_M):
        if not mask_m[m]:
            continue
            
        row = rm[m]
        
        # Process in chunks of 64 (absmax block size)
        for n_start in range(0, BLOCK_N, 64):
            # Column indices for this chunk
            cols = rn[n_start:n_start+64]
            col_mask = mask_n[n_start:n_start+64] & (cols < N)
            
            if not tl.sum(col_mask.to(tl.int32)) > 0:
                continue
            
            # Calculate block indices
            block_idx = cols[0] // 64
            absmax_idx = row * tl.cdiv(N, 64) + block_idx
            absmax32_idx = row * tl.cdiv(tl.cdiv(N, 64), 4) + (block_idx // 4)
            
            # Load scaling factors once per 64-element block
            absmax_val = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            absmax32_val = tl.load(absmax32_ptr + absmax32_idx)
            combined_scale = absmax_val * scale_factor * absmax32_val
            
            # Process 64 elements in vectorized fashion
            for i in range(0, 64, 32):
                n_vec = cols[i:i+32]
                vec_mask = col_mask[i:i+32]
                
                if not tl.sum(vec_mask.to(tl.int32)) > 0:
                    continue
                
                # Linear indices
                linear_idx = row * N + n_vec
                packed_idx = linear_idx >> 1
                
                # Load packed data
                packed = tl.load(qweight_ptr + packed_idx, mask=vec_mask, other=0)
                
                # Extract nibbles efficiently
                is_odd = (linear_idx & 1).to(tl.int32)
                nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
                
                # Vectorized lookup using gather
                nf4_vals = tl.gather(nf4_lut, nibbles, mask=vec_mask, other=0.0)
                
                # Apply scaling and store
                output_vals = nf4_vals * combined_scale
                output_idx = row * N + n_vec
                tl.store(output_ptr + output_idx, output_vals, mask=vec_mask)

@triton.jit 
def _tesla_t4_nf4_kernel_v2(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr, 
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """V2: Optimized for memory bandwidth with prefetching."""
    pid = tl.program_id(0)
    
    # Work distribution
    elements_per_thread = BLOCK_SIZE
    start_idx = pid * elements_per_thread
    
    if start_idx >= M * N:
        return
    
    # NF4 values as constants
    NF4_0 = -1.0
    NF4_1 = -0.6961928009986877
    NF4_2 = -0.5250730514526367
    NF4_3 = -0.39491748809814453
    NF4_4 = -0.28444138169288635
    NF4_5 = -0.18477343022823334
    NF4_6 = -0.09105003625154495
    NF4_7 = 0.0
    NF4_8 = 0.07958029955625534
    NF4_9 = 0.16093020141124725
    NF4_10 = 0.24611230194568634
    NF4_11 = 0.33791524171829224
    NF4_12 = 0.44070982933044434
    NF4_13 = 0.5626170039176941
    NF4_14 = 0.7229568362236023
    NF4_15 = 1.0
    
    SCALE = 0.00787401574803149606
    
    # Process in chunks optimized for T4's memory hierarchy
    for chunk_start in range(0, elements_per_thread, 128):
        # Calculate indices
        idx = start_idx + chunk_start + tl.arange(0, 128)
        mask = idx < (M * N)
        
        # Convert to 2D coords
        row = idx // N
        col = idx % N
        
        # Skip invalid rows
        valid_mask = mask & (row < M)
        
        if not tl.sum(valid_mask.to(tl.int32)) > 0:
            continue
        
        # Calculate scaling indices
        block_idx = col >> 6
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
        
        # Prefetch scaling factors
        absmax_vals = tl.load(absmax_ptr + absmax_idx, mask=valid_mask, other=0).to(tl.float32)
        absmax32_vals = tl.load(absmax32_ptr + absmax32_idx, mask=valid_mask, other=1.0)
        scales = absmax_vals * SCALE * absmax32_vals
        
        # Load packed weights
        packed_idx = idx >> 1
        packed = tl.load(qweight_ptr + packed_idx, mask=valid_mask, other=0)
        
        # Extract nibbles with optimized bit ops
        shift = ((idx & 1) << 2).to(tl.int32)
        nibbles = (packed >> shift) & 0xF
        
        # Optimized lookup with reduced branches
        # Split into groups for better instruction scheduling
        is_neg = nibbles < 8
        is_zero = nibbles == 7
        
        # Compute negative values
        neg_vals = tl.where(nibbles == 0, NF4_0,
                   tl.where(nibbles == 1, NF4_1,
                   tl.where(nibbles == 2, NF4_2,
                   tl.where(nibbles == 3, NF4_3,
                   tl.where(nibbles == 4, NF4_4,
                   tl.where(nibbles == 5, NF4_5,
                   tl.where(nibbles == 6, NF4_6, 0.0)))))))
        
        # Compute positive values
        pos_vals = tl.where(nibbles == 8, NF4_8,
                   tl.where(nibbles == 9, NF4_9,
                   tl.where(nibbles == 10, NF4_10,
                   tl.where(nibbles == 11, NF4_11,
                   tl.where(nibbles == 12, NF4_12,
                   tl.where(nibbles == 13, NF4_13,
                   tl.where(nibbles == 14, NF4_14,
                   NF4_15)))))))
        
        # Combine results
        nf4_vals = tl.where(is_zero, 0.0, tl.where(is_neg, neg_vals, pos_vals))
        
        # Apply scaling and store
        output = nf4_vals * scales
        tl.store(output_ptr + idx, output, mask=valid_mask)

def tesla_t4_optimized_dequantize_nf4(module):
    """Tesla T4 optimized NF4 dequantization targeting 1.15x+ speedup."""
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
    
    # Tensor preparation
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
    
    # Tesla T4 has 40 SMs, optimize grid accordingly
    total_elements = M * N
    
    if M <= 64 and N % 256 == 0:
        # Small M, well-aligned N - use 2D blocking
        BLOCK_M = min(16, M)
        BLOCK_N = 256
        
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        _tesla_t4_nf4_kernel[grid](
            qweight.contiguous(),
            absmax.contiguous(),
            absmax32.contiguous(),
            output,
            M, N,
            qweight.stride(0), qweight.stride(1),
            absmax.stride(0), absmax.stride(1),
            absmax32.stride(0), absmax32.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M, BLOCK_N,
        )
    else:
        # General case - optimize for memory bandwidth
        # Tesla T4: 320 GB/s memory bandwidth
        # Optimize block size for bandwidth utilization
        BLOCK_SIZE = 2048  # Tuned for T4
        
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        
        _tesla_t4_nf4_kernel_v2[grid](
            qweight.contiguous().view(-1),
            absmax.contiguous().view(-1), 
            absmax32.contiguous().view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            BLOCK_SIZE,
        )
    
    return output