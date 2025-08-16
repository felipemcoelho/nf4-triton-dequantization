import torch
import triton
import triton.language as tl

# NF4 lookup table as constants for better performance
NF4_TABLE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]

@triton.jit
def _optimized_nf4_dequantize_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    stride_qw_m, stride_qw_n,
    stride_am_m, stride_am_n,
    stride_a32_m, stride_a32_n,
    stride_o_m, stride_o_n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Highly optimized NF4 dequantization kernel with vectorized operations."""
    
    # Grid indexing for better parallelism
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    # Calculate row and block indices
    row = pid // blocks_per_row
    block_in_row = pid % blocks_per_row
    
    if row >= M:
        return
    
    # Calculate column range for this block (64 elements per block)
    col_start = block_in_row * 64
    if col_start >= N:
        return
    
    # Load scale factors (double dequantization)
    absmax_idx = row * stride_am_m + block_in_row
    absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
    
    absmax32_block = block_in_row >> 2  # block_in_row // 4
    absmax32_idx = row * stride_a32_m + absmax32_block
    absmax32 = tl.load(absmax32_ptr + absmax32_idx).to(tl.float32)
    
    # Combined scale factor (NF4 quantization constant)
    scale = absmax * 0.00787401574803149606 * absmax32
    
    # Calculate base pointers
    qweight_row_base = row * stride_qw_m
    output_row_base = row * stride_o_m
    
    # Vectorized load of 32 packed bytes (64 nibbles)
    # Each byte contains 2 4-bit values
    packed_offset = qweight_row_base + (col_start >> 1)
    
    # Load 32 bytes at once using vectorized load
    packed_data = tl.load(qweight_ptr + packed_offset + tl.arange(0, 32))
    
    # Extract low and high nibbles efficiently
    low_nibbles = packed_data & 0xF
    high_nibbles = (packed_data >> 4) & 0xF
    
    # Optimized NF4 lookup using polynomial approximation for groups
    # This reduces the number of conditional branches significantly
    
    # For low nibbles - using arithmetic operations to reduce branches
    low_vals = tl.where(low_nibbles == 0, -1.0,
               tl.where(low_nibbles == 1, -0.6961928009986877,
               tl.where(low_nibbles == 2, -0.5250730514526367,
               tl.where(low_nibbles == 3, -0.39491748809814453,
               tl.where(low_nibbles == 4, -0.28444138169288635,
               tl.where(low_nibbles == 5, -0.18477343022823334,
               tl.where(low_nibbles == 6, -0.09105003625154495,
               tl.where(low_nibbles == 7, 0.0,
               tl.where(low_nibbles == 8, 0.07958029955625534,
               tl.where(low_nibbles == 9, 0.16093020141124725,
               tl.where(low_nibbles == 10, 0.24611230194568634,
               tl.where(low_nibbles == 11, 0.33791524171829224,
               tl.where(low_nibbles == 12, 0.44070982933044434,
               tl.where(low_nibbles == 13, 0.5626170039176941,
               tl.where(low_nibbles == 14, 0.7229568362236023, 1.0)))))))))))))))
    
    # For high nibbles - same lookup
    high_vals = tl.where(high_nibbles == 0, -1.0,
                tl.where(high_nibbles == 1, -0.6961928009986877,
                tl.where(high_nibbles == 2, -0.5250730514526367,
                tl.where(high_nibbles == 3, -0.39491748809814453,
                tl.where(high_nibbles == 4, -0.28444138169288635,
                tl.where(high_nibbles == 5, -0.18477343022823334,
                tl.where(high_nibbles == 6, -0.09105003625154495,
                tl.where(high_nibbles == 7, 0.0,
                tl.where(high_nibbles == 8, 0.07958029955625534,
                tl.where(high_nibbles == 9, 0.16093020141124725,
                tl.where(high_nibbles == 10, 0.24611230194568634,
                tl.where(high_nibbles == 11, 0.33791524171829224,
                tl.where(high_nibbles == 12, 0.44070982933044434,
                tl.where(high_nibbles == 13, 0.5626170039176941,
                tl.where(high_nibbles == 14, 0.7229568362236023, 1.0)))))))))))))))
    
    # Apply scale
    low_scaled = low_vals * scale
    high_scaled = high_vals * scale
    
    # Vectorized store with coalesced memory access
    output_base = output_row_base + col_start
    
    # Store using vectorized operations - interleave low and high values
    # Process in chunks for better memory coalescing
    for i in tl.static_range(16):  # Process 16 pairs at a time
        idx = i * 2
        # Store two consecutive values (low, high from same byte)
        if col_start + idx < N:
            tl.store(output_ptr + output_base + idx, low_scaled[i])
        if col_start + idx + 1 < N:
            tl.store(output_ptr + output_base + idx + 1, high_scaled[i])
    
    # Process remaining 16 pairs
    for i in tl.static_range(16, 32):
        idx = i * 2
        if col_start + idx < N:
            tl.store(output_ptr + output_base + idx, low_scaled[i])
        if col_start + idx + 1 < N:
            tl.store(output_ptr + output_base + idx + 1, high_scaled[i])


@triton.jit
def _ultra_optimized_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Ultra-optimized kernel with better parallelism and memory access."""
    
    # 2D grid for better GPU utilization
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate the range of rows and columns this program handles
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    
    if row_start >= M or col_start >= N:
        return
    
    # Process multiple rows in parallel
    for row_offset in range(min(BLOCK_M, M - row_start)):
        row = row_start + row_offset
        
        # Process columns in 64-element chunks (NF4 block size)
        for col_block_start in range(col_start, min(col_start + BLOCK_N, N), 64):
            block_idx = col_block_start // 64
            
            # Load scale factors
            absmax_idx = row * blocks_per_row + block_idx
            absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            
            absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx).to(tl.float32)
            
            # Combined scale
            scale = absmax * 0.00787401574803149606 * absmax32
            
            # Calculate offsets
            qweight_offset = row * (N >> 1) + (col_block_start >> 1)
            output_offset = row * N + col_block_start
            
            # Vectorized load of 32 bytes
            packed = tl.load(qweight_ptr + qweight_offset + tl.arange(0, 32))
            
            # Extract nibbles
            low = packed & 0xF
            high = (packed >> 4) & 0xF
            
            # Optimized lookup with reduced branches
            # Group similar values to reduce comparisons
            low_vals = tl.where(low == 7, 0.0,
                       tl.where(low == 0, -1.0,
                       tl.where(low == 15, 1.0,
                       tl.where(low < 7,
                           # Negative values - use select to reduce branches
                           tl.where(low == 1, -0.6961928009986877,
                           tl.where(low == 2, -0.5250730514526367,
                           tl.where(low == 3, -0.39491748809814453,
                           tl.where(low == 4, -0.28444138169288635,
                           tl.where(low == 5, -0.18477343022823334,
                           -0.09105003625154495))))),
                           # Positive values
                           tl.where(low == 8, 0.07958029955625534,
                           tl.where(low == 9, 0.16093020141124725,
                           tl.where(low == 10, 0.24611230194568634,
                           tl.where(low == 11, 0.33791524171829224,
                           tl.where(low == 12, 0.44070982933044434,
                           tl.where(low == 13, 0.5626170039176941,
                           0.7229568362236023))))))
                       ))))
            
            high_vals = tl.where(high == 7, 0.0,
                        tl.where(high == 0, -1.0,
                        tl.where(high == 15, 1.0,
                        tl.where(high < 7,
                            tl.where(high == 1, -0.6961928009986877,
                            tl.where(high == 2, -0.5250730514526367,
                            tl.where(high == 3, -0.39491748809814453,
                            tl.where(high == 4, -0.28444138169288635,
                            tl.where(high == 5, -0.18477343022823334,
                            -0.09105003625154495))))),
                            tl.where(high == 8, 0.07958029955625534,
                            tl.where(high == 9, 0.16093020141124725,
                            tl.where(high == 10, 0.24611230194568634,
                            tl.where(high == 11, 0.33791524171829224,
                            tl.where(high == 12, 0.44070982933044434,
                            tl.where(high == 13, 0.5626170039176941,
                            0.7229568362236023))))))
                        ))))
            
            # Apply scale
            low_scaled = low_vals * scale
            high_scaled = high_vals * scale
            
            # Vectorized store with loop unrolling
            for i in tl.static_range(8):  # Unroll in groups of 8
                base_idx = i * 4
                for j in range(4):
                    idx = base_idx + j
                    out_idx = idx * 2
                    if col_block_start + out_idx < N:
                        tl.store(output_ptr + output_offset + out_idx, low_scaled[idx])
                    if col_block_start + out_idx + 1 < N:
                        tl.store(output_ptr + output_offset + out_idx + 1, high_scaled[idx])


def optimized_triton_dequantize_nf4(module):
    """Optimized NF4 dequantization with improved performance."""
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
    
    # Handle tensor shapes
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        elif absmax.numel() == M * blocks_per_row:
            absmax = absmax.view(M, blocks_per_row)
    
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    # Ensure contiguous memory layout
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Choose kernel configuration based on matrix size
    if M * N < 4096 * 4096:  # Small to medium matrices
        # Use 1D grid with optimized kernel
        total_blocks = M * blocks_per_row
        
        # Get strides for proper indexing
        stride_qw_m = qweight.stride(0) if qweight.dim() > 1 else N // 2
        stride_qw_n = 1
        stride_am_m = absmax.stride(0) if absmax.dim() > 1 else blocks_per_row
        stride_am_n = 1
        stride_a32_m = absmax32.stride(0) if absmax32.dim() > 1 else absmax32_per_row
        stride_a32_n = 1
        stride_o_m = N
        stride_o_n = 1
        
        _optimized_nf4_dequantize_kernel[(total_blocks,)](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            stride_qw_m, stride_qw_n,
            stride_am_m, stride_am_n,
            stride_a32_m, stride_a32_n,
            stride_o_m, stride_o_n,
            blocks_per_row,
            absmax32_per_row,
            BLOCK_SIZE=64,
            num_warps=2,
            num_stages=2,
        )
    else:  # Large matrices
        # Use 2D grid with ultra-optimized kernel
        BLOCK_M = 2
        BLOCK_N = 256
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        _ultra_optimized_nf4_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            BLOCK_M, BLOCK_N,
            num_warps=4,
            num_stages=3,
        )
    
    return output