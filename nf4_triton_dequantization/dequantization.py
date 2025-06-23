import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize
from .max_perf import max_perf_triton_dequantize_nf4
from .ultra_optimized import ultra_fast_triton_dequantize_nf4
from .final_optimized import final_optimized_dequantize_nf4

@triton.jit
def _optimized_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Grid dimensions: (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate the block's starting position
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    # Pre-compute NF4 lookup table in registers
    nf4_lut = tl.load(tl.arange(0, 16) * 0 + tl.arange(0, 16).to(tl.int32))
    nf4_lut = tl.where(nf4_lut == 0, -1.0,
              tl.where(nf4_lut == 1, -0.6961928009986877,
              tl.where(nf4_lut == 2, -0.5250730514526367,
              tl.where(nf4_lut == 3, -0.39491748809814453,
              tl.where(nf4_lut == 4, -0.28444138169288635,
              tl.where(nf4_lut == 5, -0.18477343022823334,
              tl.where(nf4_lut == 6, -0.09105003625154495,
              tl.where(nf4_lut == 7, 0.0,
              tl.where(nf4_lut == 8, 0.07958029955625534,
              tl.where(nf4_lut == 9, 0.16093020141124725,
              tl.where(nf4_lut == 10, 0.24611230194568634,
              tl.where(nf4_lut == 11, 0.33791524171829224,
              tl.where(nf4_lut == 12, 0.44070982933044434,
              tl.where(nf4_lut == 13, 0.5626170039176941,
              tl.where(nf4_lut == 14, 0.7229568362236023,
              1.0)))))))))))))))
    
    # Process the block
    for m in range(BLOCK_SIZE_M):
        row = row_start + m
        if row >= M:
            break
            
        # Pre-load absmax32 for this row (secondary dequantization)
        # absmax32 has one value per 256 elements (4 blocks of 64)
        absmax32_base = row * absmax32_per_row
        
        for n_offset in range(0, BLOCK_SIZE_N, 64):  # Process in 64-element chunks
            col = col_start + n_offset
            if col >= N:
                break
                
            # Calculate indices
            block_idx = col >> 6  # col // 64
            absmax_idx = row * blocks_per_row + block_idx
            absmax32_idx = absmax32_base + (block_idx >> 2)  # block_idx // 4
            
            # Load absmax values (fused double dequantization)
            absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx)
            scale = absmax * 0.00787401574803149606 * absmax32  # 1/127 = 0.00787401574803149606
            
            # Process 64 elements in this block
            for i in range(0, 64, 32):  # Process 32 elements at a time
                if col + i >= N:
                    break
                    
                n_vec = col + i + tl.arange(0, 32)
                mask = n_vec < N
                
                # Calculate linear index for packed weight
                linear_idx = row * N + n_vec
                packed_idx = linear_idx >> 1  # // 2
                
                # Load packed weights (16 uint8 values containing 32 nibbles)
                packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
                
                # Extract nibbles efficiently
                # Even indices: low nibbles, odd indices: high nibbles
                is_odd = (linear_idx & 1).to(tl.int32)
                shift = is_odd << 2  # 0 or 4
                nibbles = (packed >> shift) & 0xF
                
                # Lookup NF4 values using the precomputed LUT
                # This is more efficient than the branching approach
                nf4_vals = tl.load(nf4_lut + nibbles, mask=mask, other=0.0)
                
                # Apply scaling and store
                output_idx = row * N + n_vec
                output = nf4_vals * scale
                tl.store(output_ptr + output_idx, output, mask=mask)

def triton_dequantize_nf4(module):
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
    
    # Prepare absmax
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        elif absmax.numel() == M * blocks_per_row:
            absmax = absmax.view(M, blocks_per_row)
    
    if absmax.shape != (M, blocks_per_row):
        return fast_dequantize(weight, quant_state)
    
    # Prepare absmax32
    absmax32_per_row = (blocks_per_row + 3) // 4
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    if absmax32.shape != (M, absmax32_per_row):
        return fast_dequantize(weight, quant_state)
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Use 2D grid for better parallelism and memory access patterns
    BLOCK_SIZE_M = 16  # Process 16 rows at a time
    BLOCK_SIZE_N = 256  # Process 256 columns at a time (4 blocks of 64)
    
    # Adjust block sizes based on matrix dimensions
    if M < BLOCK_SIZE_M:
        BLOCK_SIZE_M = triton.next_power_of_2(M)
    if N < BLOCK_SIZE_N:
        BLOCK_SIZE_N = triton.next_power_of_2(N)
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    _optimized_nf4_kernel[grid](
        qweight.contiguous(),
        absmax.contiguous(),
        absmax32.contiguous(),
        output,
        M, N,
        blocks_per_row,
        absmax32_per_row,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return output

# For backward compatibility
optimized_triton_dequantize_nf4 = max_perf_triton_dequantize_nf4
benchmark_fast_dequantize = max_perf_triton_dequantize_nf4

# Override the main function with the final optimized version
triton_dequantize_nf4 = final_optimized_dequantize_nf4

def reset_triton_dequantize_state():
    pass