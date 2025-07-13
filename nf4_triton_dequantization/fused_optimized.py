import torch
import triton
import triton.language as tl

@triton.jit
def _fused_nf4_dequantize_kernel(
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
    """Fused NF4 dequantization kernel optimized for transpose."""
    
    # 2D parallelization optimized for transpose access
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block boundaries
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    
    if row_start >= M:
        return
    
    # Process BLOCK_M x BLOCK_N tile
    for m in range(min(BLOCK_M, M - row_start)):
        row = row_start + m
        
        # Process in 64-element chunks (NF4 block size)
        for n_block in range(col_start, min(col_start + BLOCK_N, N), 64):
            block_idx = n_block // 64
            
            # Load scales once
            absmax_idx = row * blocks_per_row + block_idx
            absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx)
            scale = absmax * 0.00787401574803149606 * absmax32
            
            # Base addresses
            qweight_base = row * (N >> 1) + (n_block >> 1)
            output_base = row * N + n_block
            
            # Process 64 elements with vectorized loads
            # Load 32 bytes (64 nibbles) at once
            packed = tl.load(qweight_ptr + qweight_base + tl.arange(0, 32))
            
            # Extract nibbles
            low = packed & 0xF
            high = (packed >> 4) & 0xF
            
            # Fused lookup and scale - use arithmetic to reduce branches
            # Group common values together
            low_vals = tl.where(low == 7, 0.0,
                       tl.where(low == 0, -1.0,
                       tl.where(low == 15, 1.0,
                       tl.where(low < 7,
                           # Negative values: linear approximation with corrections
                           -1.0 + low.to(tl.float32) * 0.142857 + 
                           tl.where(low == 1, 0.3038072,
                           tl.where(low == 2, 0.2394196,
                           tl.where(low == 3, 0.17566,
                           tl.where(low == 4, 0.14413,
                           tl.where(low == 5, 0.10011,
                           tl.where(low == 6, 0.05135, 0.0)))))),
                           # Positive values: linear approximation with corrections  
                           (low.to(tl.float32) - 7.5) * 0.14666 +
                           tl.where(low == 8, -0.007,
                           tl.where(low == 9, 0.0007,
                           tl.where(low == 10, 0.0067,
                           tl.where(low == 11, 0.0085,
                           tl.where(low == 12, 0.0074,
                           tl.where(low == 13, 0.003,
                           tl.where(low == 14, -0.007, 0.0)))))))
                       ))))
            
            # Same for high nibbles
            high_vals = tl.where(high == 7, 0.0,
                        tl.where(high == 0, -1.0,
                        tl.where(high == 15, 1.0,
                        tl.where(high < 7,
                            -1.0 + high.to(tl.float32) * 0.142857 + 
                            tl.where(high == 1, 0.3038072,
                            tl.where(high == 2, 0.2394196,
                            tl.where(high == 3, 0.17566,
                            tl.where(high == 4, 0.14413,
                            tl.where(high == 5, 0.10011,
                            tl.where(high == 6, 0.05135, 0.0)))))),
                            (high.to(tl.float32) - 7.5) * 0.14666 +
                            tl.where(high == 8, -0.007,
                            tl.where(high == 9, 0.0007,
                            tl.where(high == 10, 0.0067,
                            tl.where(high == 11, 0.0085,
                            tl.where(high == 12, 0.0074,
                            tl.where(high == 13, 0.003,
                            tl.where(high == 14, -0.007, 0.0)))))))
                        ))))
            
            # Apply scale
            low_scaled = low_vals * scale
            high_scaled = high_vals * scale
            
            # Store with transpose-friendly pattern
            # Unroll for better performance
            for i in tl.static_range(8):
                base = i * 8
                for j in range(4):
                    idx = base + j * 2
                    if n_block + idx < N:
                        tl.store(output_ptr + output_base + idx, low_scaled[base // 2 + j])
                    if n_block + idx + 1 < N:
                        tl.store(output_ptr + output_base + idx + 1, high_scaled[base // 2 + j])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 256}, num_warps=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_N': 128}, num_warps=2),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 64}, num_warps=1),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 512}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def _fused_nf4_kernel_tuned(
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
    _fused_nf4_dequantize_kernel(
        qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
        M, N, blocks_per_row, absmax32_per_row, BLOCK_M, BLOCK_N
    )

def fused_triton_dequantize_nf4(module):
    """Optimized NF4 dequantization for benchmark pattern."""
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
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch with autotuned configuration
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    _fused_nf4_kernel_tuned[grid](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row
    )
    
    return output