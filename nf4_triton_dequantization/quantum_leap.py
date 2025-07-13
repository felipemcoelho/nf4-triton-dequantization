import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _quantum_leap_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantum Leap kernel - absolute minimal overhead."""
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE blocks
    total_blocks = M * blocks_per_row
    
    # NF4 constants
    SCALE: tl.constexpr = 0.00787401574803149606
    
    # Process BLOCK_SIZE blocks per program
    for i in range(BLOCK_SIZE):
        block_id = pid * BLOCK_SIZE + i
        if block_id >= total_blocks:
            return
        
        # Compute row and column
        row = block_id // blocks_per_row
        col_block = block_id % blocks_per_row
        col_start = col_block * 64
        
        if col_start >= N:
            continue
        
        # Load scales with minimal computation
        absmax_val = tl.load(absmax_ptr + block_id)
        absmax32_val = tl.load(absmax32_ptr + row * absmax32_per_row + (col_block >> 2))
        scale = absmax_val * SCALE * absmax32_val
        
        # Base addresses
        qweight_base = row * (N >> 1) + (col_start >> 1)
        output_base = row * N + col_start
        
        # Process 64 elements in 2 chunks of 32 for better cache usage
        for chunk in range(2):
            offset = chunk * 32
            
            # Load 16 packed bytes (32 nibbles)
            packed_idx = qweight_base + (offset >> 1) + tl.arange(0, 16)
            packed = tl.load(qweight_ptr + packed_idx)
            
            # Extract nibbles efficiently
            nibbles_lo = packed & 0xF
            nibbles_hi = (packed >> 4) & 0xF
            
            # Direct lookup using arithmetic for speed
            # Split computation to reduce register pressure
            
            # Process low nibbles
            vals_lo = tl.where(nibbles_lo == 0, -1.0,
                      tl.where(nibbles_lo == 1, -0.6961928009986877,
                      tl.where(nibbles_lo == 2, -0.5250730514526367,
                      tl.where(nibbles_lo == 3, -0.39491748809814453,
                      tl.where(nibbles_lo == 4, -0.28444138169288635,
                      tl.where(nibbles_lo == 5, -0.18477343022823334,
                      tl.where(nibbles_lo == 6, -0.09105003625154495,
                      tl.where(nibbles_lo == 7, 0.0,
                      tl.where(nibbles_lo == 8, 0.07958029955625534,
                      tl.where(nibbles_lo == 9, 0.16093020141124725,
                      tl.where(nibbles_lo == 10, 0.24611230194568634,
                      tl.where(nibbles_lo == 11, 0.33791524171829224,
                      tl.where(nibbles_lo == 12, 0.44070982933044434,
                      tl.where(nibbles_lo == 13, 0.5626170039176941,
                      tl.where(nibbles_lo == 14, 0.7229568362236023,
                      1.0)))))))))))))))
            
            # Process high nibbles
            vals_hi = tl.where(nibbles_hi == 0, -1.0,
                      tl.where(nibbles_hi == 1, -0.6961928009986877,
                      tl.where(nibbles_hi == 2, -0.5250730514526367,
                      tl.where(nibbles_hi == 3, -0.39491748809814453,
                      tl.where(nibbles_hi == 4, -0.28444138169288635,
                      tl.where(nibbles_hi == 5, -0.18477343022823334,
                      tl.where(nibbles_hi == 6, -0.09105003625154495,
                      tl.where(nibbles_hi == 7, 0.0,
                      tl.where(nibbles_hi == 8, 0.07958029955625534,
                      tl.where(nibbles_hi == 9, 0.16093020141124725,
                      tl.where(nibbles_hi == 10, 0.24611230194568634,
                      tl.where(nibbles_hi == 11, 0.33791524171829224,
                      tl.where(nibbles_hi == 12, 0.44070982933044434,
                      tl.where(nibbles_hi == 13, 0.5626170039176941,
                      tl.where(nibbles_hi == 14, 0.7229568362236023,
                      1.0)))))))))))))))
            
            # Apply scale
            out_lo = vals_lo * scale
            out_hi = vals_hi * scale
            
            # Store with optimal pattern
            out_idx = output_base + offset
            
            # Unroll the interleaved store for maximum performance
            for j in range(16):
                idx_lo = out_idx + j * 2
                idx_hi = out_idx + j * 2 + 1
                
                if col_start + offset + j * 2 < N:
                    tl.store(output_ptr + idx_lo, out_lo[j])
                if col_start + offset + j * 2 + 1 < N:
                    tl.store(output_ptr + idx_hi, out_hi[j])

def quantum_leap_dequantize_nf4(module):
    """Quantum Leap NF4 dequantization - ultimate performance."""
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
    
    # Ensure contiguous memory
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Optimal configuration
    total_blocks = M * blocks_per_row
    BLOCK_SIZE = 4  # Each program processes 4 blocks
    grid_size = (total_blocks + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with optimal configuration
    _quantum_leap_kernel[(grid_size,)](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    
    return output

# Export
triton_dequantize_nf4 = quantum_leap_dequantize_nf4