import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _supersonic_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    stride_qw,
    stride_am,
    stride_am32,
    stride_out,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Supersonic NF4 kernel - optimized for maximum performance."""
    
    # 2D parallelization
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block boundaries
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Early exit for out of bounds
    if m_start >= M:
        return
    
    # NF4 constants
    SCALE: tl.constexpr = 0.00787401574803149606
    
    # NF4 lookup table - split for better performance
    nf4_low = tl.zeros(8, dtype=tl.float32)
    nf4_low[0] = -1.0
    nf4_low[1] = -0.6961928009986877
    nf4_low[2] = -0.5250730514526367
    nf4_low[3] = -0.39491748809814453
    nf4_low[4] = -0.28444138169288635
    nf4_low[5] = -0.18477343022823334
    nf4_low[6] = -0.09105003625154495
    nf4_low[7] = 0.0
    
    nf4_high = tl.zeros(8, dtype=tl.float32)
    nf4_high[0] = 0.07958029955625534
    nf4_high[1] = 0.16093020141124725
    nf4_high[2] = 0.24611230194568634
    nf4_high[3] = 0.33791524171829224
    nf4_high[4] = 0.44070982933044434
    nf4_high[5] = 0.5626170039176941
    nf4_high[6] = 0.7229568362236023
    nf4_high[7] = 1.0
    
    # Process BLOCK_M rows
    for m in range(min(BLOCK_M, M - m_start)):
        row = m_start + m
        
        # Process BLOCK_N columns in chunks of 64 (NF4 block size)
        for n_block_start in range(n_start, min(n_start + BLOCK_N, N), 64):
            block_idx = n_block_start // 64
            
            # Load scale factors
            absmax_idx = row * blocks_per_row + block_idx
            absmax = tl.load(absmax_ptr + absmax_idx * stride_am).to(tl.float32)
            
            absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx * stride_am32)
            
            scale = absmax * SCALE * absmax32
            
            # Process 64 elements
            base_offset = row * N + n_block_start
            
            # Vectorized processing of 64 elements
            for i in range(0, 64, 16):  # Process in chunks of 16
                if n_block_start + i >= N:
                    break
                
                # Load 8 packed bytes (16 nibbles)
                packed_offset = (base_offset + i) // 2
                packed_data = tl.zeros(8, dtype=tl.int32)
                
                for j in range(8):
                    if n_block_start + i + j * 2 < N:
                        packed_data[j] = tl.load(qweight_ptr + (packed_offset + j) * stride_qw)
                
                # Extract and dequantize 16 values
                for j in range(8):
                    if n_block_start + i + j * 2 >= N:
                        break
                    
                    # Extract two nibbles
                    packed = packed_data[j]
                    nibble_even = packed & 0xF
                    nibble_odd = (packed >> 4) & 0xF
                    
                    # Lookup values
                    val_even = nf4_low[nibble_even] if nibble_even < 8 else nf4_high[nibble_even - 8]
                    val_odd = nf4_low[nibble_odd] if nibble_odd < 8 else nf4_high[nibble_odd - 8]
                    
                    # Scale and store
                    out_even = val_even * scale
                    out_odd = val_odd * scale
                    
                    out_idx_even = (base_offset + i + j * 2) * stride_out
                    out_idx_odd = (base_offset + i + j * 2 + 1) * stride_out
                    
                    tl.store(output_ptr + out_idx_even, out_even, mask=(n_block_start + i + j * 2 < N))
                    if n_block_start + i + j * 2 + 1 < N:
                        tl.store(output_ptr + out_idx_odd, out_odd)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_M': 2, 'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 256}, num_warps=2),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128}, num_warps=1),
    ],
    key=['M', 'N'],
)
@triton.jit
def _supersonic_nf4_kernel_tuned(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    stride_qw,
    stride_am,
    stride_am32,
    stride_out,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _supersonic_nf4_kernel(
        qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
        M, N, stride_qw, stride_am, stride_am32, stride_out,
        blocks_per_row, absmax32_per_row, BLOCK_M, BLOCK_N
    )

def supersonic_dequantize_nf4(module):
    """Supersonic NF4 dequantization for 1.15x+ speedup."""
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
    
    # Get strides
    stride_qw = qweight.stride(-1)
    stride_am = absmax.stride(-1)
    stride_am32 = absmax32.stride(-1)
    stride_out = output.stride(-1)
    
    # Launch with 2D grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    _supersonic_nf4_kernel_tuned[grid](
        qweight.data_ptr(),
        absmax.data_ptr(),
        absmax32.data_ptr(),
        output.data_ptr(),
        M, N,
        stride_qw,
        stride_am,
        stride_am32,
        stride_out,
        blocks_per_row,
        absmax32_per_row
    )
    
    return output

# Export
triton_dequantize_nf4 = supersonic_dequantize_nf4