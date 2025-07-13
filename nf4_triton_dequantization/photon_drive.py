import torch
import triton
import triton.language as tl

try:
    from unsloth.kernels.utils import fast_dequantize
except ImportError:
    fast_dequantize = None

@triton.jit
def _photon_drive_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    n_elements,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Photon Drive kernel - matches Unsloth's approach."""
    
    # Get position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Grid-stride loop
    num_programs = tl.num_programs(0)
    
    # Constants
    SCALE: tl.constexpr = 0.00787401574803149606
    
    # NF4 lookup table as constant array
    nf4_lut = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Process elements in grid-stride fashion
    for idx in range(block_start, n_elements, num_programs * BLOCK_SIZE):
        if idx >= n_elements:
            return
        
        # Compute position
        row = idx // N
        col = idx % N
        
        # Which 64-element block does this belong to?
        block_idx = col // 64
        block_offset = col % 64
        
        # Compute scale factor indices
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        
        # Load scales
        absmax_val = tl.load(absmax_ptr + absmax_idx)
        absmax32_val = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax_val * SCALE * absmax32_val
        
        # Load packed byte
        packed_idx = idx >> 1
        packed = tl.load(qweight_ptr + packed_idx)
        
        # Extract nibble
        is_odd = idx & 1
        nibble = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
        
        # Lookup value
        nf4_val = tl.load(nf4_lut + nibble)
        
        # Apply scale and store
        output_val = nf4_val * scale
        tl.store(output_ptr + idx, output_val)

@triton.jit
def _photon_drive_vectorized_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
):
    """Vectorized Photon Drive kernel."""
    
    # 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Constants
    SCALE: tl.constexpr = 0.00787401574803149606
    BLOCK_N: tl.constexpr = 128
    
    # Boundaries
    row = pid_m
    col_start = pid_n * BLOCK_N
    
    if row >= M or col_start >= N:
        return
    
    # Process BLOCK_N elements
    for block_start in range(col_start, min(col_start + BLOCK_N, N), 64):
        block_idx = block_start // 64
        
        # Load scales once per 64-element block
        absmax_idx = row * blocks_per_row + block_idx
        absmax_val = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
        
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        absmax32_val = tl.load(absmax32_ptr + absmax32_idx)
        
        scale = absmax_val * SCALE * absmax32_val
        
        # Process 64 elements
        base_idx = row * N + block_start
        qweight_base = base_idx >> 1
        
        # Load all 32 packed bytes
        packed = tl.load(qweight_ptr + qweight_base + tl.arange(0, 32))
        
        # Extract nibbles
        nibbles_even = packed & 0xF
        nibbles_odd = (packed >> 4) & 0xF
        
        # Lookup values efficiently using nested conditionals
        vals_even = tl.where(nibbles_even == 0, -1.0,
                    tl.where(nibbles_even == 1, -0.6961928009986877,
                    tl.where(nibbles_even == 2, -0.5250730514526367,
                    tl.where(nibbles_even == 3, -0.39491748809814453,
                    tl.where(nibbles_even == 4, -0.28444138169288635,
                    tl.where(nibbles_even == 5, -0.18477343022823334,
                    tl.where(nibbles_even == 6, -0.09105003625154495,
                    tl.where(nibbles_even == 7, 0.0,
                    tl.where(nibbles_even == 8, 0.07958029955625534,
                    tl.where(nibbles_even == 9, 0.16093020141124725,
                    tl.where(nibbles_even == 10, 0.24611230194568634,
                    tl.where(nibbles_even == 11, 0.33791524171829224,
                    tl.where(nibbles_even == 12, 0.44070982933044434,
                    tl.where(nibbles_even == 13, 0.5626170039176941,
                    tl.where(nibbles_even == 14, 0.7229568362236023,
                    1.0)))))))))))))))
        
        vals_odd = tl.where(nibbles_odd == 0, -1.0,
                   tl.where(nibbles_odd == 1, -0.6961928009986877,
                   tl.where(nibbles_odd == 2, -0.5250730514526367,
                   tl.where(nibbles_odd == 3, -0.39491748809814453,
                   tl.where(nibbles_odd == 4, -0.28444138169288635,
                   tl.where(nibbles_odd == 5, -0.18477343022823334,
                   tl.where(nibbles_odd == 6, -0.09105003625154495,
                   tl.where(nibbles_odd == 7, 0.0,
                   tl.where(nibbles_odd == 8, 0.07958029955625534,
                   tl.where(nibbles_odd == 9, 0.16093020141124725,
                   tl.where(nibbles_odd == 10, 0.24611230194568634,
                   tl.where(nibbles_odd == 11, 0.33791524171829224,
                   tl.where(nibbles_odd == 12, 0.44070982933044434,
                   tl.where(nibbles_odd == 13, 0.5626170039176941,
                   tl.where(nibbles_odd == 14, 0.7229568362236023,
                   1.0)))))))))))))))
        
        # Scale values
        out_even = vals_even * scale
        out_odd = vals_odd * scale
        
        # Store results
        max_idx = min(32, (N - block_start + 1) // 2)
        
        for i in range(max_idx):
            if block_start + i * 2 < N:
                tl.store(output_ptr + base_idx + i * 2, out_even[i])
            if block_start + i * 2 + 1 < N:
                tl.store(output_ptr + base_idx + i * 2 + 1, out_odd[i])

def photon_drive_dequantize_nf4(module):
    """Photon Drive NF4 dequantization."""
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
    
    # Use vectorized kernel with 2D grid
    BLOCK_N = 128
    grid = (M, triton.cdiv(N, BLOCK_N))
    
    _photon_drive_vectorized_kernel[grid](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        num_warps=2,
        num_stages=2,
    )
    
    return output

# Export
triton_dequantize_nf4 = photon_drive_dequantize_nf4