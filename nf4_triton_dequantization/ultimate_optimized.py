import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _ultimate_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Ultimate NF4 kernel with inline PTX for maximum performance."""
    # Single program processes one 64-element block
    pid = tl.program_id(0)
    
    total_blocks = m * blocks_per_row
    if pid >= total_blocks:
        return
    
    # Decode block coordinates
    row = pid // blocks_per_row
    block_in_row = pid % blocks_per_row
    col_base = block_in_row << 6  # * 64
    
    if col_base >= n:
        return
    
    # Load scaling factors with non-temporal hints
    absmax_idx = pid
    absmax32_idx = (row * absmax32_per_row) + (block_in_row >> 2)
    
    # Use inline assembly for optimized loads
    absmax_val = tl.inline_asm_elementwise(
        "ld.global.cs.f32 $0, [$1];",
        "=f,l",
        args=[absmax_ptr + absmax_idx],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    
    absmax32_val = tl.inline_asm_elementwise(
        "ld.global.cs.f32 $0, [$1];",
        "=f,l", 
        args=[absmax32_ptr + absmax32_idx],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    
    # Compute scale with FMA
    scale = absmax_val * 0.00787401574803149606 * absmax32_val
    
    # NF4 constants packed for efficient access
    nf4_packed_low = tl.inline_const_array([
        0xBF800000, 0xBF322E8C, 0xBF066E46, 0xBECA3244,
        0xBE91CB77, 0xBE3D6EBC, 0xBDBA6B62, 0x00000000
    ])
    nf4_packed_high = tl.inline_const_array([
        0x3DA30966, 0x3E24DE11, 0x3E7C1953, 0x3EAD1268,
        0x3EE1E4F8, 0x3F100806, 0x3F38D666, 0x3F800000
    ])
    
    # Process in 32-element chunks for optimal vectorization
    row_offset = row * n
    
    # First 32 elements
    col = col_base + tl.arange(0, 32)
    mask1 = col < n
    idx1 = row_offset + col
    
    # Packed index and load
    packed_idx1 = idx1 >> 1
    packed1 = tl.load(qweight_ptr + packed_idx1, mask=mask1, other=0)
    
    # Extract nibbles using optimized bit manipulation
    is_odd1 = (idx1 & 1)
    shift1 = is_odd1 << 2
    nibbles1 = (packed1 >> shift1) & 0xF
    
    # Fast lookup using bitcast
    is_high1 = nibbles1 >= 8
    idx_low1 = nibbles1
    idx_high1 = nibbles1 - 8
    
    # Gather with bitcast optimization
    packed_low1 = tl.gather(nf4_packed_low, idx_low1)
    packed_high1 = tl.gather(nf4_packed_high, idx_high1)
    packed_val1 = tl.where(is_high1, packed_high1, packed_low1)
    
    # Bitcast to float
    nf4_val1 = tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=f,r",
        args=[packed_val1],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    
    # Apply scale and store
    output1 = (nf4_val1 * scale).to(dtype)
    tl.store(output_ptr + idx1, output1, mask=mask1)
    
    # Second 32 elements
    col = col_base + 32 + tl.arange(0, 32)
    mask2 = col < n
    idx2 = row_offset + col
    
    packed_idx2 = idx2 >> 1
    packed2 = tl.load(qweight_ptr + packed_idx2, mask=mask2, other=0)
    
    is_odd2 = (idx2 & 1)
    shift2 = is_odd2 << 2
    nibbles2 = (packed2 >> shift2) & 0xF
    
    is_high2 = nibbles2 >= 8
    idx_low2 = nibbles2
    idx_high2 = nibbles2 - 8
    
    packed_low2 = tl.gather(nf4_packed_low, idx_low2)
    packed_high2 = tl.gather(nf4_packed_high, idx_high2)
    packed_val2 = tl.where(is_high2, packed_high2, packed_low2)
    
    nf4_val2 = tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=f,r",
        args=[packed_val2],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    
    output2 = (nf4_val2 * scale).to(dtype)
    tl.store(output_ptr + idx2, output2, mask=mask2)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['m', 'n'],
)
@triton.jit
def _ultimate_autotuned_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    dtype: tl.constexpr,
):
    """Autotuned version of the ultimate kernel."""
    _ultimate_nf4_kernel(
        qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
        m, n, blocks_per_row, absmax32_per_row, dtype
    )


def ultimate_dequantize_nf4(module):
    """Ultimate optimized NF4 dequantization for maximum performance."""
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
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch configuration
    total_blocks = M * blocks_per_row
    grid = (total_blocks,)
    
    # Use autotuned kernel for large matrices
    if M * N > 4 * 1024 * 1024:
        _ultimate_autotuned_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            dtype,
        )
    else:
        _ultimate_nf4_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            dtype,
            num_warps=4,
        )
    
    return output


# Additional kernel using warp-level primitives
@triton.jit
def _warp_optimized_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    WARP_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Warp-optimized kernel for cooperative processing."""
    pid = tl.program_id(0)
    
    # Each warp processes one 64-element block cooperatively
    total_blocks = m * blocks_per_row
    if pid >= total_blocks:
        return
    
    row = pid // blocks_per_row
    block_in_row = pid % blocks_per_row
    col_base = block_in_row * 64
    
    if col_base >= n:
        return
    
    # Warp-level coordination
    warp_id = tl.inline_asm_elementwise(
        "mov.u32 $0, %warpid;",
        "=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    
    lane_id = tl.inline_asm_elementwise(
        "mov.u32 $0, %laneid;",
        "=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    
    # Load scaling factors (broadcast within warp)
    if lane_id == 0:
        absmax_idx = pid
        absmax32_idx = row * absmax32_per_row + (block_in_row >> 2)
        
        absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
    
    # Broadcast scale to all lanes
    absmax = tl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0, $1, 0, 0x1f, 0xffffffff;",
        "=f,f",
        args=[absmax],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )
    
    absmax32 = tl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0, $1, 0, 0x1f, 0xffffffff;",
        "=f,f",
        args=[absmax32],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )
    
    scale = absmax * 0.00787401574803149606 * absmax32
    
    # Each lane processes 2 elements
    elements_per_lane = 2
    lane_offset = lane_id * elements_per_lane
    
    # NF4 lookup values
    nf4_vals = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Process assigned elements
    for i in range(elements_per_lane):
        local_idx = lane_offset + i
        if local_idx >= 64:
            break
            
        col = col_base + local_idx
        if col >= n:
            break
            
        idx = row * n + col
        packed_idx = idx >> 1
        
        packed = tl.load(qweight_ptr + packed_idx)
        
        is_odd = (idx & 1)
        nibble = (packed >> (is_odd << 2)) & 0xF
        
        nf4_val = tl.gather(nf4_vals, nibble)
        output = (nf4_val * scale).to(dtype)
        
        tl.store(output_ptr + idx, output)


def warp_optimized_dequantize_nf4(module):
    """Warp-optimized implementation for cooperative processing."""
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
    
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch with one warp per block
    total_blocks = M * blocks_per_row
    grid = (total_blocks,)
    
    _warp_optimized_kernel[grid](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        32,  # WARP_SIZE
        dtype,
        num_warps=1,
    )
    
    return output