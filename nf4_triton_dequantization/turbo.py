import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _turbo_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
):
    """Turbo NF4 kernel with maximum throughput."""
    pid = tl.program_id(0)
    
    # Process multiple 64-element blocks per thread
    BLOCKS_PER_THREAD = 8  # 512 elements per thread
    total_blocks = (M * N + 63) // 64
    
    start_block = pid * BLOCKS_PER_THREAD
    if start_block >= total_blocks:
        return
    
    # Constants
    SCALE = 0.00787401574803149606
    
    # NF4 lookup table as constants
    NF4_VALUES = tl.constexpr([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Process multiple blocks
    for b in range(BLOCKS_PER_THREAD):
        block_id = start_block + b
        if block_id >= total_blocks:
            break
        
        # Calculate block position
        base_idx = block_id * 64
        row = base_idx // N
        
        if row >= M:
            break
            
        col_start = base_idx - row * N
        
        # Load absmax values once per block
        block_idx = col_start >> 6
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
        
        absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax * SCALE * absmax32
        
        # Process 64 elements in 2 vectorized loads
        for offset in range(0, 64, 32):
            idx = base_idx + offset + tl.arange(0, 32)
            col = col_start + offset + tl.arange(0, 32)
            mask = col < N
            
            # Load packed data
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            
            # Extract nibbles efficiently
            is_odd = idx & 1
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Direct lookup
            nf4 = tl.where(nibbles == 0, -1.0,
                  tl.where(nibbles == 1, -0.6961928009986877,
                  tl.where(nibbles == 2, -0.5250730514526367,
                  tl.where(nibbles == 3, -0.39491748809814453,
                  tl.where(nibbles == 4, -0.28444138169288635,
                  tl.where(nibbles == 5, -0.18477343022823334,
                  tl.where(nibbles == 6, -0.09105003625154495,
                  tl.where(nibbles == 7, 0.0,
                  tl.where(nibbles == 8, 0.07958029955625534,
                  tl.where(nibbles == 9, 0.16093020141124725,
                  tl.where(nibbles == 10, 0.24611230194568634,
                  tl.where(nibbles == 11, 0.33791524171829224,
                  tl.where(nibbles == 12, 0.44070982933044434,
                  tl.where(nibbles == 13, 0.5626170039176941,
                  tl.where(nibbles == 14, 0.7229568362236023,
                  1.0)))))))))))))))
            
            # Apply scaling and store
            output = nf4 * scale
            tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def _turbo_nf4_kernel_v2(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    total_elements,
    N,
    blocks_per_row: tl.constexpr,
):
    """V2: Grid-stride loop for perfect load balancing."""
    pid = tl.program_id(0)
    
    # Grid-stride loop
    ELEMENTS_PER_THREAD = 512
    num_threads = tl.num_programs(0)
    
    # Constants
    SCALE = 0.00787401574803149606
    
    # Process elements with grid-stride pattern
    for elem_start in range(pid * ELEMENTS_PER_THREAD, total_elements, num_threads * ELEMENTS_PER_THREAD):
        # Process ELEMENTS_PER_THREAD elements
        for chunk in range(0, ELEMENTS_PER_THREAD, 64):
            base = elem_start + chunk
            if base >= total_elements:
                break
            
            # Compute position
            row = base // N
            col_start = base % N
            
            # Load absmax values
            block_idx = col_start >> 6
            absmax_idx = row * blocks_per_row + block_idx
            absmax32_idx = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
            
            absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx)
            scale = absmax * SCALE * absmax32
            
            # Process 64 elements
            idx = base + tl.arange(0, 64)
            mask = idx < total_elements
            
            # Load and extract
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            nibbles = tl.where(idx & 1, (packed >> 4) & 0xF, packed & 0xF)
            
            # Direct lookup - unrolled for performance
            nf4 = tl.where(nibbles == 0, -1.0,
                  tl.where(nibbles == 1, -0.6961928009986877,
                  tl.where(nibbles == 2, -0.5250730514526367,
                  tl.where(nibbles == 3, -0.39491748809814453,
                  tl.where(nibbles == 4, -0.28444138169288635,
                  tl.where(nibbles == 5, -0.18477343022823334,
                  tl.where(nibbles == 6, -0.09105003625154495,
                  tl.where(nibbles == 7, 0.0,
                  tl.where(nibbles == 8, 0.07958029955625534,
                  tl.where(nibbles == 9, 0.16093020141124725,
                  tl.where(nibbles == 10, 0.24611230194568634,
                  tl.where(nibbles == 11, 0.33791524171829224,
                  tl.where(nibbles == 12, 0.44070982933044434,
                  tl.where(nibbles == 13, 0.5626170039176941,
                  tl.where(nibbles == 14, 0.7229568362236023,
                  1.0)))))))))))))))
            
            # Store result
            output = nf4 * scale
            tl.store(output_ptr + idx, output, mask=mask)

def turbo_triton_dequantize_nf4(module):
    """Turbo NF4 dequantization for maximum performance."""
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
        return fast_dequantize(weight, quant_state)
    
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    if absmax32.shape != (M, absmax32_per_row):
        return fast_dequantize(weight, quant_state)
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    total_elements = M * N
    
    # Use turbo kernel with optimal grid size
    if total_elements < 131072:  # Small matrices
        # V1 kernel with block processing
        total_blocks = (total_elements + 63) // 64
        BLOCKS_PER_THREAD = 8
        grid = (triton.cdiv(total_blocks, BLOCKS_PER_THREAD),)
        
        _turbo_nf4_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
        )
    else:
        # V2 kernel with grid-stride
        # Optimal grid size for T4: 2-3 blocks per SM
        max_threads = 72  # 36 SMs * 2
        ELEMENTS_PER_THREAD = 512
        num_threads = min(max_threads, triton.cdiv(total_elements, ELEMENTS_PER_THREAD))
        grid = (num_threads,)
        
        _turbo_nf4_kernel_v2[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            total_elements,
            N,
            blocks_per_row,
        )
    
    return output