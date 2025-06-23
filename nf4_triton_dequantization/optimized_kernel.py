import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _optimized_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Optimized NF4 dequantization kernel with vectorized operations."""
    # Grid: (num_row_blocks, num_col_blocks)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Each block processes BLOCK_SIZE elements
    row = pid_m
    col_block = pid_n
    
    if row >= m:
        return
    
    # Base indices for this block
    col_start = col_block * BLOCK_SIZE
    if col_start >= n:
        return
    
    # Load absmax values for this block
    # Each 64-element block has its own absmax
    block_idx = col_start >> 6  # col_start // 64
    absmax_idx = row * blocks_per_row + block_idx
    absmax32_idx = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
    
    # Load and compute scale
    absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
    absmax32 = tl.load(absmax32_ptr + absmax32_idx)
    scale = absmax * 0.00787401574803149606 * absmax32
    
    # Process BLOCK_SIZE elements with vectorization
    # Use multiple smaller vectors for better performance
    VEC_SIZE: tl.constexpr = 32
    num_vecs = BLOCK_SIZE // VEC_SIZE
    
    for vec_idx in range(num_vecs):
        # Calculate indices for this vector
        vec_offset = vec_idx * VEC_SIZE
        col = col_start + vec_offset + tl.arange(0, VEC_SIZE)
        idx = row * n + col
        
        # Create mask for boundary checking
        mask = col < n
        
        # Load packed data (2 nibbles per byte)
        packed_idx = idx >> 1
        packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
        
        # Extract nibbles efficiently
        is_odd = (idx & 1).to(tl.int1)
        nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
        
        # Optimized NF4 lookup using a more efficient approach
        # Split into groups to reduce nested conditions
        is_negative = nibbles < 8
        abs_nibbles = tl.where(is_negative, 7 - nibbles, nibbles - 8)
        
        # Lookup table values (mirrored for negative)
        # Pre-compute lookup values
        lookup_0_3 = tl.where(abs_nibbles == 0, 0.0,
                     tl.where(abs_nibbles == 1, 0.07958029955625534,
                     tl.where(abs_nibbles == 2, 0.16093020141124725,
                     0.24611230194568634)))
        
        lookup_4_7 = tl.where(abs_nibbles == 4, 0.33791524171829224,
                     tl.where(abs_nibbles == 5, 0.44070982933044434,
                     tl.where(abs_nibbles == 6, 0.5626170039176941,
                     0.7229568362236023)))
        
        # Select the appropriate lookup value
        is_low = abs_nibbles < 4
        lookup_val = tl.where(is_low, lookup_0_3, lookup_4_7)
        
        # Handle special cases
        lookup_val = tl.where(nibbles == 0, -1.0,
                     tl.where(nibbles == 15, 1.0, 
                     tl.where(is_negative, -lookup_val, lookup_val)))
        
        # Apply scaling and store
        output = (lookup_val * scale).to(dtype)
        tl.store(output_ptr + idx, output, mask=mask)

@triton.jit
def _ultra_optimized_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    """Ultra-optimized kernel processing multiple rows and columns."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    if pid_m >= num_pid_m:
        return
    
    # Pre-compute NF4 lookup table in registers
    nf4_lut = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Process BLOCK_M x BLOCK_N tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create 2D masks
    rm_mask = rm[:, None] < m
    rn_mask = rn[None, :] < n
    mask = rm_mask & rn_mask
    
    # Process each row in the block
    for row_offset in range(BLOCK_M):
        row = pid_m * BLOCK_M + row_offset
        if row >= m:
            continue
            
        # Process columns in chunks of 64 (NF4 block size)
        for col_chunk in range(0, BLOCK_N, 64):
            col_base = pid_n * BLOCK_N + col_chunk
            if col_base >= n:
                continue
                
            # Load absmax for this 64-element chunk
            block_idx = col_base >> 6
            absmax_idx = row * blocks_per_row + block_idx
            absmax32_idx = row * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
            
            absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx)
            scale = absmax * 0.00787401574803149606 * absmax32
            
            # Process 64 elements
            cols = col_base + tl.arange(0, 64)
            col_mask = cols < n
            idx = row * n + cols
            
            # Vectorized load and nibble extraction
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=col_mask, other=0)
            
            # Extract nibbles using bit manipulation
            nibbles = tl.where((idx & 1) == 1, (packed >> 4) & 0xF, packed & 0xF)
            
            # Direct lookup using computed indices
            nf4_vals = tl.gather(nf4_lut, nibbles)
            
            # Apply scaling and store
            output = (nf4_vals * scale).to(dtype)
            tl.store(output_ptr + idx, output, mask=col_mask)

def optimized_triton_dequantize_nf4(module):
    """Optimized NF4 dequantization with auto-tuned kernel selection."""
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
    
    # Choose kernel based on problem size
    if M * N < 1024 * 1024:  # Small matrices
        # Use simpler kernel with less overhead
        BLOCK_SIZE = 128
        grid = (M, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
        
        _optimized_nf4_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            BLOCK_SIZE,
            dtype,
        )
    else:  # Large matrices
        # Use ultra-optimized kernel
        BLOCK_M = 8
        BLOCK_N = 512
        
        total_programs = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        grid = (total_programs,)
        
        _ultra_optimized_nf4_kernel[grid](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            BLOCK_M,
            BLOCK_N,
            dtype,
        )
    
    return output