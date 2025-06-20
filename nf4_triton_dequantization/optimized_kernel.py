import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

# NF4 quantization constants
NF4_QUANT_TABLE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]

@triton.jit
def _ultra_fast_nf4_kernel(
    qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
    M, N, 
    blocks_per_row: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ultra-optimized NF4 dequantization kernel with vectorized operations."""
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    # Compute block indices
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n
    
    # Skip if out of bounds
    if pid_m >= tl.cdiv(M, BLOCK_M):
        return
    
    # Block boundaries
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_base = pid_n * BLOCK_N
    
    # Create NF4 lookup table in shared memory
    nf4_table = tl.zeros((16,), dtype=tl.float32)
    for i in tl.static_range(16):
        nf4_table[i] = tl.load(tl.arange(0, 1) * 0 + NF4_QUANT_TABLE[i])
    
    # Process each row in the block
    for m in tl.static_range(BLOCK_M):
        row_idx = pid_m * BLOCK_M + m
        if row_idx >= M:
            break
            
        # Process columns in chunks of BLOCK_K (aligned to 64 for absmax blocks)
        for k in range(0, BLOCK_N, BLOCK_K):
            col_start = rn_base + k
            if col_start >= N:
                break
                
            # Calculate block index for absmax
            block_idx = col_start >> 6  # // 64
            absmax_idx = row_idx * blocks_per_row + block_idx
            
            # Load absmax values (primary and secondary dequantization)
            absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
            absmax32_idx = row_idx * ((blocks_per_row + 3) >> 2) + (block_idx >> 2)
            absmax32 = tl.load(absmax32_ptr + absmax32_idx)
            
            # Fused scaling factor
            scale = absmax * 0.00787401574803149606 * absmax32
            
            # Process BLOCK_K elements
            rn = col_start + tl.arange(0, BLOCK_K)
            mask = rn < N
            
            # Calculate packed weight indices
            linear_idx = row_idx * N + rn
            packed_idx = linear_idx >> 1
            
            # Vectorized load of packed weights
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            
            # Extract nibbles using bit manipulation
            is_odd = (linear_idx & 1).to(tl.int32)
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Vectorized NF4 lookup using gather
            nf4_vals = tl.load(nf4_table + nibbles, mask=mask, other=0.0)
            
            # Apply scaling and store with vectorized operations
            output = nf4_vals * scale
            output_idx = row_idx * N + rn
            tl.store(output_ptr + output_idx, output.to(tl.float16), mask=mask)

@triton.jit
def _hyper_optimized_nf4_kernel(
    qweight_ptr, absmax_ptr, absmax32_ptr, output_ptr,
    M, N,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Hyper-optimized kernel processing multiple rows and columns together."""
    pid = tl.program_id(0)
    
    # Calculate which elements this thread block processes
    elements_per_block = BLOCK_SIZE
    start_idx = pid * elements_per_block
    
    # Process elements in chunks
    for i in range(0, elements_per_block, 128):  # Process 128 elements at a time
        idx = start_idx + i + tl.arange(0, 128)
        mask = idx < (M * N)
        
        # Convert linear index to row/col
        row = idx // N
        col = idx % N
        
        # Skip if out of bounds
        row_mask = row < M
        full_mask = mask & row_mask
        
        # Calculate block indices
        block_idx = col >> 6
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        
        # Load scaling factors
        absmax = tl.load(absmax_ptr + absmax_idx, mask=full_mask, other=0).to(tl.float32)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx, mask=full_mask, other=1.0)
        scale = absmax * 0.00787401574803149606 * absmax32
        
        # Load packed weights
        packed_idx = idx >> 1
        packed = tl.load(qweight_ptr + packed_idx, mask=full_mask, other=0)
        
        # Extract nibbles
        is_odd = (idx & 1).to(tl.int32)
        nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
        
        # NF4 lookup using conditional selection (optimized for GPU)
        nf4_vals = tl.where(nibbles == 0, -1.0,
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
        output = nf4_vals * scale
        tl.store(output_ptr + idx, output, mask=full_mask)

def ultra_fast_triton_dequantize_nf4(module):
    """Ultra-fast Triton NF4 dequantization with aggressive optimizations."""
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
    
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Choose kernel based on matrix size
    if M * N < 4096 * 4096:  # Small to medium matrices
        BLOCK_M = 4
        BLOCK_N = 256
        BLOCK_K = 64
        
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        _ultra_fast_nf4_kernel[grid](
            qweight.contiguous(),
            absmax.contiguous(),
            absmax32.contiguous(),
            output,
            M, N,
            blocks_per_row,
            BLOCK_M, BLOCK_N, BLOCK_K,
        )
    else:  # Large matrices
        BLOCK_SIZE = 4096
        total_elements = M * N
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        
        _hyper_optimized_nf4_kernel[grid](
            qweight.contiguous().view(-1),
            absmax.contiguous().view(-1),
            absmax32.contiguous().view(-1),
            output.view(-1),
            M, N,
            blocks_per_row,
            absmax32_per_row,
            BLOCK_SIZE,
        )
    
    return output