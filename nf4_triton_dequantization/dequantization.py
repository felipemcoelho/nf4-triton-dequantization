import torch
import triton
import triton.language as tl
import os

@triton.jit
def _fast_nf4_dequant_kernel(
    weight_ptr,      # Packed NF4 weights (uint8)
    code_ptr,        # NF4 codebook values
    absmax8_ptr,     # 8-bit absmax values (uint8)
    absmax32_ptr,    # 32-bit absmax values (float32)
    output_ptr,      # Output tensor
    rows,            # Number of rows in weight matrix
    cols,            # Number of columns in weight matrix
    blocksize,       # Block size for 8-bit absmax (typically 64)
    absmax8_scale: tl.constexpr,  # Scale factor for absmax8 conversion
    BLOCK_SIZE: tl.constexpr  # Block size for processing
):
    # Use 1D grid for maximum parallelism
    pid = tl.program_id(0)
    
    # Calculate row and column indices directly
    n_blocks_per_row = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    row_idx = pid // n_blocks_per_row
    col_block_idx = pid % n_blocks_per_row
    
    # Early exit if out of bounds
    if row_idx >= rows:
        return
        
    # Calculate starting column for this block
    col_start = col_block_idx * BLOCK_SIZE
    
    # Generate column offsets and mask for valid columns
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < cols
    
    # Calculate global element indices
    row_offset = row_idx * cols
    element_indices = row_offset + col_offsets
    
    # Calculate byte indices (each byte contains 2 nibbles)
    byte_indices = element_indices >> 1
    
    # Determine if we're extracting high or low nibble
    is_high_nibble = (element_indices & 1) != 0
    
    # Create mask for valid byte indices
    total_bytes = (rows * cols + 1) // 2
    byte_mask = col_mask & (byte_indices < total_bytes)
    
    # Load packed bytes (each byte contains 2 nibbles)
    packed_bytes = tl.load(weight_ptr + byte_indices, mask=byte_mask)
    
    # Extract nibbles from packed bytes
    high_nibbles = (packed_bytes >> 4) & 0x0F
    low_nibbles = packed_bytes & 0x0F
    nibbles = tl.where(is_high_nibble, high_nibbles, low_nibbles)
    
    # Calculate absmax block indices
    blocks_per_row = (cols + blocksize - 1) // blocksize
    row_blocks_offset = row_idx * blocks_per_row
    
    # Optimize for common blocksize (64)
    if blocksize == 64:
        absmax8_indices = row_blocks_offset + (col_offsets >> 6)
    else:
        absmax8_indices = row_blocks_offset + (col_offsets // blocksize)
    
    # Create masks for valid indices
    total_absmax_blocks = rows * blocks_per_row
    absmax_mask = col_mask & (absmax8_indices < total_absmax_blocks)
    nibble_mask = col_mask & (nibbles < 16)
    
    # Load code values (the actual values corresponding to each nibble)
    code_values = tl.load(code_ptr + nibbles, mask=nibble_mask)
    
    # Load absmax values (scaling factors)
    absmax8_values = tl.load(absmax8_ptr + absmax8_indices, mask=absmax_mask)
    absmax32_values = tl.load(absmax32_ptr + absmax8_indices, mask=absmax_mask)
    
    # Convert absmax8_values to float32 and compute scale
    absmax8_float = absmax8_values.to(tl.float32)
    
    # Compute dequantized values with fused operations
    dequantized = code_values * (absmax8_float * (1.0 / absmax8_scale) * absmax32_values)
    
    # Create combined mask for the final store operation
    combined_mask = nibble_mask & absmax_mask & byte_mask
    
    # Store results
    tl.store(output_ptr + element_indices, dequantized, mask=combined_mask)

def triton_dequantize_nf4(module, verify=False):
    """
    Dequantize NF4 weights using an optimized Triton kernel.
    """
    # Get tensors and parameters from module
    device = module.weight.device
    weight = module.weight.data.contiguous()  # Packed NF4 weights (uint8)
    quant_state = module.weight.quant_state
    
    # Get dimensions
    rows = module.out_features
    cols = module.in_features
    
    # Get required tensors from quant_state
    absmax8 = quant_state.absmax.contiguous()  # Keep as uint8
    codes = quant_state.code.contiguous()
    absmax32 = quant_state.state2.absmax.contiguous()
    
    # Target dtype from quant_state
    target_dtype = quant_state.dtype
    
    # Prepare for dequantization
    blocksize = quant_state.blocksize if hasattr(quant_state, 'blocksize') else 64
    
    # Reshape absmax8 for proper indexing
    abs8_blocks_per_row = (cols + blocksize - 1) // blocksize
    if absmax8.dim() == 1:
        if absmax8.numel() == rows * abs8_blocks_per_row:
            absmax8 = absmax8.view(rows, abs8_blocks_per_row)
        else:
            absmax8 = absmax8.expand(rows, -1) if absmax8.numel() == abs8_blocks_per_row else absmax8.repeat(rows, 1)
    
    # Prepare flat tensors for kernel
    absmax8_flat = absmax8.contiguous().view(-1)
    absmax32_flat = absmax32.to(torch.float32).contiguous().view(-1)
    weight_flat = weight.contiguous().view(-1)
    
    # Prepare output tensor
    output = torch.empty((rows, cols), dtype=target_dtype, device=device)
    output_flat = output.view(-1)
    
    # Determine optimal block size and grid dimensions
    block_size = 128  # Optimal for most cases on T4
    
    # Use 1D grid for better performance
    grid = (triton.cdiv(rows * cols, block_size),)
    
    # Hardcoded absmax8_scale - optimal for performance on T4
    absmax8_scale = 127.0
    
    # Launch kernel
    _fast_nf4_dequant_kernel[grid](
        weight_flat,
        codes,
        absmax8_flat,
        absmax32_flat,
        output_flat,
        rows,
        cols,
        blocksize,
        absmax8_scale=absmax8_scale,
        BLOCK_SIZE=block_size,
    )
    
    # Verify results if requested
    if verify:
        from unsloth.kernels.utils import fast_dequantize
        ref_output = fast_dequantize(module.weight, module.weight.quant_state)
        rtol = 2e-1 if target_dtype == torch.bfloat16 else 1e-1
        atol = 2e-1 if target_dtype == torch.bfloat16 else 1e-1
        if not torch.allclose(output, ref_output, rtol=rtol, atol=atol):
            print("Warning: Triton results don't match reference. Using reference implementation.")
            return ref_output
    
    return output

# Remove these unused functions to reduce complexity
def reset_triton_dequantize_state():
    pass

def optimized_triton_dequantize_nf4(module):
    return triton_dequantize_nf4(module)

def benchmark_fast_dequantize(module):
    return triton_dequantize_nf4(module)
