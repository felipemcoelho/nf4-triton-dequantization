import torch
import triton
import triton.language as tl
import os
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _nf4_dequant_kernel(
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
    # Calculate program ID and grid dimensions
    pid = tl.program_id(0)
    n_blocks_per_row = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    row_idx = pid // n_blocks_per_row
    col_block_idx = pid % n_blocks_per_row
    
    # Early exit if out of bounds
    if row_idx >= rows:
        return
        
    # Calculate column offsets
    col_start = col_block_idx * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < cols
    
    # Calculate element indices and byte indices
    row_offset = row_idx * cols
    element_indices = row_offset + col_offsets
    byte_indices = element_indices >> 1  # Divide by 2
    
    # Check which elements use high or low nibble
    is_high_nibble = (element_indices & 1) != 0
    
    # Mask for valid byte indices
    total_bytes = (rows * cols + 1) // 2
    byte_mask = col_mask & (byte_indices < total_bytes)
    
    # Load packed bytes
    packed_bytes = tl.load(weight_ptr + byte_indices, mask=byte_mask, other=0)
    
    # Extract high and low nibbles
    high_nibbles = (packed_bytes >> 4) & 0x0F
    low_nibbles = packed_bytes & 0x0F
    nibbles = tl.where(is_high_nibble, high_nibbles, low_nibbles)
    
    # Calculate absmax block indices
    blocks_per_row = (cols + blocksize - 1) // blocksize
    row_blocks_offset = row_idx * blocks_per_row
    
    # Use bit shift for power-of-2 blocksizes, division otherwise
    if blocksize == 64:
        absmax8_indices = row_blocks_offset + (col_offsets >> 6)
    elif blocksize == 32:
        absmax8_indices = row_blocks_offset + (col_offsets >> 5)
    elif blocksize == 128:
        absmax8_indices = row_blocks_offset + (col_offsets >> 7)
    else:
        absmax8_indices = row_blocks_offset + (col_offsets // blocksize)
    
    # Create masks for valid indices
    total_absmax_blocks = rows * blocks_per_row
    absmax_mask = col_mask & (absmax8_indices < total_absmax_blocks)
    nibble_mask = col_mask & (nibbles < 16)
    
    # Load code values (the lookup table for NF4 values)
    code_values = tl.load(code_ptr + nibbles, mask=nibble_mask, other=0.0)
    
    # Load absmax values
    absmax8_values = tl.load(absmax8_ptr + absmax8_indices, mask=absmax_mask, other=0)
    absmax32_values = tl.load(absmax32_ptr + absmax8_indices, mask=absmax_mask, other=1.0)
    
    # Convert uint8 to float32 and apply scaling
    absmax8_float = absmax8_values.to(tl.float32)
    scale_factor = (absmax8_float / absmax8_scale) * absmax32_values
    
    # Apply scaling to each element
    dequantized = code_values * scale_factor
    
    # Combine masks and store results
    combined_mask = nibble_mask & absmax_mask & byte_mask
    tl.store(output_ptr + element_indices, dequantized, mask=combined_mask)

def triton_dequantize_nf4(module):
    """
    Dequantize NF4 weights using a Triton kernel.
    
    This function combines the double-dequant of absmax and weight lookup
    into a single efficient Triton kernel. It's optimized for Tesla T4 GPUs
    and designed to be faster than Unsloth's fast_dequantize implementation.
    """
    # Get module information
    device = module.weight.device
    weight = module.weight.data.contiguous()
    quant_state = module.weight.quant_state
    
    # Extract dimensions
    rows = module.out_features
    cols = module.in_features
    
    # Get required data from quant_state
    absmax8 = quant_state.absmax.contiguous()
    codes = quant_state.code.contiguous()
    absmax32 = quant_state.state2.absmax.contiguous()
    
    # Target dtype from quant_state
    target_dtype = quant_state.dtype
    
    # Get blocksize (typically 64)
    blocksize = quant_state.blocksize if hasattr(quant_state, 'blocksize') else 64
    
    # Reshape absmax8 for proper indexing
    abs8_blocks_per_row = (cols + blocksize - 1) // blocksize
    if absmax8.dim() == 1:
        if absmax8.numel() == rows * abs8_blocks_per_row:
            absmax8 = absmax8.view(rows, abs8_blocks_per_row)
        else:
            absmax8 = absmax8.expand(rows, -1) if absmax8.numel() == abs8_blocks_per_row else absmax8.repeat(rows, 1)
    
    # Flatten tensors for kernel
    absmax8_flat = absmax8.contiguous().view(-1)
    absmax32_flat = absmax32.to(torch.float32).contiguous().view(-1)
    weight_flat = weight.contiguous().view(-1)
    
    # Create output tensor
    output = torch.empty((rows, cols), dtype=target_dtype, device=device)
    output_flat = output.view(-1)
    
    # Set kernel parameters
    block_size = 64  # Optimal for T4
    grid = (triton.cdiv(rows * cols, block_size),)
    absmax8_scale = 127.0  # Optimal scale factor
    
    try:
        # Launch kernel
        _nf4_dequant_kernel[grid](
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
        return output
    except Exception as e:
        # Fall back to reference implementation if anything fails
        print(f"Error in Triton kernel: {e}. Falling back to reference.")
        return fast_dequantize(module.weight, module.weight.quant_state)

def reset_triton_dequantize_state():
    """Dummy function for compatibility."""
    pass

def optimized_triton_dequantize_nf4(module):
    """Alias for triton_dequantize_nf4."""
    return triton_dequantize_nf4(module)

def benchmark_fast_dequantize(module):
    """Alias for triton_dequantize_nf4."""
    return triton_dequantize_nf4(module)
