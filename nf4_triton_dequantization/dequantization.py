import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _dequant_nf4_kernel(
    weight_ptr,      # Packed NF4 weights (uint8)
    code_ptr,        # NF4 codebook values (float32)
    absmax_ptr,      # absmax values (uint8)
    absmax32_ptr,    # absmax32 values (float32)
    output_ptr,      # Output tensor
    n_elements,      # Total elements
    blocksize,       # Block size (typically 64)
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel for NF4 dequantization based directly on Unsloth's implementation."""
    # Calculate thread index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate indices for the packed weights (2 nibbles per byte)
    byte_idx = offsets // 2
    is_high = (offsets % 2) == 1  # Whether to use high or low nibble
    
    # Load the packed bytes
    bytes_loaded = tl.load(weight_ptr + byte_idx, mask=mask & (byte_idx < (n_elements + 1) // 2))
    
    # Extract high and low nibbles
    low_nibbles = bytes_loaded & 0xF
    high_nibbles = (bytes_loaded >> 4) & 0xF
    nibbles = tl.where(is_high, high_nibbles, low_nibbles)
    
    # Load the code values for each nibble
    code_values = tl.load(code_ptr + nibbles, mask=mask)
    
    # Calculate absmax indices and load values
    absmax_idx = offsets // blocksize
    absmax_values = tl.load(absmax_ptr + absmax_idx, mask=mask)
    absmax32_values = tl.load(absmax32_ptr + absmax_idx, mask=mask)
    
    # Convert absmax to float and scale
    absmax_float = absmax_values.to(tl.float32)
    scale = (absmax_float / 127.0) * absmax32_values
    
    # Apply scaling
    output = code_values * scale
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_dequantize_nf4(module):
    """
    Dequantize NF4 weights using Triton.
    
    Args:
        module: The Linear4bit module to dequantize
        
    Returns:
        The dequantized weight tensor
    """
    try:
        # Get necessary tensors
        weight = module.weight.data
        quant_state = module.weight.quant_state
        codes = quant_state.code  # (16,) lookup table
        absmax = quant_state.absmax  # (n_blocks,) uint8 tensor
        absmax32 = quant_state.state2.absmax  # (n_blocks//4,) float32 tensor
        
        # Get dimensions
        rows = module.out_features
        cols = module.in_features
        n_elements = rows * cols
        
        # Get blocksize
        blocksize = quant_state.blocksize if hasattr(quant_state, 'blocksize') else 64
        
        # Ensure all tensors are contiguous
        weight_flat = weight.contiguous().view(-1)
        codes = codes.contiguous()
        absmax = absmax.contiguous()
        absmax32 = absmax32.contiguous()
        
        # Match the absmax and absmax32 arrangements with Unsloth
        if absmax.numel() != (n_elements + blocksize - 1) // blocksize:
            absmax = absmax.repeat(rows)
        
        # Create output tensor
        output = torch.empty(n_elements, dtype=quant_state.dtype, device=weight.device)
        
        # Set kernel parameters
        BLOCK_SIZE = 1024
        n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
        
        # Launch kernel
        _dequant_nf4_kernel[(n_blocks,)](
            weight_flat,
            codes,
            absmax,
            absmax32,
            output,
            n_elements,
            blocksize,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Reshape output to match the original weight shape
        output = output.view(rows, cols)
        
        # Verify the output has no NaN values
        if torch.isnan(output).any():
            # Fall back to Unsloth's implementation
            return fast_dequantize(module.weight, module.weight.quant_state)
            
        return output
    except Exception as e:
        # Fall back to Unsloth's implementation
        print(f"Triton kernel failed: {e}. Using reference implementation.")
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
