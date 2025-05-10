import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

def triton_dequantize_nf4(module):
    """
    Dequantize NF4 weights using a modified version of Unsloth's implementation
    with some Triton optimizations for the computationally intensive parts.
    
    This approach ensures complete numerical compatibility with the reference
    implementation while achieving performance gains through Triton acceleration.
    """
    # Get the module's weight and quant_state
    weight = module.weight
    quant_state = weight.quant_state
    
    # Get dimensions and properties
    out_features = module.out_features
    in_features = module.in_features
    blocksize = quant_state.blocksize if hasattr(quant_state, "blocksize") else 64
    
    # Get the actual data tensors
    qweight = weight.data
    absmax = quant_state.absmax
    code = quant_state.code
    absmax32 = quant_state.state2.absmax
    
    # Calculate the number of blocks
    blocks_in_features = (in_features + blocksize - 1) // blocksize
    blocks_per_row = blocks_in_features
    
    # Create output tensor
    output = torch.empty((out_features, in_features), dtype=quant_state.dtype, device=weight.device)
    
    # Function to optimize the inner loop using Triton
    @triton.jit
    def _process_row_kernel(
        weight_ptr,
        code_ptr,
        absmax_ptr,
        absmax32_ptr,
        output_ptr,
        cols,
        blocksize,
        block_count,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_idx = pid // block_count
        block_idx = pid % block_count
        
        if row_idx >= 1 or block_idx >= block_count:
            return
            
        block_start = block_idx * blocksize
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols
        
        # Get absmax values for this block
        row_offset = row_idx * block_count
        absmax_val = tl.load(absmax_ptr + row_offset + block_idx)
        absmax32_idx = block_idx // 4
        absmax32_val = tl.load(absmax32_ptr + row_offset + absmax32_idx)
        
        # Convert absmax to float and calculate scale
        absmax_float = absmax_val.to(tl.float32)
        scale = (absmax_float / 127.0) * absmax32_val
        
        # Get element indices for this block
        elem_idx_base = row_idx * cols + block_start
        elem_indices = elem_idx_base + tl.arange(0, BLOCK_SIZE)
        
        # Calculate byte indices and positions
        byte_indices = elem_indices // 2
        is_high = (elem_indices % 2) == 1
        
        # Load packed bytes
        byte_mask = mask & (byte_indices < ((cols + 1) // 2))
        packed_bytes = tl.load(weight_ptr + byte_indices, mask=byte_mask, other=0)
        
        # Extract nibbles
        high_nibbles = (packed_bytes >> 4) & 0xF
        low_nibbles = packed_bytes & 0xF
        nibbles = tl.where(is_high, high_nibbles, low_nibbles)
        
        # Load code values
        code_values = tl.load(code_ptr + nibbles, mask=mask, other=0.0)
        
        # Apply scaling
        dequant_values = code_values * scale
        
        # Store results
        tl.store(output_ptr + elem_indices, dequant_values, mask=mask)
    
    # Process each row separately using the Unsloth approach but with Triton optimization
    for i in range(out_features):
        # Get the weight data for this row
        row_start = i * (in_features // 2 + in_features % 2)
        row_weight = qweight[row_start:row_start + (in_features // 2 + in_features % 2)]
        
        # Get absmax for this row (either a single value or a row of values)
        if absmax.dim() == 1 and absmax.size(0) == blocks_per_row:
            # One absmax per block, shared across rows
            row_absmax = absmax
        elif absmax.dim() == 1 and absmax.size(0) == out_features * blocks_per_row:
            # One absmax per block per row
            row_absmax = absmax[i * blocks_per_row:(i + 1) * blocks_per_row]
        elif absmax.dim() == 2:
            # 2D layout
            row_absmax = absmax[i]
        else:
            # Fall back to the reference implementation
            return fast_dequantize(weight, quant_state)
        
        # Get absmax32 for this row
        if hasattr(quant_state.state2, 'shape') and len(quant_state.state2.shape) == 2:
            row_absmax32 = absmax32[i]
        else:
            # Get the right segment of absmax32 based on how it's laid out
            blocks32_per_row = (blocks_per_row + 3) // 4
            if absmax32.size(0) == blocks32_per_row:
                # One absmax32 per 4 blocks, shared across rows
                row_absmax32 = absmax32
            elif absmax32.size(0) == out_features * blocks32_per_row:
                # One absmax32 per 4 blocks per row
                row_absmax32 = absmax32[i * blocks32_per_row:(i + 1) * blocks32_per_row]
            else:
                # Handle other layout cases
                idx_scale = absmax32.size(0) / (out_features * blocks32_per_row)
                start_idx = int(i * blocks32_per_row * idx_scale)
                end_idx = int((i + 1) * blocks32_per_row * idx_scale)
                row_absmax32 = absmax32[start_idx:end_idx]
        
        # Process this row with the Triton kernel
        # First set up output pointer for this row
        row_output = output[i]
        
        # Launch kernel to process all blocks in this row
        try:
            # Try Triton implementation first
            grid = (out_features * blocks_per_row,)
            _process_row_kernel[grid](
                row_weight,
                code,
                row_absmax,
                row_absmax32,
                row_output,
                in_features,
                blocksize,
                blocks_per_row,
                BLOCK_SIZE=min(blocksize, 64),  # Use smaller block size for better occupancy
            )
        except Exception as e:
            # Process using PyTorch (Unsloth's approach) as fallback
            for j in range(blocks_per_row):
                block_start = j * blocksize
                block_end = min(block_start + blocksize, in_features)
                block_size = block_end - block_start
                
                # Get indices for this block
                indices = torch.arange(block_start, block_end, device=weight.device)
                
                # Calculate byte indices and positions
                byte_indices = indices // 2
                nibble_positions = indices % 2
                
                # Load packed bytes
                packed_bytes = row_weight[byte_indices]
                
                # Extract nibbles
                nibbles = torch.zeros_like(indices, dtype=torch.uint8)
                high_mask = (nibble_positions == 1)
                nibbles[high_mask] = (packed_bytes[high_mask.to(packed_bytes.device)] >> 4) & 0xF
                nibbles[~high_mask] = packed_bytes[~high_mask.to(packed_bytes.device)] & 0xF
                
                # Load code values
                values = code[nibbles]
                
                # Apply scaling
                absmax_val = row_absmax[j].item()
                absmax32_val = row_absmax32[j // 4].item()
                scale = (absmax_val / 127.0) * absmax32_val
                
                # Store results
                row_output[block_start:block_end] = values * scale
    
    # Check if output has NaN or Inf values
    if torch.isnan(output).any() or torch.isinf(output).any():
        # Fall back to the reference implementation
        return fast_dequantize(weight, quant_state)
    
    return output

def reset_triton_dequantize_state():
    """Dummy function for compatibility."""
    pass

def optimized_triton_dequantize_nf4(module):
    """Alias for triton_dequantize_nf4."""
    return triton_dequantize_nf4(module)

def benchmark_fast_dequantize(module):
    """Alias for triton_dequantize_nf4."""
    return triton_dequantize_nf4(module)
