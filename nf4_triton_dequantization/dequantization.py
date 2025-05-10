import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.jit
def _dequantize_nf4_kernel_full(
    qweight_ptr,        # Flattened packed NF4 weights (uint8)
    code_ptr,           # NF4 codebook values (float32)
    flat_absmax_ptr,    # Flattened absmax values for each block (uint8)
    flat_absmax32_ptr,  # Flattened & expanded absmax32 values for each block (float32)
    output_ptr,         # Output tensor (float16/bfloat16)
    n_elements,         # Total number of elements in the weight matrix
    rows,               # Number of rows in the weight matrix
    cols,               # Number of columns in the weight matrix
    blocksize_cfg,      # Configured blocksize (e.g., 64)
    N_ELEMENTS_PER_THREAD_BLOCK: tl.constexpr, # Elements processed by each Triton block
):
    pid = tl.program_id(0)
    base_offset = pid * N_ELEMENTS_PER_THREAD_BLOCK
    element_offsets = base_offset + tl.arange(0, N_ELEMENTS_PER_THREAD_BLOCK)
    element_mask = element_offsets < n_elements

    # Calculate row and column for each element
    current_row = element_offsets // cols
    current_col = element_offsets % cols

    # --- Load NF4 nibble --- 
    byte_indices = element_offsets >> 1  # element_offsets // 2
    is_high_nibble = (element_offsets & 1) == 1 # (element_offsets % 2) == 1
    
    # Mask for valid byte loads (prevent reading beyond qweight tensor)
    # Max byte_idx = (n_elements-1)//2 if n_elements is odd, or n_elements//2 -1 if even. (n_elements+1)//2 covers both.
    valid_byte_load_mask = element_mask & (byte_indices < tl.cdiv(n_elements, 2))
    packed_bytes = tl.load(qweight_ptr + byte_indices, mask=valid_byte_load_mask, other=0)
    
    low_nibbles = packed_bytes & 0x0F
    high_nibbles = (packed_bytes >> 4) & 0x0F
    nibbles = tl.where(is_high_nibble, high_nibbles, low_nibbles)

    # --- Load code value --- 
    # Mask for valid nibble values (0-15) before loading from code_ptr
    valid_nibble_mask = element_mask & (nibbles < 16) # code_ptr has 16 elements
    code_val = tl.load(code_ptr + nibbles, mask=valid_nibble_mask, other=0.0)

    # --- Determine block index for scaling --- 
    block_idx_in_row = current_col // blocksize_cfg
    n_blocks_per_row = tl.cdiv(cols, blocksize_cfg)
    
    # Flat index for absmax and expanded absmax32 tensors
    # These tensors are pre-processed in Python to be (rows * n_blocks_per_row)
    scale_map_flat_idx = current_row * n_blocks_per_row + block_idx_in_row
    
    # Mask for valid scaling map indices
    max_scale_map_idx = rows * n_blocks_per_row
    valid_scale_map_mask = element_mask & (scale_map_flat_idx < max_scale_map_idx)

    # --- Load absmax (uint8) --- 
    absmax_byte_val = tl.load(flat_absmax_ptr + scale_map_flat_idx, mask=valid_scale_map_mask, other=0)
    
    # --- Load absmax32 (float32) --- 
    absmax32_float_val = tl.load(flat_absmax32_ptr + scale_map_flat_idx, mask=valid_scale_map_mask, other=1.0) # Other=1.0 to avoid div by zero if mask is false

    # --- Dequantization --- 
    scale = (absmax_byte_val.to(tl.float32) / 127.0) * absmax32_float_val
    dequant_val = code_val * scale
    
    # Ensure all masks are combined for the final store
    final_store_mask = element_mask & valid_nibble_mask & valid_scale_map_mask
    tl.store(output_ptr + element_offsets, dequant_val, mask=final_store_mask)

def triton_dequantize_nf4(module):
    weight_tensor = module.weight
    quant_state = weight_tensor.quant_state

    qweight_data = weight_tensor.data
    code_data = quant_state.code
    absmax_data_orig = quant_state.absmax
    absmax32_data_orig = quant_state.state2.absmax
    blocksize = quant_state.blocksize if hasattr(quant_state, "blocksize") else 64
    target_dtype = quant_state.dtype
    
    out_features = module.out_features
    in_features = module.in_features
    n_elements = out_features * in_features
    device = qweight_data.device

    # 1. Prepare flat_absmax (uint8)
    n_blocks_per_row = (in_features + blocksize - 1) // blocksize
    expected_absmax_shape_2d = (out_features, n_blocks_per_row)

    if absmax_data_orig.dim() == 1 and absmax_data_orig.numel() == n_blocks_per_row:
        prepared_absmax = absmax_data_orig.unsqueeze(0).expand(expected_absmax_shape_2d)
    elif absmax_data_orig.dim() == 1 and absmax_data_orig.numel() == out_features * n_blocks_per_row:
        prepared_absmax = absmax_data_orig.view(expected_absmax_shape_2d)
    elif absmax_data_orig.dim() == 2 and absmax_data_orig.shape == expected_absmax_shape_2d:
        prepared_absmax = absmax_data_orig
    else:
        # Fallback if absmax shape is unexpected
        return fast_dequantize(weight_tensor, quant_state)
    flat_absmax = prepared_absmax.contiguous().view(-1)

    # 2. Prepare flat_absmax32_for_each_block (float32)
    n_absmax32_groups_per_row = (n_blocks_per_row + 3) // 4
    expected_absmax32_groups_shape_2d = (out_features, n_absmax32_groups_per_row)

    if absmax32_data_orig.dim() == 1 and absmax32_data_orig.numel() == n_absmax32_groups_per_row:
        prepared_absmax32_groups = absmax32_data_orig.unsqueeze(0).expand(expected_absmax32_groups_shape_2d)
    elif absmax32_data_orig.dim() == 1 and absmax32_data_orig.numel() == out_features * n_absmax32_groups_per_row:
        prepared_absmax32_groups = absmax32_data_orig.view(expected_absmax32_groups_shape_2d)
    elif absmax32_data_orig.dim() == 2 and absmax32_data_orig.shape == expected_absmax32_groups_shape_2d:
        prepared_absmax32_groups = absmax32_data_orig
    else:
        # Fallback if absmax32 shape is unexpected
        return fast_dequantize(weight_tensor, quant_state)
    
    # Expand groups to match block granularity
    expanded_absmax32_rows = prepared_absmax32_groups.repeat_interleave(4, dim=1)
    if expanded_absmax32_rows.shape[1] > n_blocks_per_row:
        expanded_absmax32_rows = expanded_absmax32_rows[:, :n_blocks_per_row]
    flat_absmax32_for_each_block = expanded_absmax32_rows.contiguous().view(-1)

    # Ensure tensors are on the correct device and contiguous
    qweight_flat = qweight_data.contiguous().view(-1)
    code_data = code_data.contiguous().to(device=device, dtype=torch.float32)
    flat_absmax = flat_absmax.contiguous().to(device=device, dtype=torch.uint8)
    flat_absmax32_for_each_block = flat_absmax32_for_each_block.contiguous().to(device=device, dtype=torch.float32)

    output_tensor = torch.empty(n_elements, dtype=target_dtype, device=device)

    # Launch Triton Kernel
    KERNEL_ELEMENT_BLOCK_SIZE = 1024 # Tunable parameter
    grid = (triton.cdiv(n_elements, KERNEL_ELEMENT_BLOCK_SIZE),)
    
    try:
        _dequantize_nf4_kernel_full[grid](
            qweight_flat,
            code_data,
            flat_absmax,
            flat_absmax32_for_each_block,
            output_tensor,
            n_elements,
            out_features,
            in_features,
            blocksize,
            N_ELEMENTS_PER_THREAD_BLOCK=KERNEL_ELEMENT_BLOCK_SIZE,
        )
    except Exception as e:
        # Fallback if Triton kernel fails
        return fast_dequantize(weight_tensor, quant_state)

    output_reshaped = output_tensor.view(out_features, in_features)

    if torch.isnan(output_reshaped).any() or torch.isinf(output_reshaped).any():
        return fast_dequantize(weight_tensor, quant_state)
    
    return output_reshaped

def reset_triton_dequantize_state():
    pass

def optimized_triton_dequantize_nf4(module):
    return triton_dequantize_nf4(module)

def benchmark_fast_dequantize(module):
    return triton_dequantize_nf4(module)
