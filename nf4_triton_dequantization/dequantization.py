import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize

@triton.autotune(
    configs=[
        triton.Config({'AUTOTUNE_BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'AUTOTUNE_BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'AUTOTUNE_BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'AUTOTUNE_BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'AUTOTUNE_BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['n_elements', 'rows', 'cols', 'blocksize_cfg'],
)
@triton.jit
def _dequantize_nf4_kernel_full(
    qweight_ptr,        # Flattened packed NF4 weights (uint8)
    code_ptr,           # NF4 codebook values (float32)
    flat_absmax_ptr,    # Flattened absmax values for each block (uint8)
    flat_absmax32_grouped_ptr,  # Flattened absmax32 values, one per 4-block group (float32)
    output_ptr,         # Output tensor (float16/bfloat16)
    n_elements,         # Total number of elements in the weight matrix
    rows,               # Number of rows in the weight matrix
    cols,               # Number of columns in the weight matrix
    blocksize_cfg,      # Configured blocksize (e.g., 64)
    n_blocks_per_row_cfg, # Number of blocks per row
    n_absmax32_groups_per_row_cfg, # Number of absmax32 groups per row
    AUTOTUNE_BLOCK_SIZE: tl.constexpr, # Elements processed by each Triton program instance (tuned)
):
    pid = tl.program_id(0)
    base_offset = pid * AUTOTUNE_BLOCK_SIZE
    element_offsets = base_offset + tl.arange(0, AUTOTUNE_BLOCK_SIZE)
    element_mask = element_offsets < n_elements

    current_row = element_offsets // cols
    current_col = element_offsets % cols

    # --- Load NF4 nibble --- 
    byte_indices = element_offsets >> 1
    is_high_nibble = (element_offsets & 1) == 1
    valid_byte_load_mask = element_mask & (byte_indices < tl.cdiv(n_elements, 2))
    packed_bytes = tl.load(qweight_ptr + byte_indices, mask=valid_byte_load_mask, other=0)
    low_nibbles = packed_bytes & 0x0F
    high_nibbles = (packed_bytes >> 4) & 0x0F
    nibbles = tl.where(is_high_nibble, high_nibbles, low_nibbles)

    # --- Load code value --- 
    valid_nibble_mask = element_mask & (nibbles < 16)
    code_val = tl.load(code_ptr + nibbles, mask=valid_nibble_mask, other=0.0)

    # --- Determine block index for scaling --- 
    block_idx_in_row = current_col // blocksize_cfg
    
    # --- Load absmax (uint8) --- 
    # Flat index for absmax tensor: (rows * n_blocks_per_row_cfg)
    flat_absmax_idx = current_row * n_blocks_per_row_cfg + block_idx_in_row
    max_flat_absmax_idx = rows * n_blocks_per_row_cfg
    valid_absmax_load_mask = element_mask & (flat_absmax_idx < max_flat_absmax_idx)
    absmax_byte_val = tl.load(flat_absmax_ptr + flat_absmax_idx, mask=valid_absmax_load_mask, other=0)
    
    # --- Load absmax32 (float32) from grouped tensor --- 
    absmax32_group_offset_in_row = block_idx_in_row // 4
    # Flat index for absmax32_grouped tensor: (rows * n_absmax32_groups_per_row_cfg)
    flat_absmax32_grouped_idx = current_row * n_absmax32_groups_per_row_cfg + absmax32_group_offset_in_row
    max_flat_absmax32_grouped_idx = rows * n_absmax32_groups_per_row_cfg
    valid_absmax32_load_mask = element_mask & (flat_absmax32_grouped_idx < max_flat_absmax32_grouped_idx)
    absmax32_float_val = tl.load(flat_absmax32_grouped_ptr + flat_absmax32_grouped_idx, mask=valid_absmax32_load_mask, other=0.0)

    # --- Dequantization --- 
    scale = (absmax_byte_val.to(tl.float32) / 127.0) * absmax32_float_val
    dequant_val = code_val * scale
    
    final_store_mask = element_mask # Individual masks already applied to intermediate values via 'other'
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
        return fast_dequantize(weight_tensor, quant_state) # Fallback
    flat_absmax = prepared_absmax.contiguous().view(-1)

    # 2. Prepare flat_absmax32_grouped (float32)
    n_absmax32_groups_per_row = (n_blocks_per_row + 3) // 4
    expected_absmax32_groups_shape_2d = (out_features, n_absmax32_groups_per_row)

    if absmax32_data_orig.dim() == 1 and absmax32_data_orig.numel() == n_absmax32_groups_per_row:
        prepared_absmax32_groups = absmax32_data_orig.unsqueeze(0).expand(expected_absmax32_groups_shape_2d)
    elif absmax32_data_orig.dim() == 1 and absmax32_data_orig.numel() == out_features * n_absmax32_groups_per_row:
        prepared_absmax32_groups = absmax32_data_orig.view(expected_absmax32_groups_shape_2d)
    elif absmax32_data_orig.dim() == 2 and absmax32_data_orig.shape == expected_absmax32_groups_shape_2d:
        prepared_absmax32_groups = absmax32_data_orig
    else:
        return fast_dequantize(weight_tensor, quant_state) # Fallback
    
    flat_absmax32_grouped = prepared_absmax32_groups.contiguous().view(-1)

    qweight_flat = qweight_data.contiguous().view(-1)
    code_data = code_data.contiguous().to(device=device, dtype=torch.float32)
    flat_absmax = flat_absmax.contiguous().to(device=device, dtype=torch.uint8)
    flat_absmax32_grouped = flat_absmax32_grouped.contiguous().to(device=device, dtype=torch.float32)

    output_tensor = torch.empty(n_elements, dtype=target_dtype, device=device)

    # Grid is now a lambda function for the autotuner
    grid = lambda META: (triton.cdiv(n_elements, META['AUTOTUNE_BLOCK_SIZE']),)
    
    try:
        _dequantize_nf4_kernel_full[grid](
            qweight_flat,
            code_data,
            flat_absmax,
            flat_absmax32_grouped,
            output_tensor,
            n_elements,
            out_features,
            in_features,
            blocksize, 
            n_blocks_per_row,
            n_absmax32_groups_per_row,
            # AUTOTUNE_BLOCK_SIZE is passed by the autotuner
        )
    except Exception as e:
        # Consider logging the exception e for debugging
        return fast_dequantize(weight_tensor, quant_state) # Fallback

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
