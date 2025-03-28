import torch
import triton
import triton.language as tl
import math # For ceiling division in Python host code

@triton.jit
def _nf4_kernel(
    weight_ptr,  # Pointer to packed uint8 weight data, logically [R, C//2]
    abs8_ptr,    # Pointer to uint8 absmax data, logically [R, C//64]
    code_ptr,    # Pointer to float32 NF4 codebook [16]
    abs32_ptr,   # Pointer to float32 absmax data, logically [R, C//256]
    out_f32_ptr, # Pointer to output float32 buffer, logically [R, C]
    R, C,        # De-quantized float dimensions (Rows, Columns)
    stride_w_row, stride_w_col,   # Strides for logical weight_ptr [R, C//2]
    stride_a8_row, stride_a8_col,  # Strides for logical abs8_ptr [R, C//64]
    stride_a32_row, stride_a32_col,# Strides for logical abs32_ptr [R, C//256]
    stride_out_row, stride_out_col,# Strides for logical out_f32_ptr [R, C]
    BLOCK_SIZE_C: tl.constexpr = 64, # Process in blocks of 64 columns
    BLOCK_SIZE_ABS32: tl.constexpr = 256, # Block size for the second absmax level
):
    """
    Triton kernel to dequantize NF4 weights with double quantization.
    Each program instance handles a block of BLOCK_SIZE_C columns for a specific row.
    """
    # Identify the row and the block column this program instance processes
    pid_row = tl.program_id(0) # Row index
    pid_blk_col = tl.program_id(1) # Block column index (0, 1, ..., ceil(C/BLOCK_SIZE_C)-1)

    # Calculate the starting column index for this block
    col_start = pid_blk_col * BLOCK_SIZE_C

    # --- Calculate Pointers to the Start of the Current Row ---
    # These pointers allow accessing data specific to the current row (pid_row)
    row_weight_ptr = weight_ptr + pid_row * stride_w_row
    row_abs8_ptr   = abs8_ptr   + pid_row * stride_a8_row
    row_abs32_ptr  = abs32_ptr  + pid_row * stride_a32_row
    row_out_ptr    = out_f32_ptr + pid_row * stride_out_row

    # --- Load Quantization Scales for this Block ---
    # abs8 scale: One scale per BLOCK_SIZE_C (64) columns. Index is pid_blk_col.
    abs8_offset = pid_blk_col * stride_a8_col
    # abs32 scale: One scale per BLOCK_SIZE_ABS32 (256) columns. Shared across 4 blocks (256/64).
    abs32_blk_idx = pid_blk_col // (BLOCK_SIZE_ABS32 // BLOCK_SIZE_C)
    abs32_offset = abs32_blk_idx * stride_a32_col

    # Load the actual scale values
    abs8_val  = tl.load(row_abs8_ptr + abs8_offset)
    abs32_val = tl.load(row_abs32_ptr + abs32_offset)

    # Compute the combined scale factor using the double quantization formula
    # scale = (absmax_block64 / 127.0) * absmax_block256
    # Cast uint8 absmax to float32 for calculation. Use multiplication for potential efficiency.
    scale_f = (tl.cast(abs8_val, tl.float32) * (1.0 / 127.0)) * abs32_val

    # --- Dequantize Block ---
    # Create a range of column indices for the current block
    cols = col_start + tl.arange(0, BLOCK_SIZE_C)
    # Create a mask to handle blocks that extend beyond the actual column dimension C
    col_mask = cols < C

    # Calculate corresponding indices in the packed weight tensor (which has C//2 columns)
    # Each byte stores two 4-bit nibbles.
    packed_byte_col_indices = cols // 2
    # Determine if the target nibble is the lower 4 bits (even columns) or upper 4 bits (odd columns)
    nibble_is_low = (cols % 2) == 0

    # Load the packed uint8 bytes containing the nibbles for this block
    # Apply col_mask to prevent reading out of bounds for partial blocks at the edge
    packed_bytes_ptr = row_weight_ptr + packed_byte_col_indices * stride_w_col
    packed_bytes = tl.load(packed_bytes_ptr, mask=col_mask, other=0) # other=0 prevents garbage data affecting extraction

    # Extract the 4-bit nibbles
    low_nibbles  = packed_bytes & 0x0F
    high_nibbles = (packed_bytes >> 4) & 0x0F
    # Select the correct nibble based on whether the original column index was even or odd
    nibbles = tl.where(nibble_is_low, low_nibbles, high_nibbles)

    # Look up the float value corresponding to the nibble from the NF4 codebook
    # Apply col_mask to prevent using indices from masked-out columns
    code_vals = tl.load(code_ptr + nibbles, mask=col_mask, other=0.0) # other=0.0 ensures masked values don't affect output

    # Apply the combined scaling factor to the codebook value
    dequantized_vals = code_vals * scale_f

    # --- Store Results ---
    # Calculate output pointer offsets for the dequantized values
    out_offsets = cols * stride_out_col
    out_ptr = row_out_ptr + out_offsets
    # Store the results, applying the mask to write only valid column data
    tl.store(out_ptr, dequantized_vals, mask=col_mask)


def _validate_tensor(tensor, expected_dtype, logical_shape, tensor_name):
    """
    Validates tensor dtype, numel, and contiguity. Reshapes if numel matches
    logical_shape but shape differs (e.g., flat tensor). Ensures contiguity.
    """
    if tensor is None:
        raise ValueError(f"{tensor_name} is None.")
    if tensor.dtype != expected_dtype:
        raise TypeError(f"{tensor_name} dtype mismatch. Expected {expected_dtype}, got {tensor.dtype}")

    expected_numel = math.prod(logical_shape)
    actual_numel = tensor.numel()

    if actual_numel != expected_numel:
        raise ValueError(
            f"{tensor_name} numel mismatch. Expected logical shape {logical_shape} ({expected_numel} elements), "
            f"but got tensor with {actual_numel} elements (shape {tensor.shape}). Cannot proceed."
        )

    # If numel matches, reshape if necessary (e.g., from flat to logical 2D)
    if tensor.shape != logical_shape:
        # print(f"Warning: {tensor_name} shape is {tensor.shape}, reshaping to logical {logical_shape}.")
        try:
            tensor = tensor.view(logical_shape) # Use view for efficiency if possible
        except RuntimeError: # Fallback to reshape if view fails (e.g., non-contiguous)
             # print(f"Warning: {tensor_name}.view failed, using reshape.")
             tensor = tensor.reshape(logical_shape)

    # Ensure contiguous memory layout for reliable stride calculations by the kernel
    if not tensor.is_contiguous():
        # print(f"Warning: {tensor_name} is not contiguous. Making it contiguous.")
        tensor = tensor.contiguous()

    return tensor

def _decode_nf4(
    weight_packed,  # uint8 tensor, logically [R, C//2]
    absmax8,        # uint8 tensor, logically [R, C//64]
    codebook,       # float32 tensor, [16]
    absmax32,       # float32 tensor, logically [R, C//256]
    R, C,           # Target float dimensions (original OutFeatures, InFeatures)
    target_dtype    # Final dtype (torch.float16 or torch.bfloat16)
):
    """
    Host function to validate inputs, set up grid, and launch the Triton kernel.
    """
    device = weight_packed.device # Assume all inputs are on the same device

    # --- Define Logical Shapes based on Original Float Dimensions R, C ---
    logical_w_shape = (R, C // 2)
    logical_a8_shape = (R, math.ceil(C / 64)) # ceil for cases where C is not multiple of 64
    logical_a32_shape = (R, math.ceil(C / 256)) # ceil for cases where C is not multiple of 256
    logical_code_shape = (16,)

    # --- Validate Inputs against Logical Structure ---
    # This ensures tensors have the correct type, element count, and are contiguous.
    # It reshapes them to the logical 2D structure if they were provided flat but have correct numel.
    weight_packed = _validate_tensor(weight_packed, torch.uint8, logical_w_shape, "Packed Weight")
    absmax8 = _validate_tensor(absmax8, torch.uint8, logical_a8_shape, "Absmax8 (scale level 1)")
    absmax32 = _validate_tensor(absmax32, torch.float32, logical_a32_shape, "Absmax32 (scale level 2)")
    codebook = _validate_tensor(codebook, torch.float32, logical_code_shape, "Codebook")

    # Allocate output tensor in float32 for kernel computation
    out_f32 = torch.empty((R, C), device=device, dtype=torch.float32)

    # --- Launch Triton Kernel ---
    # Grid: One program per row (R), and one program per 64-column block within the row
    grid = (R, math.ceil(C / 64))

    _nf4_kernel[grid](
        weight_packed, absmax8, codebook, absmax32, out_f32,
        R, C,
        # Pass strides of the validated, contiguous tensors
        weight_packed.stride(0), weight_packed.stride(1),
        absmax8.stride(0), absmax8.stride(1),
        absmax32.stride(0), absmax32.stride(1),
        out_f32.stride(0), out_f32.stride(1),
        # BLOCK_SIZE_C=64 # Default defined in kernel
        num_warps=4 # A common default, may require tuning
    )

    # Cast the float32 result to the desired target dtype (float16 or bfloat16)
    return out_f32.to(target_dtype)


def triton_dequantize_nf4(module):
    """
    Entry point for dequantizing an NF4 Linear layer (like bnb.Linear4bit).

    Retrieves necessary tensors (weights, scales, codebook) from the module,
    determines original dimensions, and calls the Triton-based decoding function.
    """
    # 1. Retrieve the packed uint8 weight tensor
    # Common locations: module.weight.data or module.weight
    W_packed = getattr(module.weight, 'data', module.weight)
    if not isinstance(W_packed, torch.Tensor) or W_packed.dtype != torch.uint8:
        raise TypeError(f"Could not find packed uint8 weight tensor in module: {type(module).__name__}")

    # 2. Access quantization state
    quant_state = getattr(module, 'quant_state', None)
    if quant_state is None:
        raise AttributeError(f"Module {type(module).__name__} lacks required 'quant_state'.")

    # 3. Determine target dtype (float16/bfloat16)
    target_dtype = getattr(quant_state, 'dtype', torch.float16) # Default if attribute missing
    if target_dtype not in [torch.float16, torch.bfloat16]:
        print(f"Warning: Unusual target dtype {target_dtype} found in quant_state.")

    # 4. Get original float tensor dimensions (R=out_features, C=in_features)
    # Prefer quant_state.shape as the source of truth
    shape = getattr(quant_state, 'shape', None)
    if shape is not None and len(shape) == 2:
        R_orig, C_orig = shape
    else:
        # Fallback to module attributes if quant_state.shape is unavailable/invalid
        R_orig = getattr(module, 'out_features', None)
        C_orig = getattr(module, 'in_features', None)
        if R_orig is None or C_orig is None:
             raise ValueError(f"Cannot determine original shape (out/in features) for {type(module).__name__}.")
        print("Warning: Using module.out/in_features; quant_state.shape preferred.")


    # 5. Retrieve scale tensors and codebook from quant_state
    absmax8 = getattr(quant_state, 'absmax', None)         # uint8, blocksize=64
    codebook = getattr(quant_state, 'code', None)          # float32, size=16
    state2 = getattr(quant_state, 'state2', None)          # Nested state for second level quant
    absmax32 = getattr(state2, 'absmax', None) if state2 else None # float32, blocksize=256

    # Check if all necessary components were found
    if absmax8 is None or codebook is None or absmax32 is None:
        missing = [name for name, val in [('absmax', absmax8), ('code', codebook), ('state2.absmax', absmax32)] if val is None]
        raise AttributeError(f"Missing required quantization state attributes: {', '.join(missing)}.")

    # 6. Call the core decoding function (includes validation)
    out_final = _decode_nf4(
        W_packed,
        absmax8,
        codebook,
        absmax32,
        R=R_orig, C=C_orig, # Pass the *original float* dimensions
        target_dtype=target_dtype
    )

    return out_final