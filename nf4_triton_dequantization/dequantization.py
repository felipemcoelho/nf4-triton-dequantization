import torch
import triton
import triton.language as tl
import math

@triton.jit
def _dequantize_nf4_kernel(
    # Input pointers
    weight_ptr,  # [R, C//2] uint8 packed weights
    absmax8_ptr, # [R, C//64] uint8 first-level absmax
    codes_ptr,   # [16] float32 codebook
    absmax32_ptr,# float32 second-level absmax (potentially reshaped)
    # Output pointer
    output_ptr,  # [R, C] output dequantized weights
    # Dimensions
    R, C,        # Original weight tensor dimensions (before packing)
    # Strides for addressing
    weight_row_stride, weight_col_stride,
    absmax8_row_stride, absmax8_col_stride,
    absmax32_row_stride, absmax32_col_stride, # Strides for the potentially reshaped absmax32
    output_row_stride, output_col_stride,
    # Constants
    absmax8_block_size: tl.constexpr, # Should be 64
    absmax32_block_size: tl.constexpr, # Should be 256
    absmax32_rows_per_block: tl.constexpr, # How many rows share one absmax32 value
    BLOCK_SIZE: tl.constexpr, # Processing block size
):
    # Get block indices
    row_idx = tl.program_id(0)  # Current row
    col_block_idx = tl.program_id(1)  # Current block of columns
    
    # Calculate starting column index for this program instance
    start_col = col_block_idx * BLOCK_SIZE
    
    # Create a range for column indices in this block
    cols = start_col + tl.arange(0, BLOCK_SIZE)
    mask = cols < C  # Mask out columns beyond the tensor boundary
    
    # Get pointers for current row
    row_weight_ptr = weight_ptr + row_idx * weight_row_stride
    row_absmax8_ptr = absmax8_ptr + row_idx * absmax8_row_stride
    row_output_ptr = output_ptr + row_idx * output_row_stride
    
    # --- Load absmax8 scale --- 
    absmax8_block = cols // absmax8_block_size
    # Load uint8 absmax value - rely on mask, default 0 if out of bounds
    absmax8_ptr_offset = absmax8_block * absmax8_col_stride
    absmax8 = tl.load(row_absmax8_ptr + absmax8_ptr_offset, mask=mask)
    
    # --- Load absmax32 scale using its own strides --- 
    # Calculate target row/col in the logical absmax32 tensor
    absmax32_target_row = row_idx // absmax32_rows_per_block
    absmax32_target_col = cols // absmax32_block_size
    # Calculate pointer offset using absmax32 strides passed from host
    # Note: Bounds for absmax32_target_col must be implicitly handled by host reshape/masking
    absmax32_ptr_offset = absmax32_target_row * absmax32_row_stride + absmax32_target_col * absmax32_col_stride
    absmax32 = tl.load(absmax32_ptr + absmax32_ptr_offset, mask=mask) # Rely on mask, default 0.0
    
    # --- Unpack nibbles --- 
    byte_idx = cols // 2
    nibble_pos = cols % 2
    weight_ptr_offset = byte_idx * weight_col_stride
    packed_bytes = tl.load(row_weight_ptr + weight_ptr_offset, mask=mask)
    nibbles = tl.where(
        nibble_pos == 0,
        packed_bytes & 0x0F,
        (packed_bytes >> 4) & 0x0F
    )
    
    # --- Codebook lookup --- 
    valid_nibbles = tl.minimum(tl.maximum(nibbles, 0), 15)
    code_values = tl.load(codes_ptr + valid_nibbles, mask=mask)
    
    # --- Final Scale Calculation --- 
    scale = (tl.cast(absmax8, tl.float32) / 127.0) * absmax32
    
    # Apply scaling
    dequantized = code_values * scale
    
    # Store results
    output_offset = cols * output_col_stride
    tl.store(row_output_ptr + output_offset, dequantized, mask=mask)

def triton_dequantize_nf4(module):
    """
    Entry point for dequantizing a bnb.Linear4bit NF4 layer.
    Retrieves tensors from the module's quant_state and calls the Triton kernel.
    """
    # Extract weight and quant_state
    weight = getattr(module.weight, 'data', module.weight)
    quant_state = getattr(module, 'quant_state', None)
    
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 weight tensor, got {type(weight)} with dtype {getattr(weight, 'dtype', None)}")
    
    if quant_state is None:
        raise AttributeError(f"Module {type(module).__name__} lacks required 'quant_state'.")
    
    # Get dimensions and data from quant_state
    target_dtype = getattr(quant_state, 'dtype', torch.float16)
    if target_dtype not in [torch.float16, torch.bfloat16]:
        print(f"Warning: Unusual target dtype {target_dtype} found in quant_state.")
    
    # Original tensor dimensions
    shape = getattr(quant_state, 'shape', None)
    if shape is not None and len(shape) == 2:
        rows, cols = shape
    else:
        rows = getattr(module, 'out_features', None)
        cols = getattr(module, 'in_features', None)
        if rows is None or cols is None:
            raise ValueError(f"Cannot determine dimensions for {type(module).__name__}")
    
    # Get quantization parameters
    absmax8 = getattr(quant_state, 'absmax', None)  # First-level quantization (uint8)
    codes = getattr(quant_state, 'code', None)     # Codebook (float32)
    state2 = getattr(quant_state, 'state2', None)
    absmax32 = getattr(state2, 'absmax', None) if state2 else None  # Second-level quantization (float32)
    
    # Validate all tensors
    if absmax8 is None or codes is None or absmax32 is None:
        raise ValueError("Missing required quantization tensors")
    
    # Block sizes
    absmax8_blocksize = getattr(quant_state, 'blocksize', 64)
    absmax32_blocksize = getattr(state2, 'blocksize', 256) if state2 else 256
    
    # --- Reshape absmax8 --- 
    logical_a8_shape = (rows, math.ceil(cols / absmax8_blocksize))
    if absmax8.shape != logical_a8_shape:
        if absmax8.numel() == math.prod(logical_a8_shape):
            absmax8 = absmax8.reshape(logical_a8_shape)
        else:
            raise ValueError(
                f"absmax8 shape mismatch. Expected {logical_a8_shape} ({math.prod(logical_a8_shape)} elements), "
                f"got {absmax8.shape} ({absmax8.numel()} elements)."
            )
            
    # --- Calculate absmax32 row sharing and reshape --- 
    num_abs32_cols_logical = math.ceil(cols / absmax32_blocksize)
    absmax32_rows_per_block = 1 # Default to no sharing
    if absmax32.numel() > 0 and rows > 0 and num_abs32_cols_logical > 0:
        if absmax32.numel() % num_abs32_cols_logical == 0:
            num_abs32_rows_stored = absmax32.numel() // num_abs32_cols_logical
            if num_abs32_rows_stored > 0 and rows > num_abs32_rows_stored:
                 if rows % num_abs32_rows_stored == 0:
                     absmax32_rows_per_block = rows // num_abs32_rows_stored
                 else:
                      print(f"Warning: Ambiguous absmax32 sharing. rows ({rows}) not divisible by stored rows ({num_abs32_rows_stored}). Assuming no sharing.")
        else:
             print(f"Warning: absmax32 numel ({absmax32.numel()}) not divisible by logical cols ({num_abs32_cols_logical}). Assuming no row sharing.")

    logical_a32_shape = (math.ceil(rows / absmax32_rows_per_block), num_abs32_cols_logical)
    if absmax32.shape != logical_a32_shape:
        if absmax32.numel() == math.prod(logical_a32_shape):
            absmax32 = absmax32.reshape(logical_a32_shape)
        else:
             if absmax32.numel() == math.prod(logical_a32_shape):
                 absmax32 = absmax32.flatten() 
             else:
                 raise ValueError(
                    f"absmax32 shape mismatch. Expected {logical_a32_shape} or flat ({math.prod(logical_a32_shape)} elements), "
                    f"got {absmax32.shape} ({absmax32.numel()} elements)."
                )

    # Make sure all tensors are contiguous for reliable stride calculations
    weight = weight.contiguous()
    absmax8 = absmax8.contiguous() 
    codes = codes.contiguous()
    absmax32 = absmax32.contiguous()
    
    # --- Kernel Launch --- 
    BLOCK_SIZE = 128  # Process this many columns per program instance
    grid = (rows, triton.cdiv(cols, BLOCK_SIZE))
    
    output = torch.empty((rows, cols), dtype=torch.float32, device=weight.device)
    
    # Get strides for absmax32 *after* potential reshape/flattening
    absmax32_stride_0 = absmax32.stride(0) if absmax32.dim() > 1 else absmax32.stride(0) # Use stride(0) if flat
    absmax32_stride_1 = absmax32.stride(1) if absmax32.dim() > 1 else 0 # Use 0 if flat

    _dequantize_nf4_kernel[grid](
        weight, absmax8, codes, absmax32, output,
        rows, cols,
        weight.stride(0), weight.stride(1),
        absmax8.stride(0), absmax8.stride(1),
        absmax32_stride_0, absmax32_stride_1, # Pass potentially updated strides
        output.stride(0), output.stride(1),
        absmax8_blocksize, absmax32_blocksize, absmax32_rows_per_block,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.to(target_dtype)