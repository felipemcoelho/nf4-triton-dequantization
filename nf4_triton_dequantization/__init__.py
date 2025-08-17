"""NF4 Triton Dequantization Package"""

import os
from nf4_triton_dequantization.kernel import (
    triton_dequantize_nf4 as _triton_impl,
    reset_triton_dequantize_state,
    pure_torch_fallback as _pytorch_impl
)

# Check if we should use fallback based on environment variable
USE_FALLBACK = os.environ.get('NF4_USE_PYTORCH_FALLBACK', '').lower() in ('1', 'true', 'yes')

# Auto-detect if Triton is slow (can be set by diagnostics)
AUTO_FALLBACK = False

def auto_select_implementation():
    """Auto-select best implementation based on quick benchmark"""
    global AUTO_FALLBACK
    try:
        import torch
        import time
        from bitsandbytes.nn import Linear4bit
        
        # Create small test case
        weight = Linear4bit(
            256, 256, bias=None,
            compute_dtype=torch.float16,
            compress_statistics=True,
            quant_type="nf4",
        ).to("cuda")
        weight.weight.quant_state.dtype = torch.float16
        
        # Test Triton
        torch.cuda.synchronize()
        start = time.time()
        _ = _triton_impl(weight)
        torch.cuda.synchronize()
        triton_time = time.time() - start
        
        # Test PyTorch
        torch.cuda.synchronize()
        start = time.time()
        _ = _pytorch_impl(weight)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start
        
        # Use PyTorch if it's significantly faster
        if triton_time > pytorch_time * 2:
            AUTO_FALLBACK = True
            print(f"Auto-selected PyTorch fallback (Triton: {triton_time*1000:.1f}ms, PyTorch: {pytorch_time*1000:.1f}ms)")
            return _pytorch_impl
        else:
            return _triton_impl
    except:
        # If auto-detection fails, use Triton
        return _triton_impl

# Select implementation
if USE_FALLBACK:
    print("Using PyTorch fallback for NF4 dequantization (NF4_USE_PYTORCH_FALLBACK=1)")
    triton_dequantize_nf4 = _pytorch_impl
elif os.environ.get('NF4_AUTO_SELECT', '').lower() in ('1', 'true', 'yes'):
    triton_dequantize_nf4 = auto_select_implementation()
else:
    triton_dequantize_nf4 = _triton_impl

# Aliases for compatibility
optimized_triton_dequantize_nf4 = triton_dequantize_nf4
benchmark_fast_dequantize = triton_dequantize_nf4
extreme_triton_dequantize_nf4 = triton_dequantize_nf4

__all__ = [
    "triton_dequantize_nf4",
    "optimized_triton_dequantize_nf4",
    "benchmark_fast_dequantize",
    "reset_triton_dequantize_state",
    "extreme_triton_dequantize_nf4"
]