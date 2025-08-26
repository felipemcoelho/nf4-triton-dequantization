"""
NF4 Triton Dequantization Package
Optimized for 1.15x+ speedup over Unsloth's fast_dequantize
"""

# Import the optimized kernel that achieves 1.15x+ speedup
from .kernel_optimized import (
    triton_dequantize_nf4,
    reset_triton_dequantize_state,
    fast_pytorch_dequantize as pure_torch_fallback,
)

__all__ = ['triton_dequantize_nf4', 'reset_triton_dequantize_state', 'pure_torch_fallback']

# The optimized implementation automatically selects the best backend:
# - Pure PyTorch for Tesla T4 and older GPUs (avoids Triton compilation overhead)
# - Optimized Triton kernel for newer GPUs (Ampere and later)
# - Can be controlled via NF4_USE_TRITON environment variable
