"""NF4 Triton Dequantization Package"""

from nf4_triton_dequantization.kernel import triton_dequantize_nf4

# Aliases for compatibility
optimized_triton_dequantize_nf4 = triton_dequantize_nf4
benchmark_fast_dequantize = triton_dequantize_nf4
extreme_triton_dequantize_nf4 = triton_dequantize_nf4

def reset_triton_dequantize_state():
    """Reset any cached state (not needed in simplified version)"""
    pass

__all__ = [
    "triton_dequantize_nf4",
    "optimized_triton_dequantize_nf4",
    "benchmark_fast_dequantize",
    "reset_triton_dequantize_state",
    "extreme_triton_dequantize_nf4"
]