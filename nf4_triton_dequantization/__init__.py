from nf4_triton_dequantization.dequantization import (
    triton_dequantize_nf4,
    optimized_triton_dequantize_nf4,
    benchmark_fast_dequantize,
    reset_triton_dequantize_state
)

try:
    from nf4_triton_dequantization.extreme_optimized import extreme_triton_dequantize_nf4
except ImportError:
    extreme_triton_dequantize_nf4 = triton_dequantize_nf4

__all__ = [
    "triton_dequantize_nf4",
    "optimized_triton_dequantize_nf4",
    "benchmark_fast_dequantize",
    "reset_triton_dequantize_state",
    "extreme_triton_dequantize_nf4"
]