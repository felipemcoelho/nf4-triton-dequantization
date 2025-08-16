"""
Diagnostic script to identify Triton performance issues
"""

import torch
import time
from nf4_triton_dequantization import triton_dequantize_nf4
from unsloth.kernels.utils import fast_dequantize
from bitsandbytes.nn import Linear4bit

def create_test_weight(m=1024, n=1024, dtype=torch.float16):
    """Create a test Linear4bit weight."""
    weight = Linear4bit(
        n, m, bias=None,
        compute_dtype=dtype,
        compress_statistics=True,
        quant_type="nf4",
    ).to("cuda")
    weight.weight.quant_state.dtype = dtype
    return weight

def test_single_dequantize():
    """Test a single dequantization to see timing."""
    print("Testing single dequantization...")
    
    weight = create_test_weight(1024, 1024)
    
    # Warmup
    _ = fast_dequantize(weight.weight, weight.weight.quant_state)
    torch.cuda.synchronize()
    
    # Time Unsloth
    start = time.time()
    result_unsloth = fast_dequantize(weight.weight, weight.weight.quant_state)
    torch.cuda.synchronize()
    unsloth_time = time.time() - start
    print(f"Unsloth single call: {unsloth_time*1000:.2f}ms")
    
    # Time Triton (first call - includes compilation)
    start = time.time()
    result_triton = triton_dequantize_nf4(weight)
    torch.cuda.synchronize()
    triton_first_time = time.time() - start
    print(f"Triton first call: {triton_first_time*1000:.2f}ms")
    
    # Time Triton (second call - should be cached)
    start = time.time()
    result_triton = triton_dequantize_nf4(weight)
    torch.cuda.synchronize()
    triton_second_time = time.time() - start
    print(f"Triton second call: {triton_second_time*1000:.2f}ms")
    
    # Check correctness
    if torch.allclose(result_unsloth, result_triton, rtol=0.1, atol=0.1):
        print("✅ Results match")
    else:
        print("❌ Results don't match")
        print(f"Max diff: {(result_unsloth - result_triton).abs().max()}")

def test_multiple_sizes():
    """Test different matrix sizes."""
    print("\nTesting different matrix sizes...")
    
    sizes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]
    
    for m, n in sizes:
        weight = create_test_weight(m, n)
        
        # Warmup
        _ = triton_dequantize_nf4(weight)
        torch.cuda.synchronize()
        
        # Time 10 calls
        start = time.time()
        for _ in range(10):
            _ = triton_dequantize_nf4(weight)
        torch.cuda.synchronize()
        triton_time = time.time() - start
        
        start = time.time()
        for _ in range(10):
            _ = fast_dequantize(weight.weight, weight.weight.quant_state)
        torch.cuda.synchronize()
        unsloth_time = time.time() - start
        
        print(f"Size {m}x{n}: Triton={triton_time*1000:.2f}ms, Unsloth={unsloth_time*1000:.2f}ms, Ratio={triton_time/unsloth_time:.2f}x")

if __name__ == "__main__":
    test_single_dequantize()
    test_multiple_sizes()