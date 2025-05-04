#!/usr/bin/env python
"""
Test script for the optimized Triton dequantization function.

This script benchmarks the optimized Triton dequantization function against
Unsloth's reference implementation to verify that it achieves the target speedup.
"""
import torch
import time
from transformers import set_seed
from bitsandbytes.nn import Linear4bit
from unsloth.kernels.utils import fast_dequantize
from nf4_triton_dequantization import optimized_triton_dequantize_nf4, reset_triton_dequantize_state

def unsloth_dequantize(weight):
    """Wrapper for Unsloth's dequantization function."""
    return fast_dequantize(weight.weight, weight.weight.quant_state)

def benchmark_dequantization(iterations=100, warmup=5):
    """Benchmark the dequantization functions."""
    print("Benchmarking dequantization functions...")
    
    # Test configurations
    configs = [
        (2048, 8192, torch.float16, "fp16 (medium)"),
        (4096, 14336, torch.bfloat16, "bf16 (large)"),
        (1024, 4096, torch.bfloat16, "bf16 (small)"),
    ]
    
    results = []
    
    for hd, m, dtype, name in configs:
        print(f"\nTesting {name} configuration: {hd}x{m} {dtype}")
        
        # Create a test layer
        layer = Linear4bit(
            hd, m, bias=None,
            compute_dtype=dtype,
            compress_statistics=True,
            quant_type="nf4",
        ).to("cuda")
        layer.weight.quant_state.dtype = dtype
        
        # Ensure CUDA is initialized
        torch.cuda.synchronize()
        
        # Reset Triton state
        reset_triton_dequantize_state()
        
        # Warmup
        print("Warming up...")
        for _ in range(warmup):
            # Verify correctness
            triton_output = optimized_triton_dequantize_nf4(layer)
            unsloth_output = unsloth_dequantize(layer)
            
            # Check if outputs match with relaxed tolerance
            rtol = 5e-3 if dtype == torch.bfloat16 else 2e-3
            atol = 5e-3 if dtype == torch.bfloat16 else 2e-3
            
            if not torch.allclose(triton_output, unsloth_output, rtol=rtol, atol=atol):
                abs_diff = (triton_output - unsloth_output).abs()
                max_abs_diff = abs_diff.max().item()
                max_abs_idx = abs_diff.argmax().item()
                row_idx, col_idx = max_abs_idx // m, max_abs_idx % m
                
                print(f"Warning: Outputs don't match exactly. Max diff: {max_abs_diff} at ({row_idx}, {col_idx})")
                print(f"Triton: {triton_output[row_idx, col_idx].item()}, Unsloth: {unsloth_output[row_idx, col_idx].item()}")
            else:
                print("Outputs match within tolerance.")
        
        # Benchmark Unsloth
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            unsloth_output = unsloth_dequantize(layer)
            torch.cuda.synchronize()
        unsloth_time = time.time() - start
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            triton_output = optimized_triton_dequantize_nf4(layer)
            torch.cuda.synchronize()
        triton_time = time.time() - start
        
        # Calculate speedup
        speedup = unsloth_time / triton_time
        
        print(f"Unsloth time: {unsloth_time:.4f}s")
        print(f"Triton time: {triton_time:.4f}s")
        print(f"Speedup: {speedup:.4f}x")
        
        if speedup >= 1.15:
            print("✅ Target speedup of 1.15x achieved!")
        else:
            print(f"❌ Target speedup not reached: {speedup:.4f}x")
        
        results.append((name, unsloth_time, triton_time, speedup))
    
    # Print summary
    print("\nSummary:")
    print("-" * 60)
    print(f"{'Configuration':<15} | {'Unsloth (s)':<12} | {'Triton (s)':<12} | {'Speedup':<10}")
    print("-" * 60)
    
    for name, unsloth_time, triton_time, speedup in results:
        status = "✅" if speedup >= 1.15 else "❌"
        print(f"{name:<15} | {unsloth_time:<12.4f} | {triton_time:<12.4f} | {speedup:<10.4f} {status}")
    
    # Calculate average speedup
    avg_speedup = sum(s for _, _, _, s in results) / len(results)
    print("-" * 60)
    print(f"Average speedup: {avg_speedup:.4f}x")
    
    if avg_speedup >= 1.15:
        print("✅ Overall target speedup of 1.15x achieved!")
    else:
        print(f"❌ Overall target speedup not reached: {avg_speedup:.4f}x")

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    # Run benchmark
    benchmark_dequantization()