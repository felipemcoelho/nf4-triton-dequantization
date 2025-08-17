#!/usr/bin/env python3
"""
Run all diagnostics to identify performance issues
Execute this after installing the package
"""

import sys
import torch
import time
import triton
import triton.language as tl

print("="*60)
print("NF4 TRITON DEQUANTIZATION DIAGNOSTICS")
print("="*60)

# 1. Environment Check
print("\n1. ENVIRONMENT CHECK")
print("-"*40)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Triton version: {triton.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    device = torch.cuda.get_device_properties(0)
    print(f"GPU: {device.name}")
    print(f"Compute capability: {device.major}.{device.minor}")
    print(f"SMs: {device.multi_processor_count}")
    print(f"Memory: {device.total_memory / 1024**3:.1f} GB")

# 2. Simple Triton Test
print("\n2. TRITON BASIC FUNCTIONALITY TEST")
print("-"*40)

@triton.jit
def simple_add(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

try:
    n = 1024
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')
    output = torch.empty_like(x)
    
    # First run (includes compilation)
    start = time.time()
    simple_add[(1,)](x, y, output, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    first_time = time.time() - start
    
    # Subsequent runs
    start = time.time()
    for _ in range(100):
        simple_add[(1,)](x, y, output, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    run_time = time.time() - start
    
    print(f"✅ Triton works")
    print(f"  First run (with compilation): {first_time*1000:.2f}ms")
    print(f"  100 runs: {run_time*1000:.2f}ms ({run_time*10:.2f}ms per run)")
    
    if first_time > 1.0:
        print("  ⚠️ High compilation overhead detected")
    if run_time/100 > 0.001:
        print("  ⚠️ Simple kernel is slow")
        
except Exception as e:
    print(f"❌ Triton failed: {e}")

# 3. NF4 Performance Test
print("\n3. NF4 DEQUANTIZATION PERFORMANCE TEST")
print("-"*40)

try:
    from nf4_triton_dequantization import triton_dequantize_nf4
    from unsloth.kernels.utils import fast_dequantize
    from bitsandbytes.nn import Linear4bit
    
    # Create test weight
    def create_weight(m=1024, n=1024):
        weight = Linear4bit(
            n, m, bias=None,
            compute_dtype=torch.float16,
            compress_statistics=True,
            quant_type="nf4",
        ).to("cuda")
        weight.weight.quant_state.dtype = torch.float16
        return weight
    
    weight = create_weight(1024, 1024)
    
    # Test Unsloth
    start = time.time()
    result_unsloth = fast_dequantize(weight.weight, weight.weight.quant_state)
    torch.cuda.synchronize()
    unsloth_first = time.time() - start
    
    start = time.time()
    for _ in range(10):
        result_unsloth = fast_dequantize(weight.weight, weight.weight.quant_state)
    torch.cuda.synchronize()
    unsloth_time = time.time() - start
    
    # Test Triton
    start = time.time()
    result_triton = triton_dequantize_nf4(weight)
    torch.cuda.synchronize()
    triton_first = time.time() - start
    
    start = time.time()
    for _ in range(10):
        result_triton = triton_dequantize_nf4(weight)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    # Check correctness
    correct = torch.allclose(result_unsloth, result_triton, rtol=0.1, atol=0.1)
    
    print(f"Correctness: {'✅' if correct else '❌'}")
    print(f"\nUnsloth:")
    print(f"  First: {unsloth_first*1000:.2f}ms")
    print(f"  10 runs: {unsloth_time*1000:.2f}ms ({unsloth_time*100:.2f}ms avg)")
    print(f"\nTriton:")
    print(f"  First: {triton_first*1000:.2f}ms")
    print(f"  10 runs: {triton_time*1000:.2f}ms ({triton_time*100:.2f}ms avg)")
    print(f"\nSpeedup: {unsloth_time/triton_time:.4f}x")
    
    if triton_time > unsloth_time * 10:
        print("\n⚠️ SEVERE PERFORMANCE ISSUE DETECTED")
        print("Triton is >10x slower than expected")
        
        # Try pure PyTorch fallback
        print("\n4. TESTING PURE PYTORCH FALLBACK")
        print("-"*40)
        
        from nf4_triton_dequantization.kernel import pure_torch_fallback
        
        start = time.time()
        result_pytorch = pure_torch_fallback(weight)
        torch.cuda.synchronize()
        pytorch_first = time.time() - start
        
        start = time.time()
        for _ in range(10):
            result_pytorch = pure_torch_fallback(weight)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start
        
        pytorch_correct = torch.allclose(result_unsloth, result_pytorch, rtol=0.1, atol=0.1)
        
        print(f"Correctness: {'✅' if pytorch_correct else '❌'}")
        print(f"PyTorch fallback:")
        print(f"  First: {pytorch_first*1000:.2f}ms")
        print(f"  10 runs: {pytorch_time*1000:.2f}ms ({pytorch_time*100:.2f}ms avg)")
        print(f"  Speedup vs Unsloth: {unsloth_time/pytorch_time:.4f}x")
        
        if unsloth_time/pytorch_time >= 1.15:
            print(f"\n🎉 PyTorch fallback achieves {unsloth_time/pytorch_time:.2f}x speedup!")
            print("Consider using pure_torch_fallback instead of Triton")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure to run 'pip install -e .' first")
except Exception as e:
    print(f"❌ Test failed: {e}")

# 5. Diagnosis Summary
print("\n" + "="*60)
print("DIAGNOSIS SUMMARY")
print("="*60)

if 'triton_time' in locals() and 'unsloth_time' in locals():
    slowdown = triton_time / unsloth_time
    if slowdown > 10:
        print("🔴 CRITICAL: Triton has severe performance issues")
        print("Possible causes:")
        print("  1. Triton recompiling kernel on every call")
        print("  2. Incompatible Triton version with GPU")
        print("  3. CUDA/driver compatibility issues")
        print("\nRecommended actions:")
        print("  1. Use pure_torch_fallback function instead")
        print("  2. Update Triton: pip install -U triton")
        print("  3. Check CUDA compatibility")
    elif slowdown > 1:
        print("🟡 WARNING: Triton is slower than expected")
        print(f"Current: {1/slowdown:.2f}x speedup (target: 1.15x)")
    else:
        print("🟢 SUCCESS: Triton performs well")
        print(f"Achieved: {1/slowdown:.2f}x speedup")