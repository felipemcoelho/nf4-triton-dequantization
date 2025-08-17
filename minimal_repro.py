"""
Minimal reproducible example to isolate Triton performance issue
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np

print("=== Minimal Triton Performance Test ===")
print(f"Triton version: {triton.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Simplest possible Triton kernel
@triton.jit
def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)

def test_triton_vs_torch():
    """Compare Triton copy vs torch.clone()"""
    sizes = [1024, 4096, 16384, 65536]
    
    for size in sizes:
        print(f"\nSize: {size}")
        x = torch.randn(size, device='cuda')
        y_triton = torch.empty_like(x)
        
        # Triton copy
        BLOCK_SIZE = min(1024, size)
        grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Warmup
        copy_kernel[(grid,)](x, y_triton, size, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()
        
        # Time Triton
        start = time.time()
        for _ in range(100):
            copy_kernel[(grid,)](x, y_triton, size, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()
        triton_time = time.time() - start
        
        # Time PyTorch
        start = time.time()
        for _ in range(100):
            y_torch = x.clone()
        torch.cuda.synchronize()
        torch_time = time.time() - start
        
        print(f"  Triton: {triton_time*1000:.2f}ms")
        print(f"  PyTorch: {torch_time*1000:.2f}ms")
        print(f"  Ratio: {triton_time/torch_time:.2f}x")

if __name__ == "__main__":
    test_triton_vs_torch()