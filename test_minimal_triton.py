"""
Test minimal Triton kernel to isolate performance issues
"""

import torch
import triton
import triton.language as tl
import time

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple addition kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def test_simple_kernel():
    """Test a simple Triton kernel."""
    print("Testing simple Triton kernel...")
    
    n = 1024 * 1024
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')
    output_triton = torch.empty_like(x)
    
    # Grid and block size
    BLOCK_SIZE = 1024
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Warmup
    add_kernel[grid,](x, y, output_triton, n, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    
    # Time Triton kernel
    start = time.time()
    for _ in range(100):
        add_kernel[grid,](x, y, output_triton, n, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    # Time PyTorch
    start = time.time()
    for _ in range(100):
        output_torch = x + y
    torch.cuda.synchronize()
    torch_time = time.time() - start
    
    print(f"Triton: {triton_time*1000:.2f}ms")
    print(f"PyTorch: {torch_time*1000:.2f}ms")
    print(f"Ratio: {triton_time/torch_time:.2f}x")
    
    # Check correctness
    output_torch = x + y
    if torch.allclose(output_triton, output_torch):
        print("✅ Results match")
    else:
        print("❌ Results don't match")

if __name__ == "__main__":
    test_simple_kernel()