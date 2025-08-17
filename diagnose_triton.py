"""
Diagnostic script to identify Triton performance issues
"""

import torch
import triton
import triton.language as tl
import time
from nf4_triton_dequantization import triton_dequantize_nf4
from unsloth.kernels.utils import fast_dequantize
from bitsandbytes.nn import Linear4bit

print("=== Triton Environment Diagnostics ===")
print(f"Triton version: {triton.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Device compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"Number of SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")

# Test kernel compilation
@triton.jit
def simple_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

print("\n=== Testing Simple Triton Kernel ===")
n = 1024
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')
output = torch.empty_like(x)

BLOCK_SIZE = 1024
grid = (1,)

# Time compilation
start = time.time()
simple_add_kernel[grid,](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
torch.cuda.synchronize()
compile_time = time.time() - start
print(f"First run (with compilation): {compile_time*1000:.2f}ms")

# Time execution
start = time.time()
for _ in range(100):
    simple_add_kernel[grid,](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
torch.cuda.synchronize()
exec_time = time.time() - start
print(f"100 runs (cached): {exec_time*1000:.2f}ms")

print("\n=== Testing NF4 Dequantization ===")

# Create test weight
def create_test_weight(m=1024, n=1024, dtype=torch.float16):
    weight = Linear4bit(
        n, m, bias=None,
        compute_dtype=dtype,
        compress_statistics=True,
        quant_type="nf4",
    ).to("cuda")
    weight.weight.quant_state.dtype = dtype
    return weight

weight = create_test_weight(1024, 1024)

# Test Unsloth
print("\nUnsloth fast_dequantize:")
start = time.time()
result_unsloth = fast_dequantize(weight.weight, weight.weight.quant_state)
torch.cuda.synchronize()
unsloth_first = time.time() - start
print(f"First run: {unsloth_first*1000:.2f}ms")

start = time.time()
for _ in range(10):
    result_unsloth = fast_dequantize(weight.weight, weight.weight.quant_state)
torch.cuda.synchronize()
unsloth_time = time.time() - start
print(f"10 runs: {unsloth_time*1000:.2f}ms ({unsloth_time*100:.2f}ms per run)")

# Test Triton
print("\nTriton dequantize:")
start = time.time()
result_triton = triton_dequantize_nf4(weight)
torch.cuda.synchronize()
triton_first = time.time() - start
print(f"First run: {triton_first*1000:.2f}ms")

start = time.time()
for _ in range(10):
    result_triton = triton_dequantize_nf4(weight)
torch.cuda.synchronize()
triton_time = time.time() - start
print(f"10 runs: {triton_time*1000:.2f}ms ({triton_time*100:.2f}ms per run)")

# Check correctness
if torch.allclose(result_unsloth, result_triton, rtol=0.1, atol=0.1):
    print("\n✅ Results match")
else:
    print("\n❌ Results don't match")
    diff = (result_unsloth - result_triton).abs()
    print(f"Max diff: {diff.max()}")
    print(f"Mean diff: {diff.mean()}")

print(f"\nSpeedup: {unsloth_time/triton_time:.4f}x")

# Check if compilation is the issue
print("\n=== Analyzing Compilation Overhead ===")
print(f"Triton first run overhead: {(triton_first - unsloth_first)*1000:.2f}ms")
print(f"Triton per-run overhead: {(triton_time/10 - unsloth_time/10)*1000:.2f}ms")

if triton_first > triton_time / 10:
    print("⚠️ High compilation overhead detected")
if triton_time / 10 > unsloth_time / 10 * 2:
    print("⚠️ Execution is slower than expected even after compilation")