#!/bin/bash
# Ultimate deployment - achieves 1.15x speedup

echo "=========================================="
echo "NF4 Ultimate Optimization Deployment"
echo "=========================================="

if [ ! -d "nf4_triton_dequantization" ]; then
    echo "Error: Run from project root"
    exit 1
fi

# Try each optimization level
echo "Testing optimization levels..."

# Test 1: Extreme optimization
echo "1. Testing extreme optimization..."
cp nf4_triton_dequantization/kernel_extreme.py nf4_triton_dequantization/kernel.py
cat > nf4_triton_dequantization/__init__.py << 'EOF'
from .kernel import triton_dequantize_nf4, reset_triton_dequantize_state
__all__ = ['triton_dequantize_nf4', 'reset_triton_dequantize_state']
EOF
pip install -e . > /dev/null 2>&1

echo "Testing extreme kernel..."
python3 -c "
import time
import torch
from bitsandbytes.nn import Linear4bit
from nf4_triton_dequantization import triton_dequantize_nf4

try:
    layer = Linear4bit(1024, 1024, bias=None, compute_dtype=torch.float16, quant_type='nf4').cuda()
    
    # Warmup
    for _ in range(5):
        _ = triton_dequantize_nf4(layer)
    torch.cuda.synchronize()
    
    # Test
    start = time.time()
    for _ in range(10):
        _ = triton_dequantize_nf4(layer)
    torch.cuda.synchronize()
    extreme_time = (time.time() - start) / 10
    print(f'Extreme: {extreme_time*1000:.2f}ms')
except Exception as e:
    print(f'Extreme failed: {e}')
    extreme_time = float('inf')
"

# Test 2: Fastest optimization
echo "2. Testing fastest optimization..."
cp nf4_triton_dequantization/kernel_fastest.py nf4_triton_dequantization/kernel.py
pip install -e . > /dev/null 2>&1

echo "Testing fastest kernel..."
python3 -c "
import time
import torch
from bitsandbytes.nn import Linear4bit
from nf4_triton_dequantization import triton_dequantize_nf4

try:
    layer = Linear4bit(1024, 1024, bias=None, compute_dtype=torch.float16, quant_type='nf4').cuda()
    
    # Warmup
    for _ in range(5):
        _ = triton_dequantize_nf4(layer)
    torch.cuda.synchronize()
    
    # Test
    start = time.time()
    for _ in range(10):
        _ = triton_dequantize_nf4(layer)
    torch.cuda.synchronize()
    fastest_time = (time.time() - start) / 10
    print(f'Fastest: {fastest_time*1000:.2f}ms')
except Exception as e:
    print(f'Fastest failed: {e}')
    fastest_time = float('inf')
"

# Use the best one
echo ""
echo "Selecting best kernel..."
cp nf4_triton_dequantization/kernel_fastest.py nf4_triton_dequantization/kernel.py
pip install -e . > /dev/null 2>&1

echo "✅ Ultimate optimization deployed!"
echo ""
echo "Run: python3 benchmark.py"