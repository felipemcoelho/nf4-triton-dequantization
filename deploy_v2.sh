#!/bin/bash
# Deploy V2 optimized solution

echo "=========================================="
echo "Deploying V2 Optimized Solution"
echo "=========================================="

if [ ! -d "nf4_triton_dequantization" ]; then
    echo "Error: Run from project root"
    exit 1
fi

# First try the working solution
echo "1. Testing working solution..."
cp nf4_triton_dequantization/kernel_working.py nf4_triton_dequantization/kernel.py
pip install -e . > /dev/null 2>&1

python3 -c "
import torch
from bitsandbytes.nn import Linear4bit
from nf4_triton_dequantization import triton_dequantize_nf4
try:
    layer = Linear4bit(256, 256, bias=None, compute_dtype=torch.float16, quant_type='nf4').cuda()
    result = triton_dequantize_nf4(layer)
    print('✅ Working solution OK')
except Exception as e:
    print(f'❌ Working solution failed: {e}')
"

# Now try V2
echo "2. Testing V2 optimized..."
cp nf4_triton_dequantization/kernel_v2.py nf4_triton_dequantization/kernel.py
pip install -e . > /dev/null 2>&1

python3 -c "
import torch
from bitsandbytes.nn import Linear4bit
from nf4_triton_dequantization import triton_dequantize_nf4
try:
    layer = Linear4bit(256, 256, bias=None, compute_dtype=torch.float16, quant_type='nf4').cuda()
    result = triton_dequantize_nf4(layer)
    print('✅ V2 solution OK')
    USE_V2=true
except Exception as e:
    print(f'⚠️ V2 failed, using working solution: {e}')
    USE_V2=false
"

# Use the best working version
if [ "$USE_V2" = "true" ]; then
    echo "Using V2 optimized kernel"
else
    echo "Using stable working kernel"
    cp nf4_triton_dequantization/kernel_working.py nf4_triton_dequantization/kernel.py
    pip install -e . > /dev/null 2>&1
fi

echo ""
echo "✅ Deployment complete!"
echo "Run: python3 benchmark.py"