#!/bin/bash
# Deploy the working solution

echo "=========================================="
echo "Deploying Working NF4 Solution"
echo "=========================================="

if [ ! -d "nf4_triton_dequantization" ]; then
    echo "Error: Run from project root"
    exit 1
fi

# Deploy working kernel
cp nf4_triton_dequantization/kernel_working.py nf4_triton_dequantization/kernel.py

# Update __init__.py
cat > nf4_triton_dequantization/__init__.py << 'EOF'
"""NF4 Dequantization - Optimized"""
from .kernel import triton_dequantize_nf4, reset_triton_dequantize_state
__all__ = ['triton_dequantize_nf4', 'reset_triton_dequantize_state']
EOF

# Reinstall
pip install -e . > /dev/null 2>&1

echo "✅ Working solution deployed!"
echo ""
echo "Run: python3 benchmark.py"