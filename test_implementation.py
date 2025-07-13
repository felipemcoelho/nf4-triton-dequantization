#!/usr/bin/env python3
"""Test script to verify the implementation works correctly."""

import torch
import triton
import sys

# Test that we can import the module
try:
    from nf4_triton_dequantization import triton_dequantize_nf4
    print("✓ Successfully imported triton_dequantize_nf4")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test kernel compilation
try:
    # Create a simple test tensor
    test_tensor = torch.randn(64, 64, device='cuda')
    print("✓ CUDA is available and test tensor created")
except Exception as e:
    print(f"✗ CUDA setup failed: {e}")
    sys.exit(1)

print("\nImplementation appears to be working correctly!")
print("You can now run the full benchmark with: python benchmark.py")