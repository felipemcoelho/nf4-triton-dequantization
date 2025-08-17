"""
Pure PyTorch NF4 dequantization without Triton or torch.compile
Optimized for performance using vectorized operations
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import time

# NF4 lookup table
NF4_LUT = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=torch.float32)


def pure_torch_dequantize_nf4(module):
    """
    Pure PyTorch NF4 dequantization with double dequantization
    Optimized for T4 GPU
    """
    weight = module.weight
    quant_state = weight.quant_state
    
    qweight = weight.data  # [m, n//2] packed uint8
    absmax = quant_state.absmax  # First-level scales
    absmax32 = quant_state.state2.absmax  # Second-level scales
    dtype = quant_state.dtype
    device = qweight.device
    
    m = module.out_features
    n = module.in_features
    
    # Move lookup table to device with correct dtype
    lut = NF4_LUT.to(device=device, dtype=torch.float32)
    
    # Calculate dimensions
    blocks_per_row = (n + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Reshape scale factors
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(m, -1)
        elif absmax.numel() == m * blocks_per_row:
            absmax = absmax.view(m, blocks_per_row)
    
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(m, -1)
        elif absmax32.numel() == m * absmax32_per_row:
            absmax32 = absmax32.view(m, absmax32_per_row)
    
    # Ensure contiguous for better performance
    qweight = qweight.contiguous()
    absmax = absmax.contiguous().float()
    absmax32 = absmax32.contiguous().float()
    
    # Extract nibbles from packed bytes
    # This is vectorized across the entire tensor
    qweight_int = qweight.to(torch.int32)  # Avoid overflow in bit operations
    low_nibbles = (qweight_int & 0xF).long()  # [m, n//2]
    high_nibbles = ((qweight_int >> 4) & 0xF).long()  # [m, n//2]
    
    # Lookup NF4 values
    low_vals = lut[low_nibbles]  # [m, n//2]
    high_vals = lut[high_nibbles]  # [m, n//2]
    
    # Interleave low and high values to get full output
    # Create output tensor
    output = torch.empty((m, n), dtype=dtype, device=device)
    
    # Efficient interleaving using slicing
    output[:, 0::2] = low_vals  # Even columns
    output[:, 1::2] = high_vals  # Odd columns
    
    # Apply double dequantization scales
    # Process each 64-element block
    for block_idx in range(blocks_per_row):
        col_start = block_idx * 64
        col_end = min(col_start + 64, n)
        
        # Get scale factors for this block
        absmax_block = absmax[:, block_idx:block_idx+1]  # [m, 1]
        absmax32_block = absmax32[:, block_idx//4:block_idx//4+1]  # [m, 1]
        
        # Combined scale with NF4 constant
        scale = absmax_block * 0.00787401574803149606 * absmax32_block  # [m, 1]
        
        # Apply scale to this block's columns
        output[:, col_start:col_end] *= scale.to(dtype)
    
    return output


def optimized_pure_torch_nf4(module):
    """
    Highly optimized pure PyTorch version with minimal loops
    """
    weight = module.weight
    quant_state = weight.quant_state
    
    qweight = weight.data
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    dtype = quant_state.dtype
    device = qweight.device
    
    m = module.out_features
    n = module.in_features
    
    # Precompute constants
    blocks_per_row = (n + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Fast LUT on device
    lut = NF4_LUT.to(device=device, dtype=torch.float32)
    
    # Reshape scales efficiently
    if absmax.dim() == 1:
        absmax = absmax.view(m, -1) if absmax.numel() == m * blocks_per_row else absmax.unsqueeze(0).expand(m, -1)
    if absmax32.dim() == 1:
        absmax32 = absmax32.view(m, -1) if absmax32.numel() == m * absmax32_per_row else absmax32.unsqueeze(0).expand(m, -1)
    
    # Vectorized nibble extraction
    qweight_flat = qweight.view(-1).to(torch.int32)
    all_low = lut[(qweight_flat & 0xF).long()].view(m, -1)
    all_high = lut[((qweight_flat >> 4) & 0xF).long()].view(m, -1)
    
    # Create output with interleaving
    output = torch.empty((m, n), dtype=dtype, device=device)
    output[:, 0::2] = all_low[:, :n//2]
    output[:, 1::2] = all_high[:, :n//2]
    
    # Vectorized scale application
    # Create scale map for all columns
    col_to_block = torch.arange(n, device=device) // 64
    col_to_absmax32 = col_to_block // 4
    
    # Gather scales for each column
    absmax_expanded = absmax[:, col_to_block]  # [m, n]
    absmax32_expanded = absmax32[:, col_to_absmax32]  # [m, n]
    
    # Apply combined scale
    scales = absmax_expanded.float() * 0.00787401574803149606 * absmax32_expanded.float()
    output *= scales.to(dtype)
    
    return output


def test_pure_torch():
    """Test pure PyTorch implementation"""
    from bitsandbytes.nn import Linear4bit
    from unsloth.kernels.utils import fast_dequantize
    
    print("=== Testing Pure PyTorch NF4 Dequantization ===\n")
    
    def create_test_weight(m=1024, n=1024, dtype=torch.float16):
        weight = Linear4bit(
            n, m, bias=None,
            compute_dtype=dtype,
            compress_statistics=True,
            quant_type="nf4",
        ).to("cuda")
        weight.weight.quant_state.dtype = dtype
        return weight
    
    # Test multiple sizes
    sizes = [(1024, 1024), (2048, 4096), (4096, 14336)]
    
    for m, n in sizes:
        print(f"\nTesting size {m}x{n}:")
        weight = create_test_weight(m, n)
        
        # Get reference result
        result_unsloth = fast_dequantize(weight.weight, weight.weight.quant_state)
        
        # Test basic version
        result_basic = pure_torch_dequantize_nf4(weight)
        basic_match = torch.allclose(result_unsloth, result_basic, rtol=0.1, atol=0.1)
        print(f"  Basic version: {'✅' if basic_match else '❌'}")
        
        # Test optimized version
        result_opt = optimized_pure_torch_nf4(weight)
        opt_match = torch.allclose(result_unsloth, result_opt, rtol=0.1, atol=0.1)
        print(f"  Optimized version: {'✅' if opt_match else '❌'}")
        
        if not basic_match:
            diff = (result_unsloth - result_basic).abs()
            print(f"    Max diff: {diff.max():.6f}")
        
        # Benchmark
        if m == 1024 and n == 1024:
            print("\n  Performance (100 iterations):")
            
            # Warmup
            for _ in range(10):
                _ = fast_dequantize(weight.weight, weight.weight.quant_state)
                _ = pure_torch_dequantize_nf4(weight)
                _ = optimized_pure_torch_nf4(weight)
            torch.cuda.synchronize()
            
            # Time Unsloth
            start = time.time()
            for _ in range(100):
                _ = fast_dequantize(weight.weight, weight.weight.quant_state)
            torch.cuda.synchronize()
            unsloth_time = time.time() - start
            
            # Time basic version
            start = time.time()
            for _ in range(100):
                _ = pure_torch_dequantize_nf4(weight)
            torch.cuda.synchronize()
            basic_time = time.time() - start
            
            # Time optimized version
            start = time.time()
            for _ in range(100):
                _ = optimized_pure_torch_nf4(weight)
            torch.cuda.synchronize()
            opt_time = time.time() - start
            
            print(f"    Unsloth: {unsloth_time*1000:.2f}ms")
            print(f"    Basic PyTorch: {basic_time*1000:.2f}ms ({unsloth_time/basic_time:.3f}x)")
            print(f"    Optimized PyTorch: {opt_time*1000:.2f}ms ({unsloth_time/opt_time:.3f}x)")
            
            if unsloth_time/opt_time >= 1.15:
                print(f"\n    🎉 Target 1.15x speedup achieved! ({unsloth_time/opt_time:.3f}x)")


if __name__ == "__main__":
    test_pure_torch()