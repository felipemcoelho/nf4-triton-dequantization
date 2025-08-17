"""
Alternative NF4 dequantization using torch.compile instead of Triton
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import time

# NF4 lookup table as a constant tensor
NF4_TABLE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=torch.float32)

@torch.compile(mode="max-autotune", fullgraph=True)
def nf4_dequantize_compiled(
    qweight: Tensor,
    absmax: Tensor,
    absmax32: Tensor,
    m: int,
    n: int,
) -> Tensor:
    """
    Compiled NF4 dequantization using torch.compile
    
    Args:
        qweight: Packed 4-bit weights [m, n//2]
        absmax: First-level scale factors
        absmax32: Second-level scale factors
        m: Output rows
        n: Output columns
    
    Returns:
        Dequantized weights [m, n]
    """
    device = qweight.device
    dtype = absmax.dtype
    
    # Move lookup table to device
    nf4_table = NF4_TABLE.to(device)
    
    # Calculate dimensions
    blocks_per_row = (n + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Reshape scale factors if needed
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
    
    # Allocate output
    output = torch.zeros((m, n), dtype=dtype, device=device)
    
    # Process each row
    for row in range(m):
        row_data = qweight[row]
        
        # Process 64-element blocks
        for block_idx in range(blocks_per_row):
            col_start = block_idx * 64
            if col_start >= n:
                break
            
            # Get scale factors (double dequantization)
            absmax_val = absmax[row, block_idx].float()
            absmax32_val = absmax32[row, block_idx // 4].float()
            scale = absmax_val * 0.00787401574803149606 * absmax32_val
            
            # Extract 32 bytes for this block
            byte_start = col_start // 2
            byte_end = min(byte_start + 32, n // 2)
            packed_bytes = row_data[byte_start:byte_end]
            
            # Extract nibbles
            low_nibbles = packed_bytes & 0xF
            high_nibbles = (packed_bytes >> 4) & 0xF
            
            # Lookup NF4 values
            low_vals = nf4_table[low_nibbles]
            high_vals = nf4_table[high_nibbles]
            
            # Apply scale
            low_scaled = low_vals * scale
            high_scaled = high_vals * scale
            
            # Interleave and store
            for i in range(len(packed_bytes)):
                idx_low = col_start + i * 2
                idx_high = idx_low + 1
                
                if idx_low < n:
                    output[row, idx_low] = low_scaled[i]
                if idx_high < n:
                    output[row, idx_high] = high_scaled[i]
    
    return output


def torch_compile_dequantize_nf4(module):
    """
    Main entry point compatible with the existing interface
    """
    weight = module.weight
    quant_state = weight.quant_state
    
    qweight = weight.data
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    
    m = module.out_features
    n = module.in_features
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    return nf4_dequantize_compiled(qweight, absmax, absmax32, m, n)


# Optimized version with vectorized operations
@torch.compile(mode="max-autotune", fullgraph=True)
def nf4_dequantize_vectorized(
    qweight: Tensor,
    absmax: Tensor, 
    absmax32: Tensor,
    m: int,
    n: int,
) -> Tensor:
    """
    Vectorized NF4 dequantization using torch operations
    """
    device = qweight.device
    dtype = torch.float16 if absmax.dtype == torch.uint8 else absmax.dtype
    
    # NF4 lookup table
    nf4_table = NF4_TABLE.to(device).to(dtype)
    
    blocks_per_row = (n + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Handle tensor shapes
    if absmax.dim() == 1:
        absmax = absmax.view(m, -1) if absmax.numel() == m * blocks_per_row else absmax.unsqueeze(0).expand(m, -1)
    if absmax32.dim() == 1:
        absmax32 = absmax32.view(m, -1) if absmax32.numel() == m * absmax32_per_row else absmax32.unsqueeze(0).expand(m, -1)
    
    # Extract all nibbles at once
    qweight_flat = qweight.view(-1)
    low_nibbles = (qweight_flat & 0xF).long()
    high_nibbles = ((qweight_flat >> 4) & 0xF).long()
    
    # Lookup NF4 values
    low_vals = nf4_table[low_nibbles].view(m, -1)
    high_vals = nf4_table[high_nibbles].view(m, -1)
    
    # Create output by interleaving
    output = torch.empty((m, n), dtype=dtype, device=device)
    
    # Interleave low and high values
    output[:, 0::2] = low_vals[:, :n//2]
    output[:, 1::2] = high_vals[:, :n//2]
    
    # Apply scales block by block
    for block_idx in range(blocks_per_row):
        col_start = block_idx * 64
        col_end = min(col_start + 64, n)
        
        # Calculate combined scale
        absmax_val = absmax[:, block_idx:block_idx+1].float()
        absmax32_val = absmax32[:, block_idx//4:block_idx//4+1].float()
        scale = absmax_val * 0.00787401574803149606 * absmax32_val
        
        # Apply scale to this block
        output[:, col_start:col_end] *= scale
    
    return output


def test_torch_compile_solution():
    """Test the torch.compile solution"""
    from bitsandbytes.nn import Linear4bit
    from unsloth.kernels.utils import fast_dequantize
    
    print("=== Testing torch.compile NF4 Dequantization ===\n")
    
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
    
    # Test correctness
    print("1. Correctness Test:")
    result_unsloth = fast_dequantize(weight.weight, weight.weight.quant_state)
    result_compiled = torch_compile_dequantize_nf4(weight)
    
    if torch.allclose(result_unsloth, result_compiled, rtol=0.1, atol=0.1):
        print("✅ Results match")
    else:
        print("❌ Results don't match")
        diff = (result_unsloth - result_compiled).abs()
        print(f"Max diff: {diff.max()}")
    
    # Benchmark
    print("\n2. Performance Test:")
    
    # Warmup
    for _ in range(3):
        _ = fast_dequantize(weight.weight, weight.weight.quant_state)
        _ = torch_compile_dequantize_nf4(weight)
    torch.cuda.synchronize()
    
    # Time Unsloth
    start = time.time()
    for _ in range(100):
        _ = fast_dequantize(weight.weight, weight.weight.quant_state)
    torch.cuda.synchronize()
    unsloth_time = time.time() - start
    
    # Time torch.compile (first run includes compilation)
    start = time.time()
    _ = torch_compile_dequantize_nf4(weight)
    torch.cuda.synchronize()
    compile_first = time.time() - start
    
    # Time torch.compile (subsequent runs)
    start = time.time()
    for _ in range(100):
        _ = torch_compile_dequantize_nf4(weight)
    torch.cuda.synchronize()
    compile_time = time.time() - start
    
    print(f"Unsloth: {unsloth_time*1000:.2f}ms for 100 runs")
    print(f"torch.compile first: {compile_first*1000:.2f}ms")
    print(f"torch.compile 100 runs: {compile_time*1000:.2f}ms")
    print(f"Speedup: {unsloth_time/compile_time:.4f}x")
    
    # Test vectorized version
    print("\n3. Testing Vectorized Version:")
    
    # Override the function
    global nf4_dequantize_compiled
    nf4_dequantize_compiled = nf4_dequantize_vectorized
    
    result_vectorized = torch_compile_dequantize_nf4(weight)
    
    if torch.allclose(result_unsloth, result_vectorized, rtol=0.1, atol=0.1):
        print("✅ Vectorized results match")
    else:
        print("❌ Vectorized results don't match")
    
    # Time vectorized
    start = time.time()
    for _ in range(100):
        _ = torch_compile_dequantize_nf4(weight)
    torch.cuda.synchronize()
    vectorized_time = time.time() - start
    
    print(f"Vectorized 100 runs: {vectorized_time*1000:.2f}ms")
    print(f"Speedup vs Unsloth: {unsloth_time/vectorized_time:.4f}x")


if __name__ == "__main__":
    test_torch_compile_solution()