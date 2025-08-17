"""
Profiling script to identify where time is spent in NF4 dequantization
"""

import torch
import time
import cProfile
import pstats
from io import StringIO
from contextlib import contextmanager
from nf4_triton_dequantization import triton_dequantize_nf4
from unsloth.kernels.utils import fast_dequantize
from bitsandbytes.nn import Linear4bit

@contextmanager
def profile_section(name):
    """Context manager for timing code sections"""
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"{name}: {elapsed*1000:.2f}ms")

def create_test_weight(m=1024, n=1024, dtype=torch.float16):
    """Create a test Linear4bit weight"""
    weight = Linear4bit(
        n, m, bias=None,
        compute_dtype=dtype,
        compress_statistics=True,
        quant_type="nf4",
    ).to("cuda")
    weight.weight.quant_state.dtype = dtype
    return weight

def profile_triton_kernel():
    """Profile the Triton kernel execution"""
    print("=== Profiling NF4 Dequantization ===\n")
    
    weight = create_test_weight(1024, 1024)
    
    # Profile data preparation
    print("1. Data Preparation Phase:")
    with profile_section("  Create weight"):
        weight = create_test_weight(1024, 1024)
    
    with profile_section("  Extract quant_state"):
        qweight = weight.weight.data
        absmax = weight.weight.quant_state.absmax
        absmax32 = weight.weight.quant_state.state2.absmax
        dtype = weight.weight.quant_state.dtype
    
    # Profile Unsloth
    print("\n2. Unsloth Execution:")
    with profile_section("  First run"):
        result_unsloth = fast_dequantize(weight.weight, weight.weight.quant_state)
    
    with profile_section("  10 runs"):
        for _ in range(10):
            result_unsloth = fast_dequantize(weight.weight, weight.weight.quant_state)
    
    # Profile Triton with detailed breakdown
    print("\n3. Triton Execution (Detailed):")
    
    # Import to get access to internals
    from nf4_triton_dequantization.kernel import _nf4_dequantize_kernel
    
    m = weight.out_features
    n = weight.in_features
    blocks_per_row = (n + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    with profile_section("  Tensor reshaping"):
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
    
    with profile_section("  Make contiguous"):
        qweight = qweight.contiguous()
        absmax = absmax.contiguous()
        absmax32 = absmax32.contiguous()
    
    with profile_section("  Allocate output"):
        output = torch.empty((m, n), dtype=dtype, device="cuda")
    
    total_blocks = m * blocks_per_row
    
    with profile_section("  Kernel launch (first)"):
        _nf4_dequantize_kernel[(total_blocks,)](
            qweight.view(-1),
            absmax.view(-1),
            absmax32.view(-1),
            output.view(-1),
            m, n,
            blocks_per_row,
            absmax32_per_row,
            num_warps=2,
            num_stages=2,
        )
    
    with profile_section("  Kernel launch (10 runs)"):
        for _ in range(10):
            _nf4_dequantize_kernel[(total_blocks,)](
                qweight.view(-1),
                absmax.view(-1),
                absmax32.view(-1),
                output.view(-1),
                m, n,
                blocks_per_row,
                absmax32_per_row,
                num_warps=2,
                num_stages=2,
            )
    
    # Profile full function
    print("\n4. Full Function Profiling:")
    
    # Using cProfile for detailed analysis
    profiler = cProfile.Profile()
    
    profiler.enable()
    for _ in range(10):
        result_triton = triton_dequantize_nf4(weight)
    torch.cuda.synchronize()
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())
    
    # Memory profiling
    print("\n5. Memory Usage:")
    print(f"  Weight size: {qweight.element_size() * qweight.nelement() / 1024:.2f} KB")
    print(f"  Output size: {output.element_size() * output.nelement() / 1024:.2f} KB")
    print(f"  Absmax size: {absmax.element_size() * absmax.nelement() / 1024:.2f} KB")
    print(f"  Absmax32 size: {absmax32.element_size() * absmax32.nelement() / 1024:.2f} KB")
    
    # CUDA memory stats
    print(f"\n  GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"  GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == "__main__":
    profile_triton_kernel()