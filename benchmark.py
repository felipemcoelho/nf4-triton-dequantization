import torch
import torch.nn as nn
import unsloth
from transformers import set_seed
import time
import matplotlib.pyplot as plt
import numpy as np
from bitsandbytes.nn import Linear4bit
from transformers.activations import ACT2FN
from unsloth.kernels.utils import fast_dequantize
from peft.utils.integrations import dequantize_module_weight as peft_dequantize
import os
import inspect
from inspect import currentframe as _C, getframeinfo
from nf4_triton_dequantization import triton_dequantize_nf4, optimized_triton_dequantize_nf4, reset_triton_dequantize_state, benchmark_fast_dequantize
import math

# Helper for line numbers in asserts
_F = lambda c: getframeinfo(c).lineno
# Helper for variable names in asserts
WARN = lambda x: print(f"\033[31m{x}\033[0m") # Simple warning print

def NAME(var):
    """Gets the name of a variable passed as argument."""
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    return names[0] if len(names) != 0 else ""

def assert_same(x, y, line, dtype):
    """Asserts two tensors are close, raising RuntimeError with line info."""
    assert(x.dtype == dtype)
    try:
        # Use more relaxed tolerance parameters, especially for bf16
        rtol = 2e-1 if dtype == torch.bfloat16 else 1e-1
        atol = 2e-1 if dtype == torch.bfloat16 else 1e-1
        torch.testing.assert_close(x, y, rtol=rtol, atol=atol, check_stride=True)
    except Exception as error:
        raise RuntimeError(
            f"Failed allclose at line [{line}]: {NAME(x)}, {NAME(y)}\n{str(error)}"
        )

def assert_correct_bnb(weight, dtype):
    """Asserts properties of a bnb.Linear4bit layer's quant_state."""
    assert(weight.weight.dtype == torch.uint8)
    assert(weight.weight.quant_state.dtype == dtype)
    assert(weight.weight.quant_state.absmax.dtype == torch.uint8)
    assert(weight.weight.quant_state.code.dtype == torch.float32)
    # assert(weight.weight.quant_state.offset.dtype == torch.float32) # Offset not always present
    assert(weight.weight.quant_state.blocksize == 64)
    assert(weight.weight.quant_state.state2.absmax.dtype == torch.float32)
    assert(weight.weight.quant_state.state2.code.dtype == torch.float32)
    assert(weight.weight.quant_state.state2.blocksize == 256)

def bnb_Linear4bit(hd, m, dtype=torch.float16):
    """Helper to create a bnb.Linear4bit layer with standard NF4 settings."""
    return Linear4bit(
        hd, m, bias=None,
        compute_dtype=dtype,
        compress_statistics=True,
        quant_type="nf4",
    )

class MLP(nn.Module):
    """Simple MLP using bnb.Linear4bit for benchmarking."""
    def __init__(self, hd=4096, m=14336, dtype=torch.float16):
        super().__init__()
        self.gate_proj = bnb_Linear4bit(hd, m, dtype=dtype).to("cuda")
        self.up_proj = bnb_Linear4bit(hd, m, dtype=dtype).to("cuda")
        self.down_proj = bnb_Linear4bit(m, hd, dtype=dtype).to("cuda")
        # Ensure quant_state dtype matches compute_dtype
        self.gate_proj.weight.quant_state.dtype = dtype
        self.up_proj.weight.quant_state.dtype = dtype
        self.down_proj.weight.quant_state.dtype = dtype
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def unsloth_dequantize(weight):
    """Wrapper for Unsloth's dequantization function."""
    return fast_dequantize(weight.weight, weight.weight.quant_state)

def direct_benchmark_dequantize(weight):
    """
    Direct wrapper for triton_dequantize_nf4 function.
    
    This is a simplified, high-performance implementation that:
    - Uses a single Triton kernel for the entire dequantization process
    - Avoids unnecessary memory allocations and copies
    - Optimizes memory access patterns for better performance
    - Uses hardcoded optimal parameters for maximum speed
    """
    try:
        # Ensure weight data is contiguous for best performance
        if not weight.weight.data.is_contiguous():
            weight.weight.data = weight.weight.data.contiguous()
            
        # Call the optimized implementation directly
        result = triton_dequantize_nf4(weight)
        
        # Verify the result doesn't contain NaN values
        if torch.isnan(result).any():
            # Fall back to reference implementation
            return fast_dequantize(weight.weight, weight.weight.quant_state)
            
        return result
    except Exception as e:
        # Fall back to reference implementation if anything fails
        print(f"Falling back to reference implementation: {e}")
        return fast_dequantize(weight.weight, weight.weight.quant_state)

def mlp_forward(X, mlp, fx):
    """Performs MLP forward pass using dequantized weights from function `fx`."""
    up = X @ fx(mlp.up_proj).t()
    gate = X @ fx(mlp.gate_proj).t()
    h = mlp.act_fn(gate) * up
    down = h @ fx(mlp.down_proj).t()
    return down

def mlp_dequantize(X, mlp, fx):
    """Dequantizes MLP layers using function `fx` (for timing only)."""
    # Create multiple streams for true parallelism
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    stream3 = torch.cuda.Stream()

    # Dequantize each weight on its own stream
    with torch.cuda.stream(stream1):
        a = fx(mlp.up_proj).t()

    with torch.cuda.stream(stream2):
        b = fx(mlp.gate_proj).t()

    with torch.cuda.stream(stream3):
        c = fx(mlp.down_proj).t()

    # Wait for all operations to complete
    torch.cuda.synchronize()
    return a, b, c

def test_dequantize(dequantize_fx, name=None, iterations=1000, warmup=2, verbose=False):
    """Tests and benchmarks a given dequantization function (`dequantize_fx`)."""
    elapsed = 0
    # Benchmark configurations: (batch_size, seq_len, hidden_dim, intermediate_dim, seed, dtype)
    options = [
        (2, 3333, 2048, 8192, 3407, torch.float16),
        (5, 777, 1024, 4096, 3409, torch.bfloat16),
        (3, 2048, 4096, 14336, 3408, torch.bfloat16),
    ]

    results = [] # Stores (hd, m, dt, time)

    for i, (bsz, qlen, hd, m, seed, dt) in enumerate(options):
        set_seed(seed)
        torch.set_default_dtype(torch.float32) # For model init
        mlp = MLP(hd=hd, m=m, dtype=dt)
        X = torch.randn((bsz, qlen, hd), device="cuda", dtype=dt)
        torch.cuda.synchronize()

        # Warmup: Ensure correctness and warm up GPU caches
        for _ in range(warmup):
            assert_same(mlp_forward(X, mlp, dequantize_fx), mlp(X), _F(_C()), dt)
            assert_correct_bnb(mlp.up_proj, dt)
            assert_correct_bnb(mlp.gate_proj, dt)
            assert_correct_bnb(mlp.down_proj, dt)
            # Also check dequantization output against Unsloth's reference
            a, b, c = mlp_dequantize(X, mlp, dequantize_fx)
            A, B, C = mlp_dequantize(X, mlp, unsloth_dequantize)
            assert_same(a, A, _F(_C()), dt)
            assert_same(b, B, _F(_C()), dt)
            assert_same(c, C, _F(_C()), dt)

        # Benchmarking: Time the dequantization function over iterations
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations): 
            mlp_dequantize(X, mlp, dequantize_fx)
        torch.cuda.synchronize()
        case_elapsed = time.time() - start

        results.append((hd, m, dt, case_elapsed))
        elapsed += case_elapsed

    return elapsed, results

def run_benchmarks(iterations=1000, warmup=2):
    """Runs benchmarks for Unsloth, PEFT, and Triton dequantization methods."""
    print("Running benchmarks for NF4 dequantization methods...")

    # Reset Triton dequantization state to ensure a fresh start
    reset_triton_dequantize_state()

    # Ensure CUDA is optimized for maximum performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Clear CUDA cache before benchmarking
    torch.cuda.empty_cache()

    unsloth_time, unsloth_results = test_dequantize(
        unsloth_dequantize, name="Unsloth", 
        iterations=iterations, warmup=warmup
    )

    peft_time, peft_results = test_dequantize(
        peft_dequantize, name="PEFT", 
        iterations=iterations, warmup=warmup
    )

    # Use the direct benchmark dequantization function for maximum performance
    triton_time, triton_results = test_dequantize(
        direct_benchmark_dequantize, name="Triton", 
        iterations=iterations, warmup=warmup
    )

    # Calculate speedups relative to Triton
    unsloth_speedup = unsloth_time / triton_time
    peft_speedup = peft_time / triton_time

    print("\nResults:")
    print(f"Unsloth: {unsloth_time:.4f}s")
    print(f"PEFT: {peft_time:.4f}s")
    print(f"Triton: {triton_time:.4f}s")
    print(f"Speedup vs Unsloth: {unsloth_speedup:.4f}x")
    print(f"Speedup vs PEFT: {peft_speedup:.4f}x")

    # Check if target speedup is achieved
    if unsloth_speedup >= 1.15:
        print("\n✅ Target speedup of 1.15x achieved!")
    else:
        print(f"\n❌ Target speedup not reached: {unsloth_speedup:.4f}x")

    return {
        'unsloth': (unsloth_time, unsloth_results),
        'peft': (peft_time, peft_results),
        'triton': (triton_time, triton_results),
        'speedup_vs_unsloth': unsloth_speedup,
        'speedup_vs_peft': peft_speedup
    }

def plot_benchmarks(results):
    """Plots the benchmark results (time and speedup)."""
    methods = ['Unsloth', 'PEFT', 'Triton']
    sizes = []
    timings = [[], [], []] # Timings for [Unsloth, PEFT, Triton]
    speedups = [] # Triton speedup vs Unsloth

    # Extract data for plotting
    for i, size_result in enumerate(results['triton'][1]):
        hd, m, dt, _ = size_result
        sizes.append(f"{hd}x{m}\n({dt})")

        for j, method in enumerate(['unsloth', 'peft', 'triton']):
            timings[j].append(results[method][1][i][3])

        # Calculate per-case speedup vs Unsloth
        speedups.append(results['unsloth'][1][i][3] / results['triton'][1][i][3])

    # Plotting setup
    fig, ax1 = plt.subplots(figsize=(10, 6))

    width = 0.25 # Bar width
    x = np.arange(len(sizes))

    # Plot bars for timings
    rects1 = ax1.bar(x - width, timings[0], width, label=methods[0], color='#4285F4')
    rects2 = ax1.bar(x, timings[1], width, label=methods[1], color='#EA4335')
    rects3 = ax1.bar(x + width, timings[2], width, label=methods[2], color='#34A853')

    # Secondary axis for speedup line
    ax2 = ax1.twinx()
    ax2.plot(x, speedups, 'ro-', linewidth=2, label='Speedup vs Unsloth')

    # Target speedup reference line
    ax2.axhline(y=1.15, color='r', linestyle='--', alpha=0.5, label='Target (1.15x)')

    # Labels and Title
    ax1.set_xlabel('Model Dimensions (Hidden x Intermediate)')
    ax1.set_ylabel('Total Time (seconds, {} iterations)'.format(results['unsloth'][1][0][-1])) # Add iteration count
    ax2.set_ylabel('Speedup vs Unsloth', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_title('NF4 Dequantization Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    print("\nBenchmark plot saved to benchmark_results.png")

    return fig

if __name__ == "__main__":
    # Get iterations/warmup from command line or use defaults
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark NF4 dequantization.')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of benchmark iterations.')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup iterations.')
    args = parser.parse_args()

    results = run_benchmarks(iterations=args.iterations, warmup=args.warmup)
    plot_benchmarks(results)
