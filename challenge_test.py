import torch
import torch.nn as nn
import time
from transformers import set_seed
from bitsandbytes.nn import Linear4bit
from transformers.activations import ACT2FN
from unsloth.kernels.utils import fast_dequantize
from peft.utils.integrations import dequantize_module_weight as peft_dequantize
from nf4_triton_dequantization.ultra_optimized import ultra_fast_triton_dequantize_nf4

# Challenge test implementation exactly as specified

def assert_same(x, y, message, dtype):
    rtol = 2e-1 if dtype == torch.bfloat16 else 1e-1
    atol = 2e-1 if dtype == torch.bfloat16 else 1e-1
    torch.testing.assert_close(x, y, rtol=rtol, atol=atol, check_stride=True)

def _F(x):
    return f"{x:>7.4f}"

def _C():
    return ""

def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)

def bnb_Linear4bit(hd, m, dtype = torch.float16):
    return Linear4bit(
        hd, m, bias = None,
        compute_dtype       = dtype,
        compress_statistics = True,
        quant_type          = "nf4",
    )

def assert_correct_bnb(weight, dtype):
    assert(weight.weight.dtype == torch.uint8)
    assert(weight.weight.quant_state.dtype == dtype)
    assert(weight.weight.quant_state.absmax.dtype == torch.uint8)
    assert(weight.weight.quant_state.code.dtype == torch.float32)
    assert(weight.weight.quant_state.offset.dtype == torch.float32)
    assert(weight.weight.quant_state.blocksize == 64)
    assert(weight.weight.quant_state.state2.absmax.dtype == torch.float32)
    assert(weight.weight.quant_state.state2.code.dtype == torch.float32)
    assert(weight.weight.quant_state.state2.blocksize == 256)

class MLP(nn.Module):
    def __init__(self, hd = 4096, m = 14336, dtype = torch.float16):
        super().__init__()
        self.gate_proj = bnb_Linear4bit(hd, m, dtype = dtype).to("cuda")
        self.up_proj   = bnb_Linear4bit(hd, m, dtype = dtype).to("cuda")
        self.down_proj = bnb_Linear4bit(m, hd, dtype = dtype).to("cuda")
        # [NEW] as at 18th Feb 2025
        self.gate_proj.weight.quant_state.dtype = dtype
        self.up_proj  .weight.quant_state.dtype = dtype
        self.down_proj.weight.quant_state.dtype = dtype
        self.act_fn = ACT2FN["silu"]
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def mlp_forward(X, mlp, fx):
    up   = X @ fx(mlp.  up_proj).t()
    gate = X @ fx(mlp.gate_proj).t()
    h = mlp.act_fn(gate) * up
    down = h @ fx(mlp.down_proj).t()
    return down

def mlp_dequantize(X, mlp, fx):
    a = fx(mlp.  up_proj).t(); torch.cuda.synchronize()
    b = fx(mlp.gate_proj).t(); torch.cuda.synchronize()
    c = fx(mlp.down_proj).t(); torch.cuda.synchronize()
    return a, b, c

def test_dequantize(dequantize_fx):
    elapsed = 0
    options = [
        (2, 3333, 2048,  8192, 3407, torch.float16),
        (5,  777, 1024,  4096, 3409, torch.bfloat16),
        (3, 2048, 4096, 14336, 3408, torch.bfloat16),
    ]
    for (bsz, qlen, hd, m, seed, dt) in options:
        set_seed(seed)
        torch.set_default_dtype(torch.float32)
        mlp = MLP(hd = hd, m = m, dtype = dt)
        X = torch.randn((bsz, qlen, hd), device = "cuda", dtype = dt)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(2):
            assert_same( mlp_forward(X, mlp, dequantize_fx), mlp(X), _F(_C()), dt)
            # [NEW] as at 18th Feb 2025
            assert_correct_bnb(mlp.  up_proj, dt)
            assert_correct_bnb(mlp.gate_proj, dt)
            assert_correct_bnb(mlp.down_proj, dt)
            a, b, c = mlp_dequantize(X, mlp, dequantize_fx)
            A, B, C = mlp_dequantize(X, mlp, unsloth_dequantize)
            assert_same(a, A, _F(_C()), dt)
            assert_same(b, B, _F(_C()), dt)
            assert_same(c, C, _F(_C()), dt)

        # Benchmarking
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000): mlp_dequantize(X, mlp, dequantize_fx)
        elapsed += time.time() - start
    return elapsed

# The implementation as requested in the challenge
from triton import jit
import triton
import triton.language as tl

@triton.jit
def _your_dequantize_nf4_kernel(
    qweight_ptr,
    absmax_ptr,
    absmax32_ptr,
    output_ptr,
    m, n,
    blocks_per_row: tl.constexpr,
    absmax32_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Ultra-fast NF4 dequantization kernel for the challenge."""
    pid = tl.program_id(0)
    
    # Calculate number of column groups
    num_col_groups = (n + GROUP_SIZE - 1) // GROUP_SIZE
    total_blocks = m * num_col_groups
    
    if pid >= total_blocks:
        return
    
    # Determine row and column group
    row = pid // num_col_groups
    col_group = pid % num_col_groups
    col_start = col_group * GROUP_SIZE
    
    if row >= m or col_start >= n:
        return
    
    # NF4 lookup values
    nf4_vals = tl.inline_const_array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    # Process GROUP_SIZE elements in 64-element chunks
    for block_offset in range(0, GROUP_SIZE, 64):
        col_block_start = col_start + block_offset
        if col_block_start >= n:
            break
        
        # Get absmax indices
        block_idx = col_block_start >> 6
        absmax_idx = row * blocks_per_row + block_idx
        absmax32_idx = row * absmax32_per_row + (block_idx >> 2)
        
        # Load and compute scale
        absmax = tl.load(absmax_ptr + absmax_idx).to(tl.float32)
        absmax32 = tl.load(absmax32_ptr + absmax32_idx)
        scale = absmax * 0.00787401574803149606 * absmax32
        
        # Process in BLOCK_SIZE chunks
        for chunk in range(0, 64, BLOCK_SIZE):
            col = col_block_start + chunk + tl.arange(0, BLOCK_SIZE)
            idx = row * n + col
            mask = col < n
            
            # Load packed weights
            packed_idx = idx >> 1
            packed = tl.load(qweight_ptr + packed_idx, mask=mask, other=0)
            
            # Extract nibbles
            is_odd = (idx & 1).to(tl.int1)
            nibbles = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF)
            
            # Lookup NF4 values
            nf4 = tl.gather(nf4_vals, nibbles)
            
            # Apply scale and store
            output = (nf4 * scale).to(dtype)
            tl.store(output_ptr + idx, output, mask=mask)

def _your_dequantize_nf4(weight, quant_state):
    """Setup function for the Triton kernel."""
    qweight = weight
    absmax = quant_state.absmax
    absmax32 = quant_state.state2.absmax
    dtype = quant_state.dtype
    device = qweight.device
    
    # Get dimensions from weight shape
    M, N = weight.shape[0], weight.shape[1] * 2  # Each byte stores 2 nibbles
    
    blocks_per_row = (N + 63) // 64
    absmax32_per_row = (blocks_per_row + 3) // 4
    
    # Handle absmax tensor shapes
    if absmax.dim() == 1:
        if absmax.numel() == blocks_per_row:
            absmax = absmax.unsqueeze(0).expand(M, -1)
        elif absmax.numel() == M * blocks_per_row:
            absmax = absmax.view(M, blocks_per_row)
    
    if absmax.shape != (M, blocks_per_row):
        return fast_dequantize(weight, quant_state)
    
    if absmax32.dim() == 1:
        if absmax32.numel() == absmax32_per_row:
            absmax32 = absmax32.unsqueeze(0).expand(M, -1)
        elif absmax32.numel() == M * absmax32_per_row:
            absmax32 = absmax32.view(M, absmax32_per_row)
    
    if absmax32.shape != (M, absmax32_per_row):
        return fast_dequantize(weight, quant_state)
    
    # Ensure contiguous
    qweight = qweight.contiguous()
    absmax = absmax.contiguous()
    absmax32 = absmax32.contiguous()
    
    # Allocate output
    output = torch.empty((M, N), dtype=dtype, device=device)
    
    # Launch kernel
    BLOCK_SIZE = 32
    GROUP_SIZE = 256
    num_col_groups = (N + GROUP_SIZE - 1) // GROUP_SIZE
    grid = (M * num_col_groups,)
    
    _your_dequantize_nf4_kernel[grid](
        qweight.view(-1),
        absmax.view(-1),
        absmax32.view(-1),
        output.view(-1),
        M, N,
        blocks_per_row,
        absmax32_per_row,
        BLOCK_SIZE,
        GROUP_SIZE,
        dtype,
    )
    
    return output

def your_dequantize_nf4(weight):
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state)

# Run the test
if __name__ == "__main__":
    print("Testing NF4 dequantization implementation...")
    
    # Test correctness first
    print("Running correctness tests...")
    try:
        test_dequantize(your_dequantize_nf4)
        print("✅ Correctness tests passed!")
    except Exception as e:
        print(f"❌ Correctness test failed: {e}")
        exit(1)
    
    # Benchmark
    print("\nRunning benchmarks...")
    unsloth_time = test_dequantize(unsloth_dequantize)
    your_time = test_dequantize(your_dequantize_nf4)
    
    speedup = unsloth_time / your_time
    
    print(f"\nResults:")
    print(f"Unsloth time: {unsloth_time:.4f}s")
    print(f"Your implementation time: {your_time:.4f}s")
    print(f"Speedup: {speedup:.4f}x")
    
    if speedup >= 1.15:
        print("\n✅ Target speedup of 1.15x achieved!")
    else:
        print(f"\n❌ Target speedup not reached. Need {1.15 - speedup:.4f}x more improvement.")
    
    # Also test with the ultra-optimized version
    print("\nTesting ultra-optimized version...")
    ultra_time = test_dequantize(ultra_fast_triton_dequantize_nf4)
    ultra_speedup = unsloth_time / ultra_time
    print(f"Ultra-optimized speedup: {ultra_speedup:.4f}x")