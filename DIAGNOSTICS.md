# NF4 Triton Dequantization - Diagnostics Guide

## Running Diagnostics

After cloning and installing the package, run diagnostics to check performance:

```bash
# Clone and install
git clone https://github.com/felipemcoelho/nf4-triton-dequantization.git
cd nf4-triton-dequantization
pip install -e .

# Run diagnostics
python run_diagnostics.py
```

This will:
1. Check your environment (PyTorch, Triton, CUDA versions)
2. Test basic Triton functionality
3. Compare NF4 dequantization performance (Triton vs Unsloth)
4. Test the PyTorch fallback if Triton is slow
5. Provide recommendations

## Using the Fallback

On Tesla T4 and older GPUs, the package automatically uses the optimized PyTorch fallback.
On newer GPUs, you can force Triton to compare:

```bash
export NF4_USE_TRITON=1
python benchmark.py
```

Or explicitly call the fallback in your code:
```python
# In your code, explicitly use the fallback
from nf4_triton_dequantization.kernel_optimized import fast_pytorch_dequantize as pure_torch_fallback
result = pure_torch_fallback(module)
```

## Expected Output

### Good Performance (Triton works well):
```
Speedup: 1.20x
🟢 SUCCESS: Triton performs well
```

### Poor Performance (Use fallback):
```
Speedup: 0.02x
🔴 CRITICAL: Triton has severe performance issues
Recommended: Use pure_torch_fallback function
```

## Troubleshooting

### Issue: Triton is 50x+ slower than expected
**Cause**: Triton compilation overhead or compatibility issues
**Solution**: Use the PyTorch fallback (default on T4). To ensure fallback:
```bash
export NF4_USE_TRITON=0
```

### Issue: NaN values in output
**Cause**: Numerical instability in kernel
**Solution**: The fallback implementation handles this correctly

### Issue: Import errors
**Cause**: Missing dependencies
**Solution**: 
```bash
pip install torch triton bitsandbytes unsloth
```

## Performance Targets

- **Goal**: 1.15x speedup over Unsloth's fast_dequantize
- **Triton**: May achieve this on newer GPUs (A100, H100)
- **PyTorch Fallback**: More consistent across different GPUs

## Technical Details

The implementation provides:
1. **Double dequantization**: Applies both absmax and absmax32 scales
2. **Single kernel pass**: All operations in one GPU kernel
3. **NF4 lookup table**: 16 special quantization values
4. **64-element blocks**: Processes data in NF4 block size

## Benchmarking

Run the full benchmark suite:
```bash
# Standard benchmark
python benchmark.py

# Force Triton (on newer GPUs)
NF4_USE_TRITON=1 python benchmark.py
```

## Environment Variables

- `NF4_USE_TRITON`: `1` to force Triton backend (Ampere+ recommended), `0` or unset to use PyTorch fallback.
- `CUDA_VISIBLE_DEVICES`: Select GPU (e.g., `CUDA_VISIBLE_DEVICES=0`).

## GPU Compatibility

Tested on:
- Tesla T4 (may need fallback)
- Tesla V100 (Triton should work)
- A100 (Triton recommended)
- RTX 3090/4090 (Triton recommended)

For Tesla T4 specifically, the PyTorch fallback often performs better due to Triton compilation overhead.
