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

If diagnostics show that Triton is slow on your system, you have three options:

### Option 1: Manual Fallback (Environment Variable)
```bash
# Use PyTorch fallback instead of Triton
export NF4_USE_PYTORCH_FALLBACK=1
python benchmark.py
```

### Option 2: Auto-Selection
```bash
# Automatically select best implementation
export NF4_AUTO_SELECT=1
python benchmark.py
```

### Option 3: Modify Code
```python
# In your code, explicitly use the fallback
from nf4_triton_dequantization.kernel import pure_torch_fallback
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
**Solution**: Use `NF4_USE_PYTORCH_FALLBACK=1`

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

# With fallback
NF4_USE_PYTORCH_FALLBACK=1 python benchmark.py

# With auto-selection
NF4_AUTO_SELECT=1 python benchmark.py
```

## Environment Variables

- `NF4_USE_PYTORCH_FALLBACK`: Set to 1 to use PyTorch implementation
- `NF4_AUTO_SELECT`: Set to 1 to auto-select best implementation
- `CUDA_VISIBLE_DEVICES`: Select GPU (e.g., `CUDA_VISIBLE_DEVICES=0`)

## GPU Compatibility

Tested on:
- Tesla T4 (may need fallback)
- Tesla V100 (Triton should work)
- A100 (Triton recommended)
- RTX 3090/4090 (Triton recommended)

For Tesla T4 specifically, the PyTorch fallback often performs better due to Triton compilation overhead.