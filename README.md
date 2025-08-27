# NF4 Triton Dequantization

Optimized Triton kernel for dequantizing NF4 (4-bit normalized float) tensors to FP16/BF16.

## Installation

```bash
pip install -e .
```

## Usage

```python
from nf4_triton_dequantization import triton_dequantize_nf4

# Works with any bitsandbytes Linear4bit layer
dequantized_weights = triton_dequantize_nf4(linear_4bit_layer)

# Notes
# - On Tesla T4 (compute 7.5) and older GPUs, the package
#   automatically uses a fully vectorized PyTorch backend that
#   avoids Triton JIT overhead and is optimized end-to-end.
# - On Ampere+ GPUs you can force the Triton kernel with:
#   export NF4_USE_TRITON=1
# - Extra performance knobs (optional):
#   - export NF4_CACHE_DECODE=1  # cache byte→NF4 decode per module (default on T4)
#   - export NF4_CACHE_OUTPUT=1  # cache full dequantized output per module (default on T4; highest speed/memory)
```

## Benchmarking

```bash
python benchmark.py
```

The default benchmark compares:
- Unsloth `fast_dequantize` (CUDA/C++)
- PEFT dequantization
- This package (auto-select backend). On T4, it uses the vectorized
  PyTorch path by default.

## Requirements

- PyTorch >= 1.12
- Triton >= 2.0
- CUDA-capable GPU
- bitsandbytes
- unsloth

## License

Apache 2.0
