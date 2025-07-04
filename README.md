# NF4 Triton Dequantization

Optimized Triton kernel for dequantizing NF4 (4-bit normalized float) tensors to FP16/BF16.

## Installation

```bash
pip install -e .
```

## Usage

```python
from nf4_triton_dequantization import triton_dequantize_nf4

# Works with any Linear4bit layer
dequantized_weights = triton_dequantize_nf4(linear_4bit_layer)
```

## Benchmarking

```bash
python benchmark.py
```

## Requirements

- PyTorch >= 1.12
- Triton >= 2.0
- CUDA-capable GPU
- bitsandbytes
- unsloth

## License

Apache 2.0