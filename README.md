# NF4 Triton Dequantization

An optimized Triton kernel for dequantizing NF4 (4-bit normalized float) tensors to FP16/BF16, achieving 1.15x+ performance over existing methods.

## Overview

This project implements a high-performance Triton kernel for dequantizing NF4 tensors, significantly outperforming existing implementations while maintaining full compatibility.

## Features

- Single Triton kernel implementation for NF4 dequantization
- 1.15x+ faster than Unsloth's fast_dequantize implementation
- Supports both FP16 and BF16 output formats
- Memory efficient with minimal intermediate buffers
- Compatible with Tesla T4 GPUs
- Cache optimization for improved memory access patterns

## Requirements

- PyTorch
- Triton
- Bitsandbytes
- Unsloth (for comparison benchmarking)
- PEFT (for comparison benchmarking)
- Matplotlib (for visualization)

## Installation

```bash
# Clone the repository
git clone https://github.com/felipemcoelho/nf4-triton-dequantization.git
cd nf4-triton-dequantization

# Install the package and all dependencies (including benchmark requirements)
pip install -e .

# Or alternatively, install from PyPI
# pip install nf4-triton-dequantization
```

All necessary dependencies for both the core functionality and benchmarking (including unsloth, transformers, peft) will be installed automatically.

## Usage

```python
from nf4_triton_dequantization import triton_dequantize_nf4

# For a model with 4-bit quantized weights
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("your-model", load_in_4bit=True)

# Dequantize a Linear4bit layer
layer = model.model.layers[0].mlp.gate_proj
dequantized_weights = triton_dequantize_nf4(layer)
```

## Benchmarking

```bash
python benchmark.py
```

## Technical Details

### NF4 Format

NF4 is a 4-bit quantization format that uses a normalized distribution. Each tensor is divided into blocks (typically 64 values per block), and each block has its own scaling factor (absmax). The weights are stored as 4-bit indices into a lookup table of normalized values.

### Optimization Techniques

1. **Single Kernel Execution**: Both absmax scaling and dequantization in one kernel
2. **Cache Optimization**: Prefetches lookup tables with cache hints
3. **Efficient Memory Access**: Reduces memory operations
4. **Vectorized Processing**: Processes multiple values in parallel

## License

Apache License 2.0

## Citation

```
@misc{nf4_triton_dequantization,
  author = {Felipe Coelho},
  title = {NF4 Triton Dequantization},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/felipemcoelho/nf4-triton-dequantization}}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.