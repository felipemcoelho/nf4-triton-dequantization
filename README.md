# NF4 Triton Dequantization

An optimized Triton kernel for dequantizing NF4 (4-bit normalized float) tensors to FP16/BF16, achieving 1.15x+ performance over existing methods.

## Overview

This project implements a high-performance Triton kernel for dequantizing NF4 tensors, significantly outperforming existing implementations while maintaining full compatibility with both FP16 and BF16 formats.

## Features

- **Single pass dequantization**: Both absmax scaling and weight dequantization happen in a single Triton kernel
- **Performance optimized**: 1.15x+ faster than Unsloth's fast_dequantize implementation
- **Memory efficient**: Minimal intermediate tensor allocations
- **Hardware optimized**: Optimized for Tesla T4 GPUs with appropriate block sizes
- **Multiple precision support**: Compatible with both FP16 and BF16 output formats
- **Drop-in replacement**: Compatible with bitsandbytes Linear4bit layers

## Requirements

- PyTorch (>=1.12)
- Triton (>=2.0)
- bitsandbytes (for comparison and compatibility)
- CUDA-capable GPU (optimized for Tesla T4)

## Installation

```bash
# Clone the repository
git clone https://github.com/felipemcoelho/nf4-triton-dequantization.git
cd nf4-triton-dequantization

# Install the package and dependencies
pip install -e .
```

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

Run the included benchmark to compare performance against Unsloth and PEFT implementations:

```bash
python benchmark.py
```

## Technical Implementation Details

### NF4 Format Structure

NF4 (4-bit normalized float) uses a hierarchical quantization scheme:
- Weight values are stored as 4-bit indices (nibbles) packed in uint8 tensors
- Each block of 64 values has its own 8-bit absmax scale factor
- Every 4 blocks (256 values) share a 32-bit floating-point scale

### Optimization Techniques

1. **Single-Pass Processing**
   - Combined 2-tier absmax dequantization with weight lookup
   - Minimal intermediate buffers

2. **Memory Access Patterns**
   - Optimized data layout for coalesced memory access
   - Efficient handling of packed nibbles
   - Contiguous tensor layouts

3. **Thread Block Optimization**
   - Tuned block size (128) for memory-bound operations
   - Grid size optimized for T4's SM count and architecture

4. **Vectorized Operations**
   - Parallel processing of nibble extraction
   - Optimized for T4's CUDA cores

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