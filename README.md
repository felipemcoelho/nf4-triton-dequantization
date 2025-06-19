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

### Performance Results

The optimized Triton kernel achieves significant performance improvements:

- **Target**: 1.15x+ speedup over Unsloth's fast_dequantize
- **Key optimizations**:
  - Inline NF4 lookup table (eliminates memory loads)
  - Efficient bit manipulation for nibble extraction
  - Optimal block sizes based on matrix dimensions
  - Single-pass dequantization in one Triton kernel
  - Cache-optimized memory access patterns

## Technical Implementation Details

### NF4 Format Structure

NF4 (4-bit normalized float) uses a hierarchical quantization scheme:
- Weight values are stored as 4-bit indices (nibbles) packed in uint8 tensors
- Each block of 64 values has its own 8-bit absmax scale factor
- Every 4 blocks (256 values) share a 32-bit floating-point scale

### Optimization Techniques

1. **Single-Pass Processing**
   - Combined 2-tier absmax dequantization with weight lookup in a single kernel
   - No intermediate memory allocations

2. **Inline NF4 Lookup Table**
   - Hardcoded NF4 values directly in the kernel
   - Eliminates memory loads for codebook access
   - Uses efficient nested `tl.where` operations

3. **Optimized Memory Access**
   - Coalesced memory access patterns
   - Efficient nibble extraction using bit shifts: `(packed >> (is_odd << 2)) & 0x0F`
   - Contiguous tensor layouts for maximum bandwidth

4. **Adaptive Block Sizing**
   - Dynamic block size selection based on matrix dimensions:
     - 256 for matrices < 500K elements
     - 1024 for matrices < 5M elements
     - 2048 for matrices < 50M elements
     - 4096 for larger matrices
   - 1D grid for minimal scheduling overhead

5. **Efficient Scale Computation**
   - Precomputed constant `1.0 / 127.0` to avoid divisions
   - Fused multiply-add operations for scale application
   - Direct computation without intermediate variables

6. **Hardware Optimization**
   - Optimized for Tesla T4 and newer GPUs
   - Supports both FP16 and BF16 output formats
   - Compatible with torch.compile for additional optimization

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
