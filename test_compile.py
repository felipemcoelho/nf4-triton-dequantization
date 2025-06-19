import torch
import torch.nn as nn
from bitsandbytes.nn import Linear4bit
from nf4_triton_dequantization import triton_dequantize_nf4

def test_torch_compile_compatibility():
    """Test if the Triton kernel works with torch.compile."""
    
    # Create a simple 4-bit quantized layer
    layer = Linear4bit(
        256, 512, 
        bias=None,
        compute_dtype=torch.float16,
        compress_statistics=True,
        quant_type="nf4",
    ).to("cuda")
    
    # Set dtype for quant_state
    layer.weight.quant_state.dtype = torch.float16
    
    try:
        # Try to compile the dequantization function
        compiled_dequant = torch.compile(triton_dequantize_nf4, mode="reduce-overhead")
        
        # Test the compiled function
        result1 = triton_dequantize_nf4(layer)
        result2 = compiled_dequant(layer)
        
        # Check if results match
        if torch.allclose(result1, result2, rtol=1e-3, atol=1e-3):
            print("✅ Torch compile compatibility: PASSED")
            print("   - Compiled function produces correct results")
            return True
        else:
            print("❌ Torch compile compatibility: FAILED")
            print("   - Results don't match between compiled and non-compiled")
            return False
            
    except Exception as e:
        print("❌ Torch compile compatibility: FAILED")
        print(f"   - Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_torch_compile_compatibility()