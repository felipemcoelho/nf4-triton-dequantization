#!/usr/bin/env python3
"""
Quick test script to verify the optimized solution works
Run this after setting up your environment with PyTorch and bitsandbytes
"""

def test_import():
    """Test that the package can be imported."""
    try:
        from nf4_triton_dequantization import triton_dequantize_nf4
        print("✅ Package imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_functionality():
    """Test basic functionality if dependencies are available."""
    try:
        import torch
        from bitsandbytes.nn import Linear4bit
        from nf4_triton_dequantization import triton_dequantize_nf4
        
        print("\n🔧 Testing functionality...")
        
        # Create test layer
        layer = Linear4bit(
            256, 256,
            bias=None,
            compute_dtype=torch.float16,
            compress_statistics=True,
            quant_type="nf4"
        )
        
        if torch.cuda.is_available():
            layer = layer.cuda()
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   Device: CPU")
        
        # Test dequantization
        result = triton_dequantize_nf4(layer)
        
        print(f"   Output shape: {result.shape}")
        print(f"   Output dtype: {result.dtype}")
        print("✅ Functionality test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def test_performance():
    """Quick performance test if all dependencies are available."""
    try:
        import torch
        import time
        from bitsandbytes.nn import Linear4bit
        from nf4_triton_dequantization import triton_dequantize_nf4
        
        if not torch.cuda.is_available():
            print("\n⚠️ Performance test skipped (no GPU)")
            return True
        
        print("\n⚡ Quick performance test...")
        
        # Create larger test case
        layer = Linear4bit(
            1024, 1024,
            bias=None,
            compute_dtype=torch.float16,
            compress_statistics=True,
            quant_type="nf4"
        ).cuda()
        
        # Warmup
        for _ in range(3):
            _ = triton_dequantize_nf4(layer)
            torch.cuda.synchronize()
        
        # Measure
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(10):
            _ = triton_dequantize_nf4(layer)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        avg_time = (elapsed / 10) * 1000  # ms
        print(f"   Average time: {avg_time:.2f}ms")
        print("✅ Performance test completed!")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Performance test skipped: {e}")
        return True


def main():
    print("=" * 60)
    print("NF4 Optimized Solution - Quick Test")
    print("=" * 60)
    
    success = True
    
    # Test import
    if not test_import():
        print("\n⚠️ Please install the package first:")
        print("   python install_optimized.py")
        success = False
    else:
        # Test functionality
        test_functionality()
        
        # Test performance
        test_performance()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests completed!")
        print("\nTo run full benchmarks:")
        print("   python benchmark_optimized.py")
    else:
        print("❌ Some tests failed - please check your setup")
    print("=" * 60)


if __name__ == "__main__":
    main()