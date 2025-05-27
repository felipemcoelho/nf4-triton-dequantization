import sys
import types
import importlib

# create stub modules for torch, triton, and unsloth.fast_dequantize

torch_stub = types.ModuleType('torch')

triton_stub = types.ModuleType('triton')
triton_language_stub = types.ModuleType('triton.language')
triton_language_stub.constexpr = lambda x=None: x

def autotune_stub(*args, **kwargs):
    def decorator(fn):
        return fn
    return decorator

triton_stub.autotune = autotune_stub
def jit_stub(fn=None):
    if fn is None:
        def decorator(f):
            return f
        return decorator
    return fn

triton_stub.jit = jit_stub
class Config:
    def __init__(self, *args, **kwargs):
        pass

triton_stub.Config = Config
triton_stub.language = triton_language_stub

unsloth_stub = types.ModuleType('unsloth')
unsloth_kernels_stub = types.ModuleType('unsloth.kernels')
unsloth_utils_stub = types.ModuleType('unsloth.kernels.utils')

def fast_dequantize_stub(weight, quant_state):
    return ('fast', weight, quant_state)

unsloth_utils_stub.fast_dequantize = fast_dequantize_stub
unsloth_kernels_stub.utils = unsloth_utils_stub
unsloth_stub.kernels = unsloth_kernels_stub

# insert stubs into sys.modules before importing module under test
sys.modules['torch'] = torch_stub
sys.modules['triton'] = triton_stub
sys.modules['triton.language'] = triton_language_stub
sys.modules['unsloth'] = unsloth_stub
sys.modules['unsloth.kernels'] = unsloth_kernels_stub
sys.modules['unsloth.kernels.utils'] = unsloth_utils_stub

# import module under test
module = importlib.import_module('nf4_triton_dequantization.dequantization')

class DummyWeight:
    pass

class DummyModule:
    def __init__(self):
        self.weight = DummyWeight()
        self.weight.quant_state = 'qs'


import unittest


class BenchmarkFastDequantizeTest(unittest.TestCase):
    def test_benchmark_fast_dequantize_uses_fast(self):
        dummy = DummyModule()
        result = module.benchmark_fast_dequantize(dummy)
        expected = fast_dequantize_stub(dummy.weight, dummy.weight.quant_state)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
