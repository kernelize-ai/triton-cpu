import functools
import pytest

import triton
import triton.language as tl

from triton._internal_testing import (numpy_random, to_triton)

import triton.backends.cpu.compiler as cpu_compiler


@functools.lru_cache()
def is_x86():
    import platform
    return platform.machine() in ("x86_64", "AMD64")


@triton.jit
def kernel(X, Z, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    z = tl.sum(x, axis=0)
    tl.store(Z, z)


@pytest.mark.skipif(not is_x86(), reason="Cpu feature string test only supported on x86")
def test_cpu_features(device):

    features = cpu_compiler.get_target_features()

    if len(features) == 0:
        pytest.skip("No cpu feature string on current platform")

    avx2_features = [f for f in features.split(',') if 'avx2' in f and f.startswith('+')]
    if len(avx2_features) == 0:
        pytest.skip("AVX2 not supported on current platform")

    features_no_avx2 = features + ",-avx2"

    shape = 64
    x = numpy_random(shape, dtype="int16")
    x_tri = to_triton(x, device=device)
    z_tri = to_triton(numpy_random((1, ), dtype_str="int16"), device=device, dst_type="int16")
    compiled = kernel[(1, )](x_tri, z_tri, BLOCK=shape)

    asm = compiled.asm
    assert "ymm" not in asm, "AVX2 instructions found despite -avx2"
