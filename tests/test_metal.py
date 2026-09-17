"""Opt-in GPU tests: M5DIFFUSION_GPU_TESTS=1 pytest tests/test_metal.py."""
import os
import numpy as np
import pytest
pytestmark=pytest.mark.skipif(os.environ.get('M5DIFFUSION_GPU_TESTS')!='1',reason='GPU tests require explicit opt-in and exclusive GPU')

@pytest.mark.parametrize('size',[1,255,256,257,16384])
@pytest.mark.parametrize('coeff',[[3,.7,.3,1,0],[3,.7,.3,1.2,-.2],[.03,0,1,1,0]])
def test_fused(size,coeff):
    import mlx.core as mx
    from m5diffusion.kernels.fused_scheduler import fused_scheduler,native_scheduler
    rng=np.random.default_rng(12)
    args=[mx.array(rng.normal(size=size),dtype=mx.float32) for _ in range(4)]+[mx.array(coeff,dtype=mx.float32),mx.array(7.)]
    actual=fused_scheduler(*args);expected=native_scheduler(*args);mx.eval(actual,expected)
    for a,b in zip(actual,expected):np.testing.assert_allclose(np.array(a),np.array(b),atol=2e-5,rtol=2e-5)

def test_tensor_ops():
    from m5diffusion.metal.tensor_backend import probe
    result=probe()
    for dtype in ('fp16','bf16'):
        assert result[dtype]['passed'],result[dtype]
