"""Experimental MPP tensor matmul. Explicit opt-in probe, never a silent claim of acceleration."""
from functools import lru_cache

@lru_cache(None)
def _matmul_kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='mpp_tensor_matmul', input_names=['a','b'], output_names=['out'], header='#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n',source='''
        using namespace metal;
        using namespace mpp::tensor_ops;
        constexpr auto desc = matmul2d_descriptor(32, 32);
        matmul2d<desc, execution_simdgroup> op;
        int row = int(threadgroup_position_in_grid.y) * 32;
        int col = int(threadgroup_position_in_grid.x) * 32;
        // MPP element_type matching rejects const; these views are read-only operands.
        auto ma = tensor(const_cast<device T*>(a) + row*K, dextents<int, 2>{K,32}, array<int,2>{1,K});
        auto mb = tensor(const_cast<device T*>(b) + col, dextents<int, 2>{32,K}, array<int,2>{1,N});
        auto md = tensor(out + row*N + col, dextents<int,2>{32,32}, array<int,2>{1,N});
        op.run(ma,mb,md);
    ''')

def tensor_matmul(a,b):
    import mlx.core as mx
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:
        raise ValueError('expected compatible rank-2 matrices')
    m,k=a.shape;n=b.shape[1]
    if m%32 or n%32 or k%32 or a.dtype!=b.dtype or a.dtype not in (mx.float16,mx.bfloat16):
        raise ValueError('MPP probe requires equal FP16/BF16 types and dimensions divisible by 32')
    return _matmul_kernel()(inputs=[a,b],template=[('K',k),('N',n),('T',a.dtype)],grid=(n,m//32,1),threadgroup=(32,1,1),output_shapes=[(m,n)],output_dtypes=[mx.float32])[0]

def probe():
    import mlx.core as mx
    import numpy as np
    results={}
    rng=np.random.default_rng(6)
    for name,dtype in [('fp16',mx.float16),('bf16',mx.bfloat16)]:
        try:
            a=mx.array(rng.normal(size=(64,64)),dtype=dtype);b=mx.array(rng.normal(size=(64,64)),dtype=dtype)
            out=tensor_matmul(a,b);mx.eval(out);mx.synchronize()
            ref=np.einsum('ik,kj->ij',np.array(a.astype(mx.float32)),np.array(b.astype(mx.float32)),optimize=False)
            err=float(np.max(np.abs(np.array(out.astype(mx.float32))-ref)))
            results[name]={'compiled_and_executed':True,'max_abs_error':err,'passed':bool(np.allclose(np.array(out.astype(mx.float32)),ref,atol=0.02,rtol=0.01))}
        except Exception as exc:
            results[name]={'compiled_and_executed':False,'passed':False,'error':str(exc)}
    results['int4']={'compiled_and_executed':False,'passed':False,'reason':'Native packed INT4 MPP arithmetic not implemented; allocation alone does not establish arithmetic support.'}
    return results
