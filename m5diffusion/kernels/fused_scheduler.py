"""Optional Metal CFG + DPM++2M fusion, FP32 inputs and outputs."""
from functools import lru_cache

def native_scheduler(x, old, neg, pos, coefficients, cfg):
    sigma, ratio, factor, ca, cb = [coefficients[i] for i in range(5)]
    den = x - sigma * (neg + cfg * (pos - neg))
    return ratio * x + factor * (ca * den + cb * old), den

@lru_cache(None)
def _kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='cfg_dpmpp2m', input_names=['x','old','neg','pos','k','cfg'], output_names=['next_x','den'], source='''
        uint i = thread_position_in_grid.x;
        float d = x[i] - k[0] * (neg[i] + cfg * (pos[i] - neg[i]));
        den[i] = d;
        next_x[i] = k[1] * x[i] + k[2] * (k[3] * d + k[4] * old[i]);
    ''')

def fused_scheduler(x, old, neg, pos, coefficients, cfg):
    import mlx.core as mx
    if any(a.shape != x.shape for a in (old,neg,pos)):
        raise ValueError('latent and prediction shapes must match')
    if any(a.dtype != mx.float32 for a in (x,old,neg,pos,coefficients,cfg)):
        raise ValueError('fused scheduler requires FP32 arrays')
    if coefficients.size != 5 or cfg.size != 1:
        raise ValueError('expected five coefficients and scalar CFG')
    return tuple(_kernel()(inputs=[x,old,neg,pos,coefficients,cfg], grid=(x.size,1,1), threadgroup=(256,1,1), output_shapes=[x.shape,x.shape], output_dtypes=[mx.float32,mx.float32]))
