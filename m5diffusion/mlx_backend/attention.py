"""Keep the exact SD1.5 attention math; pad only the feature dimension.

MLX 0.32.2 has no fused full-attention kernel for D=40. Zero padding to
D=64 preserves QK^T and the first 40 output channels. Crucially the scale
remains 1/sqrt(40), not 1/sqrt(64). No tokens or attention terms are removed.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

class PaddedAttention(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.num_heads=original.num_heads
        for key in ['query_proj','key_proj','value_proj','out_proj']:
            setattr(self,key,getattr(original,key))
    def __call__(self,queries,keys,values,mask=None):
        q=self.query_proj(queries);k=self.key_proj(keys);v=self.value_proj(values)
        q,k,v=[mx.unflatten(a,-1,(self.num_heads,-1)).transpose(0,2,1,3) for a in (q,k,v)]
        d=q.shape[-1];scale=math.sqrt(1/d)
        pad=(d==40 and q.shape[-2]>=1024 and k.shape[-2]>=1024)
        if pad:q,k,v=[mx.pad(a,((0,0),(0,0),(0,0),(0,64-d))) for a in (q,k,v)]
        out=mx.fast.scaled_dot_product_attention(q,k,v,scale=scale,mask=mask,force_fused=pad)
        if pad:out=out[...,:d]
        return self.out_proj(out.transpose(0,2,1,3).flatten(-2,-1))

def install_padded_attention(model):
    replacements=[(name,PaddedAttention(m)) for name,m in model.named_modules() if isinstance(m,nn.MultiHeadAttention)]
    model.update_modules(tree_unflatten(replacements))
    return len(replacements)
