import argparse,json,time,collections
import mlx.core as mx
import mlx.nn as nn
from m5diffusion.engine.mlx_backend import MLXEngine
from m5diffusion.engine.common import ROOT,Request,inputs
p=argparse.ArgumentParser();p.add_argument('--attention',choices=['standard','padded'],default='standard');a=p.parse_args()
engine=MLXEngine(ROOT/'models/sd15',attention=a.attention)
from m5diffusion.mlx_backend.attention import PaddedAttention
r=Request();ids,noise,sigmas,ts,coeff=inputs(engine.model,r)
c=engine.text(mx.array(ids)).last_hidden_state
x=mx.array(noise.transpose(0,2,3,1));u=mx.concatenate([x,x]).astype(mx.float16);t=mx.array([float(ts[0]),float(ts[0])])
mx.eval(engine.unet(u,t,encoder_x=c))
names={id(m):name for name,m in engine.unet.named_modules()}
records=[];original={}
for cls in [nn.Conv2d,nn.Linear,nn.GroupNorm,nn.LayerNorm,nn.MultiHeadAttention,PaddedAttention]:
 orig=cls.__call__;original[cls]=orig
 def wrapper(self,*a,_orig=orig,_cls=cls,**kw):
  mx.synchronize();start=time.perf_counter();out=_orig(self,*a,**kw);mx.eval(out);mx.synchronize()
  records.append({'name':names.get(id(self),'?'),'type':_cls.__name__,'ms':(time.perf_counter()-start)*1000,'shape':list(a[0].shape)})
  return out
 cls.__call__=wrapper
mx.eval(engine.unet(u,t,encoder_x=c))
for cls,orig in original.items():cls.__call__=orig
records.sort(key=lambda r:r['ms'],reverse=True)
summary={'note':'Instrumented single UNet step; per-op synchronization adds overhead. MHA includes nested Linear time. Diagnostic only, not generation benchmark.','top10':records[:10],'all':records}
(ROOT/f'benchmark/results/mlx-{a.attention}-op-profile.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary['top10'],indent=2))
