"""Exact-attention candidates; full padding/unpadding costs included in named cases."""
import argparse,json,math
from pathlib import Path
import mlx.core as mx
from m5diffusion.engine.common import ROOT
from m5diffusion.profiling.microbench import measure,error

def manual(q,k,v):
    scores=(q*math.sqrt(1/q.shape[-1]))@k.swapaxes(-1,-2)
    return mx.softmax(scores,axis=-1,precise=True)@v

def sdpa(q,k,v):return mx.fast.scaled_dot_product_attention(q,k,v,scale=math.sqrt(1/q.shape[-1]))

def padded(size,force=True):
    def run(q,k,v):
        d=q.shape[-1];q,k,v=[mx.pad(x,((0,0),(0,0),(0,0),(0,size-d))) for x in (q,k,v)]
        return mx.fast.scaled_dot_product_attention(q,k,v,scale=math.sqrt(1/d),force_fused=force)[...,:d]
    return run

def chunked(q,k,v):
    return mx.concatenate([manual(q[:,:,i:i+256],k,v) for i in range(0,q.shape[-2],256)],axis=-2)

def tiled(q,k,v):
    # Numerically stable online softmax over all keys; no attention terms removed.
    result=[];scale=math.sqrt(1/q.shape[-1])
    for i in range(0,q.shape[-2],256):
        qi=q[:,:,i:i+256].astype(mx.float32)*scale
        maximum=mx.full((*qi.shape[:-1],1),-float('inf'));denom=mx.zeros_like(maximum);acc=mx.zeros(qi.shape,mx.float32)
        for j in range(0,k.shape[-2],512):
            scores=qi@k[:,:,j:j+512].astype(mx.float32).swapaxes(-1,-2)
            newmax=mx.maximum(maximum,mx.max(scores,axis=-1,keepdims=True));rescale=mx.exp(maximum-newmax);p=mx.exp(scores-newmax)
            acc=acc*rescale+p@v[:,:,j:j+512].astype(mx.float32);denom=denom*rescale+mx.sum(p,axis=-1,keepdims=True);maximum=newmax
        result.append((acc/denom).astype(q.dtype))
    return mx.concatenate(result,axis=-2)

def main():
 p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,default=10);p.add_argument('--output',type=Path,default=ROOT/'benchmark/optimization/attention.json');p.add_argument('--shapes',default='4096:4096:40,1024:1024:80,4096:77:40');p.add_argument('--methods',default='original_sdpa,manual,compiled_manual,pad48_auto,pad48_fused,pad64_fused,chunked,tiled');a=p.parse_args();report={'cases':{},'notes':['All math computes complete attention. Tiled online softmax accumulates FP32, so rounding differs from FP16 manual reference.','Padded cases include pad allocation/copy, SDPA and slicing/unpadding; scale stays original 1/sqrt(D).','Direct40 uses native SDPA fallback; unsupported forced-fused shapes are reported as errors. No custom Metal kernel is proposed before profiling.','Synchronized host wall includes submission and waiting, not a hardware GPU timestamp. Physical copy/allocation/kernel counts are unavailable.']}
 methods={'original_sdpa':sdpa,'manual':manual,'compiled_manual':mx.compile(manual),'pad48_auto':padded(48,False),'pad48_fused':padded(48),'pad64_fused':padded(64),'chunked':chunked,'tiled':tiled}
 for shape in a.shapes.split(','):
  nq,nk,d=map(int,shape.split(':'));q=mx.random.normal((2,8,nq,d)).astype(mx.float16);k=mx.random.normal((2,8,nk,d)).astype(mx.float16);v=mx.random.normal((2,8,nk,d)).astype(mx.float16);args=(q,k,v);mx.eval(args);reference=sdpa(*args);mx.eval(reference);rows={}
  for name in a.methods.split(','):
   if (name.startswith('pad48') and d>48) or (name.startswith('pad64') and d>64):continue
   try:
    fn=methods[name];out=fn(*args);mx.eval(out);rows[name]={'status':'ok',**error(out,reference),**measure(fn,args,a.repeat)}
   except Exception as exc:rows[name]={'status':'unsupported_or_error','error':str(exc)}
   print(shape,name,json.dumps({k:v for k,v in rows[name].items() if k!='runs'}),flush=True)
  if d==40:
   rows['padding_copy_only']=measure(lambda q,k,v:tuple(mx.pad(x,((0,0),(0,0),(0,0),(0,24))) for x in (q,k,v)),args,a.repeat)
   padded_out=mx.zeros((2,8,nq,64),mx.float16);mx.eval(padded_out);rows['unpadding_view_only']=measure(lambda x:x[...,:40],(padded_out,),a.repeat)
  report['cases'][shape]=rows;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));mx.clear_cache()
if __name__=='__main__':main()
