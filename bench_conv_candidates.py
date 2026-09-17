"""Profile-directed conv candidates using existing MPP matmul, no new kernels."""
import argparse,json
from pathlib import Path
import mlx.core as mx
from m5diffusion.engine.common import ROOT
from m5diffusion.profiling.microbench import measure,error
from m5diffusion.metal.tensor_backend import tensor_matmul


def mpp_im2col(x,w):
    b,h,width,ci=x.shape;co,kh,kw,_=w.shape
    padded=mx.contiguous(mx.pad(x,((0,0),(1,1),(1,1),(0,0))))
    hp,wp=padded.shape[1:3]
    patches=mx.as_strided(padded,shape=(b,h,width,kh,kw,ci),strides=(hp*wp*ci,wp*ci,ci,wp*ci,ci,1)).reshape(b*h*width,kh*kw*ci)
    # Existing MPP kernel takes row-contiguous A/B and accumulates FP32; this
    # includes patch formation, weight transpose/copy and output FP16 cast.
    matrix=mx.contiguous(w.reshape(co,-1).T)
    return tensor_matmul(patches,matrix).astype(x.dtype).reshape(b,h,width,co)


def batch_split(x,w):
    return mx.concatenate([mx.conv2d(x[i:i+1],w,padding=1) for i in range(x.shape[0])],axis=0)


def main():
 p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,default=10);p.add_argument('--shapes',default='32:1280:1280,32:1920:640,16:2560:1280');p.add_argument('--output',type=Path,default=ROOT/'benchmark/optimization/conv-candidates.json');a=p.parse_args();report={'cases':{},'notes':['Includes im2col formation, all contiguity/copies, FP32-to-FP16 output cast. MPP prototype has not been tuned for large convolution K.','Uses the existing verified MPP tensor_ops matmul source. No evidence of hardware neural accelerator utilization is claimed.','Batch split retains both CFG branches, all channels and all terms. Convolution kernel choice/rounding can differ.']}
 mx.set_cache_limit(2*2**30);mx.set_memory_limit(10*2**30)
 for shape in a.shapes.split(','):
  h,ci,co=map(int,shape.split(':'));mx.random.seed(12345);x=mx.random.normal((2,h,h,ci)).astype(mx.float16);w=(mx.random.normal((co,3,3,ci))*.01).astype(mx.float16);mx.eval(x,w)
  native=lambda x,w:mx.conv2d(x,w,padding=1)
  ref=native(x,w);mx.eval(ref);rows={}
  for name,fn in [('native',native),('batch_split',batch_split),('compiled_batch_split',mx.compile(batch_split)),('mpp_im2col',mpp_im2col)]:
   try:
    out=fn(x,w);mx.eval(out);rows[name]={'status':'ok',**error(out,ref),**measure(fn,(x,w),a.repeat)}
   except Exception as exc:rows[name]={'status':'error','error':str(exc)}
   print(shape,name,json.dumps({k:v for k,v in rows[name].items() if k!='runs'}),flush=True)
  report['cases'][shape]=rows;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));mx.clear_cache()
if __name__=='__main__':main()
