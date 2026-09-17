"""UNet-like Conv2d microbenchmarks; layout conversion costs are explicit."""
import argparse,json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from m5diffusion.engine.common import ROOT
from m5diffusion.profiling.microbench import measure,error

def main():
 p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,default=10);p.add_argument('--shapes',default='64:320:320,32:640:640,16:1280:1280,64:960:320');p.add_argument('--output',type=Path,default=ROOT/'benchmark/optimization/conv.json');a=p.parse_args();report={'cases':{},'notes':['MLX conv input NHWC and kernel OHWI. NCHW case includes transpose to/from the native NHWC operator; it is not an independent native NCHW convolution.','Direct MPS convolution is not included in this script; no MPS kernel-dispatch claim is made.','Eager/compiled conv+SiLU and conv+residual preserve their respective reference expressions. No production change.','TensorOps direct convolution is not exposed by the tested MPP matmul prototype; not fabricated.','Synchronized wall includes host dispatch/await; GPU-only timing unavailable.']}
 for shape in a.shapes.split(','):
  spatial,ci,co=map(int,shape.split(':'));x=mx.random.normal((2,spatial,spatial,ci)).astype(mx.float16);w=(mx.random.normal((co,3,3,ci))*.01).astype(mx.float16);bias=mx.zeros((co,),mx.float16);residual=mx.zeros((2,spatial,spatial,co),mx.float16);mx.eval(x,w,bias,residual)
  native=lambda x,w,b:mx.conv2d(x,w,padding=1)+b
  act=lambda x,w,b:nn.silu(native(x,w,b))
  add=lambda x,w,b,r:native(x,w,b)+r
  layout=lambda x,w,b:(mx.conv2d(x.transpose(0,2,3,1),w,padding=1)+b).transpose(0,3,1,2)
  ref=native(x,w,bias);mx.eval(ref);rows={}
  for name,fn,args,reference in [('nhwc_native',native,(x,w,bias),ref),('nhwc_compiled',mx.compile(native),(x,w,bias),ref),('conv_silu_eager',act,(x,w,bias),act(x,w,bias)),('conv_silu_compiled',mx.compile(act),(x,w,bias),act(x,w,bias)),('conv_residual_eager',add,(x,w,bias,residual),add(x,w,bias,residual)),('conv_residual_compiled',mx.compile(add),(x,w,bias,residual),add(x,w,bias,residual)),('nchw_conversion_roundtrip',layout,(mx.contiguous(x.transpose(0,3,1,2)),w,bias),ref.transpose(0,3,1,2))]:
   mx.eval(reference);out=fn(*args);mx.eval(out);rows[name]={**error(out,reference),**measure(fn,args,a.repeat)}
   print(shape,name,rows[name]['median_ms'],flush=True)
  rows['layout_copy_only']=measure(lambda x:mx.contiguous(x.transpose(0,3,1,2)),(x,),a.repeat)
  report['cases'][shape]=rows;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));mx.clear_cache()
if __name__=='__main__':main()
