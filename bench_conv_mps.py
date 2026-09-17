"""Same-value FP16 convolution comparisons across MLX and PyTorch MPS.

Setup/readback are outside operation timing; this is not a zero-copy MLX/MPS
interop implementation. MPS measurements describe the runtime, not a verified
individual Metal Performance Shaders hardware kernel.
"""
import argparse,json,time,gc
from pathlib import Path
import numpy as np
import mlx.core as mx
import torch
from torch.nn import functional as F
from m5diffusion.engine.common import ROOT
from m5diffusion.profiling.microbench import measure,error


def measure_mps(fn,args,repeats):
    if repeats<10:raise ValueError('at least ten measured repeats required')
    with torch.inference_mode():
        for _ in range(3):fn(*args);torch.mps.synchronize()
        rows=[]
        for _ in range(repeats):
            start=time.perf_counter();out=fn(*args);submit=time.perf_counter()-start;torch.mps.synchronize()
            rows.append({'synchronized_wall_ms':(time.perf_counter()-start)*1000,'host_submission_ms':submit*1000})
    times=np.array([r['synchronized_wall_ms'] for r in rows])
    return {'median_ms':float(np.median(times)),'mean_ms':float(times.mean()),'min_ms':float(times.min()),'max_ms':float(times.max()),'std_ms':float(times.std()),'p25_ms':float(np.percentile(times,25)),'p75_ms':float(np.percentile(times,75)),'gpu_only_elapsed_ms':None,'warmup':3,'repeat':repeats,'runs':rows}


def main():
 p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,default=10);p.add_argument('--shapes',default='32:1280:1280,32:1920:640,16:2560:1280');p.add_argument('--output',type=Path,default=ROOT/'benchmark/optimization/conv-mps.json');a=p.parse_args()
 mx.set_cache_limit(2*2**30);mx.set_memory_limit(10*2**30);torch.mps.set_per_process_memory_fraction(.42)
 report={'cases':{},'notes':['Same deterministic FP16 inputs/weights in both runtimes. Input creation, cross-runtime CPU setup transfer, and result validation/readback excluded.','NHWC view case includes NHWC→NCHW permute and output reverse permute; those can be views. Contiguous case explicitly materializes both layout conversions. Weights are preconverted once as they would be for a resident model.','This does not demonstrate MLX/MPS interop performance; crossing runtimes during generation requires additional integration/copy work.','No torch.compile claim on MPS. Only MLX conv+residual compilation compared. Hardware GPU-only timestamps unavailable.']}
 for shape in a.shapes.split(','):
  h,ci,co=map(int,shape.split(':'));rng=np.random.default_rng(12345);xn=rng.standard_normal((2,h,h,ci)).astype(np.float16);wn=(rng.standard_normal((co,3,3,ci))*.01).astype(np.float16);rn=rng.standard_normal((2,h,h,co)).astype(np.float16)
  x,w,r=map(mx.array,(xn,wn,rn));mx.eval(x,w,r);native=lambda x,w:mx.conv2d(x,w,padding=1);add=lambda x,w,r:mx.conv2d(x,w,padding=1)+r
  ref=native(x,w);mx.eval(ref);ref_cpu=np.array(ref);refadd=add(x,w,r);mx.eval(refadd);refadd_cpu=np.array(refadd)
  rows={}
  for name,fn,args,reference in [('mlx_native',native,(x,w),ref),('mlx_conv_residual',add,(x,w,r),refadd),('mlx_compiled_conv_residual',mx.compile(add),(x,w,r),refadd)]:
   out=fn(*args);mx.eval(out);rows[name]={**error(out,reference),**measure(fn,args,a.repeat)}
  mx.synchronize();mx.clear_cache()
  tx=torch.from_numpy(xn).to('mps');tw=torch.from_numpy(wn.transpose(0,3,1,2).copy()).to('mps');tr=torch.from_numpy(rn).to('mps');tn=tx.permute(0,3,1,2).contiguous();torch.mps.synchronize()
  mps_nchw=lambda x,w:F.conv2d(x,w,padding=1)
  mps_view=lambda x,w:F.conv2d(x.permute(0,3,1,2),w,padding=1).permute(0,2,3,1)
  mps_copy=lambda x,w:F.conv2d(x.permute(0,3,1,2).contiguous(),w,padding=1).permute(0,2,3,1).contiguous()
  mps_add=lambda x,w,r:mps_view(x,w)+r
  for name,fn,args,reference in [('mps_native_nchw',mps_nchw,(tn,tw),ref_cpu.transpose(0,3,1,2)),('mps_nhwc_view_boundary',mps_view,(tx,tw),ref_cpu),('mps_nhwc_contiguous_boundary',mps_copy,(tx,tw),ref_cpu),('mps_nhwc_conv_residual',mps_add,(tx,tw,tr),refadd_cpu)]:
   with torch.inference_mode():result=fn(*args).float().cpu().numpy()
   delta=result-reference.astype(np.float32);rows[name]={'max_abs_error':float(np.abs(delta).max()),'rmse':float(np.sqrt(np.mean(delta**2))),'finite':bool(np.isfinite(result).all()),**measure_mps(fn,args,a.repeat)}
  for name,row in rows.items():print(shape,name,row['median_ms'],row['max_abs_error'],flush=True)
  report['cases'][shape]=rows;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
  del tx,tw,tr,tn,args;gc.collect();torch.mps.empty_cache();mx.clear_cache()
if __name__=='__main__':main()
