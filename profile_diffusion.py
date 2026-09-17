"""Diagnostic per-operation profile, separate from production benchmark timings."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mlx.core as mx
from m5diffusion.engine.mlx_backend import MLXEngine
from m5diffusion.engine.common import ROOT,Request,inputs
from m5diffusion.profiling.operation_profiler import OperationProfiler

def main():
 p=argparse.ArgumentParser();p.add_argument('--attention',default='padded',choices=['standard','padded']);p.add_argument('--repeat',type=int,default=3);p.add_argument('--output',type=Path,default=ROOT/'benchmark/optimization/profile-diffusion.json');p.add_argument('--capture',type=Path);a=p.parse_args()
 e=MLXEngine(ROOT/'models/sd15',attention=a.attention,compiled=False);r=Request();ids,noise,sigmas,ts,coeff=inputs(e.model,r)
 c=e.text(mx.array(ids)).last_hidden_state;x=mx.array(noise.transpose(0,2,3,1))*float(sigmas[0]);old=mx.zeros_like(x);t=mx.array([float(ts[0]),float(ts[0])]);k=mx.array(coeff[0]);cfg=mx.array(r.cfg);mx.eval(c,x,old,t,k,cfg)
 args=(x,old,c,t,k,cfg);mx.eval(e._step(*args));mx.synchronize()
 baseline=[]
 for _ in range(max(3,a.repeat)):
  s=time.perf_counter();out=e._step(*args);dispatch=time.perf_counter()-s;mx.eval(out);mx.synchronize();baseline.append({'synchronized_step_ms':(time.perf_counter()-s)*1000,'python_graph_construction_ms':dispatch*1000})
 if a.capture:
  mx.metal.start_capture(str(a.capture))
  try:mx.eval(e._step(*args));mx.synchronize()
  finally:mx.metal.stop_capture()
 with OperationProfiler(e) as profiler:
  instrumented=[]
  for _ in range(a.repeat):
   start=time.perf_counter();sigma,ratio,factor,ca,cb=[k[i] for i in range(5)]
   u=profiler.call('CFG_prepare','duplicate_scale_cast',lambda x: (mx.concatenate([x,x],axis=0)*mx.rsqrt(1+sigma*sigma)).astype(mx.float16),x)
   pred=e.unet(u,t,encoder_x=c)
   pred=profiler.method('cast',pred,'astype',mx.float32);neg,pos=mx.split(pred,2,axis=0)
   den=profiler.call('CFG','guided_denoised',lambda x,n,p:x-sigma*(n+cfg*(p-n)),x,neg,pos)
   nxt=profiler.call('scheduler','DPM++2M',lambda x,d,o:ratio*x+factor*(ca*d+cb*o),x,den,old)
   mx.eval(nxt);mx.synchronize();instrumented.append((time.perf_counter()-start)*1000)
 report={**profiler.summarize(),'unmodified_step':baseline,'instrumented_step_ms':instrumented,'instrumentation_slowdown_vs_uncompiled':float(np.median(instrumented)/np.median([b['synchronized_step_ms'] for b in baseline])),'instrumented_output_max_abs':float(mx.max(mx.abs(nxt-out[0])).item()),'attention':a.attention,'sampled_timestep_index':0,'repeat':a.repeat,'gpu_only_elapsed_ms':None,'hardware_kernel_count':None,'physical_allocation_count':None,'notes':['Synchronized wall times include Python, dispatch, GPU execution and synchronization; they are not hardware GPU timestamps.','Input arrays are evaluated before each timed leaf to avoid attributing pending dependencies to that leaf.','Attention is atomic including projections/internal reshape/cast; nested records suppressed. Therefore categories do not overlap, but internal layout is attributed to attention.','Instrumented shares are diagnostic shares, not percentages of production compiled diffusion. Per-op barriers defeat fusion/overlap.','AST residual category means explicit vendor addition/subtraction. Copy means concatenate/pad; broadcast/reshape/transpose may be views, not physical data copies.','MLX Python provides Metal capture, not per-kernel timestamp/count API. Use --capture with MTL_CAPTURE_ENABLED=1 for Xcode inspection.','One initial diffusion timestep sampled repeatedly. No production engine or vendor sources changed.']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps({'categories':report['categories'],'top20':report['top20'],'overhead':report['instrumentation_slowdown_vs_uncompiled']},indent=2))
if __name__=='__main__':main()
