"""Synchronized warm microbenchmarks; GPU must be idle for meaningful numbers."""
import argparse,json,time,statistics
from pathlib import Path
import numpy as np
import mlx.core as mx
from m5diffusion.kernels.fused_scheduler import native_scheduler,fused_scheduler
from m5diffusion.metal.tensor_backend import probe,tensor_matmul

def measure(fn,args,repeats=20,batch=20):
    for _ in range(3):mx.eval(fn(*args))
    mx.synchronize();times=[]
    for _ in range(repeats):
        start=time.perf_counter()
        for _ in range(batch):mx.async_eval(fn(*args))
        mx.synchronize();times.append((time.perf_counter()-start)/batch*1000)
    return {'median_ms':statistics.median(times),'min_ms':min(times),'repeats':repeats,'batch':batch}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='benchmark/metal.json');a=p.parse_args()
    rng=np.random.default_rng(9);report={'scheduler':{},'tensor_ops':probe()}
    compiled=mx.compile(native_scheduler)
    for size in (64,96,128):
        args=[mx.array(rng.normal(size=(1,size,size,4)),dtype=mx.float32) for _ in range(4)]+[mx.array([3.,.7,.3,1.2,-.2]),mx.array(7.)]
        mx.eval(args);ref=native_scheduler(*args);mx.eval(ref)
        row={}
        for name,fn in [('eager',native_scheduler),('compiled',compiled),('custom_metal',fused_scheduler)]:
            out=fn(*args);mx.eval(out)
            row[name]={**measure(fn,args),'max_abs_error':max(float(mx.max(mx.abs(x-y)).item()) for x,y in zip(out,ref))}
        report['scheduler'][str(size*8)]=row
    if report['tensor_ops']['fp16']['passed']:
        report['matmul_fp16']={}
        for m,k,n in [(320,320,320),(1024,320,320),(1024,768,320)]:
            args=[mx.array(rng.normal(size=s),dtype=mx.float16) for s in [(m,k),(k,n)]];mx.eval(args)
            report['matmul_fp16'][f'{m}x{k}x{n}']={'native':measure(lambda a,b:a@b,args),'mpp':measure(tensor_matmul,args)}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
