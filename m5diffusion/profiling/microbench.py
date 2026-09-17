"""Microbenchmarks report synchronized host elapsed and graph construction separately."""
import time
import numpy as np
import mlx.core as mx
mx.set_cache_limit(2 * 2**30)
mx.set_memory_limit(10 * 2**30)

def measure(fn,args,repeats=10,warmup=3):
    if repeats<10:raise ValueError('at least ten repetitions required')
    mx.eval(args);mx.synchronize()
    for _ in range(warmup):mx.eval(fn(*args));mx.synchronize()
    rows=[];mx.reset_peak_memory()
    for _ in range(repeats):
        start=time.perf_counter();out=fn(*args);dispatch=time.perf_counter()-start
        mx.eval(out);mx.synchronize();rows.append({'synchronized_wall_ms':(time.perf_counter()-start)*1000,'python_graph_construction_ms':dispatch*1000})
    times=np.array([x['synchronized_wall_ms'] for x in rows])
    return {'median_ms':float(np.median(times)),'mean_ms':float(times.mean()),'min_ms':float(times.min()),'max_ms':float(times.max()),'std_ms':float(times.std()),'p25_ms':float(np.percentile(times,25)),'p75_ms':float(np.percentile(times,75)),'graph_construction_median_ms':float(np.median([r['python_graph_construction_ms'] for r in rows])),'gpu_only_elapsed_ms':None,'peak_active_mb':mx.get_peak_memory()/2**20,'warmup':warmup,'repeat':repeats,'runs':rows}

def error(out,reference):
    a=np.array(out.astype(mx.float32));b=np.array(reference.astype(mx.float32));diff=a-b
    return {'max_abs_error':float(np.abs(diff).max()),'rmse':float(np.sqrt(np.mean(diff**2))),'finite':bool(np.isfinite(a).all())}
