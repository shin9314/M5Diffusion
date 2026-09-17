"""Shared synchronized microbenchmark output. Importing this file does not initialize MLX."""
import argparse
import csv
import json
import statistics
import time
from pathlib import Path


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--repeat', type=int, default=10)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--output', type=Path)
    return p


def validate(args):
    if args.repeat < 10 or args.warmup < 1:
        raise ValueError('At least 1 warmup and 10 measured repetitions are required')


def stats(values):
    import numpy as np
    return dict(median=statistics.median(values), mean=statistics.mean(values),
                min=min(values), max=max(values), standard_deviation=statistics.pstdev(values),
                p25=float(np.percentile(values, 25)), p75=float(np.percentile(values, 75)))


def memory_guard(mx, cache_bytes=2 * 2**30):
    mx.set_cache_limit(cache_bytes)
    mx.set_memory_limit(10 * 2**30)


def is_oom(exc):
    return isinstance(exc, MemoryError) or any(token in str(exc).lower() for token in
        ('out of memory', 'memory limit', 'insufficient memory', 'allocation failed'))


def run_guarded(entry, name):
    try:
        entry()
    except (RuntimeError, MemoryError) as exc:
        if not is_oom(exc):
            raise
        import sys
        output = None
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output') + 1])
        report(name, {'not_completed': {'status': 'not_completed', 'reason': str(exc),
            'allocation_count': None}}, output, memory_limit_bytes=10 * 2**30)
        raise SystemExit(2)


def measure(mx, fn, repeat=10, warmup=1, on_sample=None):
    from m5diffusion.profiling.telemetry import thermal_snapshot
    rows=[]
    for i in range(warmup + repeat):
        mx.synchronize();mx.reset_peak_memory()
        before=mx.get_active_memory();start=time.perf_counter();cpu_start=time.process_time();thread_start=time.thread_time()
        value=fn()
        dispatch=time.perf_counter()-start
        dispatch_cpu=time.process_time()-cpu_start;dispatch_thread=time.thread_time()-thread_start
        mx.eval(value);mx.synchronize()
        elapsed=time.perf_counter()-start
        rows.append(dict(index=i,warmup=i<warmup,elapsed_s=elapsed,
                         host_dispatch_elapsed_s=dispatch,dispatch_process_cpu_s=dispatch_cpu,
                         dispatch_thread_cpu_s=dispatch_thread,process_cpu_total_s=time.process_time()-cpu_start,
                         gpu_timestamp_duration_s=None,post_dispatch_wait_s=elapsed-dispatch,
                         active_before_bytes=before,active_after_bytes=mx.get_active_memory(),
                         peak_bytes=mx.get_peak_memory(),cache_bytes=mx.get_cache_memory(),
                         allocation_count=None,total_allocated_bytes=None))
        rows[-1]['thermal_after']=thermal_snapshot()
        if on_sample is not None:
            on_sample(dict(rows[-1]))
        del value
    warm=[r['elapsed_s'] for r in rows if not r['warmup']]
    return dict(rows=rows,summary=stats(warm),cold_first_call_s=rows[0]['elapsed_s'],
                compilation_time_s=None,notes=[
                    'First call includes tracing/compilation/execution; compiler time is not separately exposed.',
                    'Host dispatch wall time can include runtime backpressure; it is not CPU-exclusive time.',
                    'Process CPU includes all process threads; thread CPU is calling-thread CPU only. Neither is a GPU timestamp.',
                    'Post-dispatch wait is not total GPU kernel duration because submission and execution overlap.',
                    'Allocation count and cumulative allocated bytes are unavailable; live/peak/cache bytes are allocator gauges.'])


def report(name, results, output=None, **metadata):
    result={'benchmark':name, 'results':results, **metadata}
    output=output or Path('benchmark/results')/f'{time.strftime("%Y%m%d-%H%M%S")}-{name}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    rows=[]
    for case,data in results.items():
        if 'summary' in data:
            print(f'{case}: median {data["summary"]["median"]:.6f}s (first {data["cold_first_call_s"]:.6f}s)',flush=True)
            rows.extend(dict(case=case,**row) for row in data['rows'])
    if rows:
        with output.with_suffix('.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(output,flush=True)
    return result
