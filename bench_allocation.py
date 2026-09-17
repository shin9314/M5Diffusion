"""Diffusion allocator gauges; cache on/off is not explicit workspace reuse."""
from bench_perf_common import memory_guard, run_guarded

from bench_perf_common import parser,validate,measure,report
from bench_unet import prepare

def main():
    p=parser(__doc__);p.add_argument('--cache-mib',type=int,default=2048);p.add_argument('--scope',choices=['none','step'],default='step');args=p.parse_args();validate(args)
    if not 0<=args.cache_mib<=4096:raise ValueError('cache size must be 0..4096 MiB')
    mx,engine,r,fn=prepare(args.scope)
    mx.synchronize();mx.clear_cache();mx.set_cache_limit(args.cache_mib*2**20)
    result=measure(mx,fn,args.repeat,args.warmup)
    report('allocation',{f'cache-{args.cache_mib}MiB':result},args.output,request=vars(r),
           allocation_count=None,cumulative_allocated_bytes=None,explicit_workspace_pool=False,
           explanation='MLX allocator cache limit comparison. No public allocation-event counter, no explicit mutable activation-buffer reuse implemented.')
if __name__=='__main__':run_guarded(main,'allocation')
