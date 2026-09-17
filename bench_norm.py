"""Representative SD1.5 UNet normalization shapes, eager versus compiled."""
from bench_perf_common import memory_guard, run_guarded

from bench_perf_common import parser,validate,measure,report

def main():
    p=parser(__doc__);args=p.parse_args();validate(args)
    import mlx.core as mx
    memory_guard(mx)
    import mlx.nn as nn
    mx.random.seed(12345);results={}
    for size,channels in [(64,320),(32,640),(16,1280)]:
        x=mx.random.normal((2,size,size,channels)).astype(mx.float16);mx.eval(x)
        for kind,layer in [('group',nn.GroupNorm(32,channels,pytorch_compatible=True)),('layer',nn.LayerNorm(channels))]:
            layer.set_dtype(mx.float16);mx.eval(layer.parameters())
            for scope,fn in [('eager',layer),('compiled',mx.compile(layer))]:
                results[f'{kind}-{size}x{size}x{channels}-{scope}']=measure(mx,lambda fn=fn:fn(x),args.repeat,args.warmup)
    report('norm',results,args.output,batch=2,dtype='float16',note='Synthetic representative tensors; synchronized microbenchmarks are not whole-UNet operator shares.')
if __name__=='__main__':run_guarded(main,'norm')
