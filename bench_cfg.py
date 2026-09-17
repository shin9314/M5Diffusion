"""CFG pointwise fusion microbenchmark; UNet batching strategies are not implied."""
from bench_perf_common import memory_guard, run_guarded

from bench_perf_common import parser,validate,measure,report

def main():
    p=parser(__doc__);args=p.parse_args();validate(args)
    import mlx.core as mx
    memory_guard(mx)
    mx.random.seed(12345)
    pred=mx.random.normal((2,64,64,4));scale=mx.array(7.,mx.float32);mx.eval(pred,scale)
    def cfg(p,s):
        neg,pos=mx.split(p,2,axis=0);return neg+s*(pos-neg)
    compiled=mx.compile(cfg)
    results={name:measure(mx,lambda fn=fn:fn(pred,scale),args.repeat,args.warmup) for name,fn in [('eager',cfg),('compiled',compiled)]}
    report('cfg',results,args.output,shape=[2,64,64,4],cfg=7.0,note='CFG arithmetic only, batched UNet remains unchanged.')
if __name__=='__main__':run_guarded(main,'cfg')
