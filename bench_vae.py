"""VAE decoding microbenchmark with fixed SD1.5 latent shape; optional saved real latent."""
from bench_perf_common import memory_guard, run_guarded

from bench_perf_common import parser,validate,measure,report

def main():
    p=parser(__doc__);p.add_argument('--latent');p.add_argument('--scope',choices=['eager','compiled'],default='eager');args=p.parse_args();validate(args)
    import mlx.core as mx
    memory_guard(mx)
    import numpy as np
    from m5diffusion.engine.common import ROOT
    import sys
    sys.path.insert(0,str(ROOT/'vendor'))
    from stable_diffusion.model_io import load_autoencoder
    vae=load_autoencoder(str(ROOT/'models/sd15'),False);mx.eval(vae.parameters())
    if args.latent:
        array=np.load(args.latent)
        if array.shape==(1,4,64,64):array=array.transpose(0,2,3,1)
        if array.shape!=(1,64,64,4):raise ValueError('latent must be 1x64x64x4 NHWC or 1x4x64x64 NCHW')
        x=mx.array(array,mx.float32)
    else:
        mx.random.seed(12345);x=mx.random.normal((1,64,64,4))
    mx.eval(x);fn=mx.compile(vae.decode) if args.scope=='compiled' else vae.decode
    report('vae',{args.scope:measure(mx,lambda:fn(x),args.repeat,args.warmup)},args.output,latent_source=args.latent or 'synthetic seed12345',dtype='float32')
if __name__=='__main__':run_guarded(main,'vae')
