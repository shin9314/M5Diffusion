"""Experimental compile-scope comparison; no production engine mutation.

Each sample executes the unchanged 512x512, 20-step diffusion request.
Run one scope per process to avoid retaining compiled graphs for all variants.
"""
from bench_perf_common import memory_guard, run_guarded

from bench_perf_common import parser, validate, measure, report


def prepare(scope='step', block=2, attention='padded', conv='native'):
    import numpy as np
    import mlx.core as mx
    memory_guard(mx)
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from m5diffusion.engine.common import ROOT, Request, inputs
    from m5diffusion.engine.mlx_backend import MLXEngine
    r=Request(seed=12345,width=512,height=512,steps=20,cfg=7.0)
    engine=MLXEngine(ROOT/'models/sd15',compiled=False,attention=attention,convolution=False)
    if conv=='im2col':
        from benchmark_conv_experiment import install
        install(engine.unet)
    elif conv!='native':
        raise ValueError('Unknown convolution mode')
    ids,noise,sigmas,ts,coeff=inputs(engine.model,r)
    c=engine.text(mx.array(ids)).last_hidden_state
    x0=mx.array(noise.transpose(0,2,3,1))*float(sigmas[0])
    times=mx.array(np.stack([ts,ts],axis=-1));ks=mx.array(coeff);cfg=mx.array(7.,mx.float32)
    old0=mx.zeros_like(x0)
    mx.eval(c,x0,old0,times,ks,cfg);mx.synchronize()
    if scope=='attention':
        from m5diffusion.mlx_backend.attention import PaddedAttention
        class CompiledAttention(nn.Module):
            def __init__(self,module):
                super().__init__();self.original=module
                self.call=mx.compile(lambda q,k,v,mask:self.original(q,k,v,mask))
            def __call__(self,q,k,v,mask=None):return self.call(q,k,v,mask)
        replacements=[(name,CompiledAttention(m)) for name,m in engine.unet.named_modules()
                      if isinstance(m,(nn.MultiHeadAttention,PaddedAttention))]
        engine.unet.update_modules(tree_unflatten(replacements))
    unet=lambda u,t,c:engine.unet(u,t,encoder_x=c)
    if scope=='unet':unet=mx.compile(unet)
    def prediction(u,t,c,cfg):
        pred=unet(u,t,c).astype(mx.float32)
        neg,pos=mx.split(pred,2,axis=0)
        return neg+cfg*(pos-neg)
    predict=mx.compile(prediction) if scope=='unet_cfg' else prediction
    def step(x,old,c,t,k,cfg):
        sigma,ratio,factor,ca,cb=[k[i] for i in range(5)]
        u=(mx.concatenate([x,x],axis=0)*mx.rsqrt(1+sigma*sigma)).astype(mx.float16)
        den=x-sigma*predict(u,t,c,cfg)
        return ratio*x+factor*(ca*den+cb*old),den
    step_fn=mx.compile(step) if scope=='step' else step
    def multi(x,old,c,t,k,cfg):
        for i in range(block):x,old=step(x,old,c,t[i],k[i],cfg)
        return x,old
    multi_fn=mx.compile(multi) if scope=='multi-step' else None
    def diffusion():
        x=x0;old=old0
        if multi_fn:
            for i in range(0,20,block):
                x,old=multi_fn(x,old,c,times[i:i+block],ks[i:i+block],cfg)
                mx.async_eval(x,old)
        else:
            for i in range(20):
                x,old=step_fn(x,old,c,times[i],ks[i],cfg)
                mx.async_eval(x,old)
        return x
    return mx,engine,r,diffusion


def main():
    p=parser(__doc__);p.add_argument('--scope',choices=['none','unet','attention','unet_cfg','step','multi-step'],default='step')
    p.add_argument('--block',type=int,choices=[2,4,5,10,20],default=2)
    p.add_argument('--attention',choices=['standard','padded'],default='padded')
    p.add_argument('--conv',choices=['native','im2col'],default='native')
    p.add_argument('--save-image',action='store_true');p.add_argument('--reference',type=__import__('pathlib').Path);args=p.parse_args();validate(args)
    if args.reference and (not args.save_image or not args.reference.is_file()):
        p.error('--reference requires --save-image and an existing reference PNG')
    mx,engine,r,fn=prepare(args.scope,args.block,args.attention,args.conv)
    import json
    from pathlib import Path
    sample_path=(args.output or Path(f'benchmark/results/unet-{args.scope}.json')).with_suffix('.samples.jsonl')
    sample_path.parent.mkdir(parents=True,exist_ok=True)
    with sample_path.open('w') as samples:
        def on_sample(row):
            samples.write(json.dumps(row)+'\n');samples.flush()
            print(f'{args.scope} sample {row["index"]+1}/{args.warmup+args.repeat}: {row["elapsed_s"]:.3f}s'+(' warmup' if row['warmup'] else ''),flush=True)
        result=measure(mx,fn,args.repeat,args.warmup,on_sample=on_sample)
    result['seconds_per_step_median']=result['summary']['median']/20
    if args.save_image:
        import numpy as np
        from PIL import Image
        from pathlib import Path
        latent=fn();mx.eval(latent);decoded=engine.vae.decode(latent);mx.eval(decoded)
        out=np.clip(np.array(decoded[0])/2+.5,0,1)
        image_path=(args.output or Path(f'benchmark/results/unet-{args.scope}.json')).with_suffix('.png')
        image_path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.rint(out*255).astype(np.uint8)).save(image_path)
        np.save(image_path.with_suffix('.npy'),np.array(latent))
        result['image']=str(image_path)
        if args.reference:
            from skimage.metrics import structural_similarity, peak_signal_noise_ratio
            reference=np.asarray(Image.open(args.reference).convert('RGB'),dtype=np.float32)/255
            candidate=np.asarray(Image.open(image_path).convert('RGB'),dtype=np.float32)/255
            if candidate.shape!=reference.shape:raise ValueError('Reference image shape mismatch')
            identical=bool(np.array_equal(reference,candidate))
            result['quality_vs_reference']={'reference':str(args.reference),'ssim':float(structural_similarity(reference,candidate,channel_axis=2,data_range=1)),
                'psnr_db':None if identical else float(peak_signal_noise_ratio(reference,candidate,data_range=1)),
                'identical':identical,'psnr_note':'infinite for identical images' if identical else None} 
    report('unet',{args.scope:result},args.output,scope=args.scope,attention=args.attention,conv=args.conv,multi_step_block=args.block,
           request=vars(r),model_load_s=engine.load_time,measurement='20 complete diffusion steps; excludes text encoding and VAE')

if __name__=='__main__':run_guarded(main,'unet')
