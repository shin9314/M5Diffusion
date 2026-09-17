from pathlib import Path
import sys,time
import numpy as np
from .common import ROOT,inputs,image_array
sys.path.insert(0,str(ROOT/'vendor'))

class MLXEngine:
    backend='mlx'
    def __init__(self,model,compiled=False,buffer_reuse=True,quantize=0,attention="standard",**kw):
        import mlx.core as mx
        from stable_diffusion.model_io import load_unet,load_text_encoder,load_autoencoder
        mx.set_cache_limit(2 * 2**30)
        mx.set_memory_limit(10 * 2**30)
        self.mx=mx;self.model=Path(model);self.compiled=compiled;self.buffer_reuse=buffer_reuse;self.cache={};self.quantize=quantize
        s=time.perf_counter()
        self.unet=load_unet(str(model),True);self.text=load_text_encoder(str(model),True);self.vae=load_autoencoder(str(model),False)
        if attention == "padded":
            from m5diffusion.mlx_backend.attention import install_padded_attention
            install_padded_attention(self.unet)
        from m5diffusion.mlx_backend.convolution import install
        if kw.get("convolution", True):
            install(self.unet)
        if quantize:
            import mlx.nn as nn
            nn.quantize(self.unet,group_size=32,bits=quantize,class_predicate=lambda _,m:isinstance(m,nn.Linear))
            nn.quantize(self.text,group_size=32,bits=quantize,class_predicate=lambda _,m:isinstance(m,nn.Linear))
        mx.eval(self.unet.parameters(),self.text.parameters(),self.vae.parameters());mx.synchronize()
        self.load_time=time.perf_counter()-s
        from .lora import LoRAManager
        self.lora_manager=LoRAManager(self)
        self._refresh_step()
    def _refresh_step(self):
        # A fresh callable prevents compiled closures reusing old adapter weights.
        fn=lambda x,old,c,t,k,cfg:self._step(x,old,c,t,k,cfg)
        self.step=self.mx.compile(fn) if self.compiled else self._step
    def set_loras(self,adapters):
        self.lora_manager.set(adapters)

    def _step(self,x,old,c,t,k,cfg):
        mx=self.mx;sigma,ratio,factor,ca,cb=[k[i] for i in range(5)]
        u=(mx.concatenate([x,x],axis=0)*mx.rsqrt(1+sigma*sigma)).astype(mx.float16)
        pred=self.unet(u,t,encoder_x=c).astype(mx.float32)
        neg,pos=mx.split(pred,2,axis=0);den=x-sigma*(neg+cfg*(pos-neg))
        return ratio*x+factor*(ca*den+cb*old),den
    def generate(self,r,save_intermediates=None):
        mx=self.mx;start=time.perf_counter();ids,noise,sigmas,ts,coeff=inputs(self.model,r)
        key=(r.steps,r.width,r.height)
        if not self.buffer_reuse or key not in self.cache:
            # Reuse immutable schedule/input metadata. MLX's allocator owns physical activation buffers.
            self.cache[key]=(mx.array(np.stack([ts,ts],axis=-1)),mx.array(coeff))
        t,k=self.cache[key];mx.eval(t,k)
        timing={'prepare_time':time.perf_counter()-start};diagnostic_time=0.0
        mx.reset_peak_memory()
        s=time.perf_counter();c=self.text(mx.array(ids)).last_hidden_state;mx.eval(c);timing['text_encoder_time']=time.perf_counter()-s
        s=time.perf_counter();x=mx.array(noise.transpose(0,2,3,1))*float(sigmas[0]);old=mx.zeros_like(x);mx.eval(x,old);timing['latent_init_time']=time.perf_counter()-s
        s=time.perf_counter()
        for i in range(r.steps):
            x,old=self.step(x,old,c,t[i],k[i],mx.array(r.cfg,mx.float32))
            # Submit asynchronously, synchronize once at the phase boundary.
            mx.async_eval(x,old)
        mx.eval(x);mx.synchronize();timing['diffusion_time']=time.perf_counter()-s
        if save_intermediates:
            diagnostic_start=time.perf_counter()
            np.save(str(save_intermediates)+'-conditioning.npy',np.array(c.astype(mx.float32)));np.save(str(save_intermediates)+'-latent.npy',np.array(x).transpose(0,3,1,2))
            diagnostic_time=time.perf_counter()-diagnostic_start
        s=time.perf_counter();out=self.vae.decode(x);mx.eval(out);timing['vae_time']=time.perf_counter()-s
        s=time.perf_counter();out=np.array(out);timing['readback_time']=time.perf_counter()-s
        s=time.perf_counter();image=image_array(out[0]);timing['postprocess_time']=time.perf_counter()-s
        timing['stage_sum_time']=sum(timing.values());timing['seconds_per_step']=timing['diffusion_time']/r.steps
        timing['backend_peak_mb']=mx.get_peak_memory()/2**20;timing['backend_allocated_mb']=mx.get_active_memory()/2**20;timing['backend_cache_mb']=mx.get_cache_memory()/2**20
        timing['wall_time']=time.perf_counter()-start
        timing['diagnostic_time']=diagnostic_time
        timing['total_time']=timing['wall_time']-diagnostic_time
        return image,timing
