from pathlib import Path
import time
import numpy as np
from .common import inputs,image_array

class TorchEngine:
    backend='torch-mps'
    def __init__(self,model,**kw):
        import torch
        from diffusers import UNet2DConditionModel,AutoencoderKL
        from transformers import CLIPTextModel
        torch.mps.set_per_process_memory_fraction(0.42)
        self.torch=torch;self.model=Path(model);t=time.perf_counter()
        self.unet=UNet2DConditionModel.from_pretrained(model,subfolder='unet',torch_dtype=torch.float16,local_files_only=True).to('mps').eval()
        self.text=CLIPTextModel.from_pretrained(model,subfolder='text_encoder',torch_dtype=torch.float16,local_files_only=True).to('mps').eval()
        self.vae=AutoencoderKL.from_pretrained(model,subfolder='vae',torch_dtype=torch.float32,local_files_only=True).to('mps').eval()
        torch.mps.synchronize();self.load_time=time.perf_counter()-t
    def generate(self,r,save_intermediates=None):
        t=self.torch;start=time.perf_counter();ids,noise,sigmas,ts,coeff=inputs(self.model,r)
        timing={'prepare_time':time.perf_counter()-start};diagnostic_time=0.0
        with t.inference_mode():
            s=time.perf_counter();c=self.text(t.tensor(ids,device='mps'))[0];t.mps.synchronize();timing['text_encoder_time']=time.perf_counter()-s
            s=time.perf_counter();x=t.tensor(noise,device='mps')*float(sigmas[0]);old=t.zeros_like(x);t.mps.synchronize();timing['latent_init_time']=time.perf_counter()-s
            s=time.perf_counter()
            for i,(sigma,ratio,factor,ca,cb) in enumerate(coeff):
                u=(t.cat([x,x])/float(np.sqrt(1+sigma*sigma))).half()
                pred=self.unet(u,t.tensor([ts[i],ts[i]],device='mps'),encoder_hidden_states=c).sample.float()
                neg,pos=pred.chunk(2);eps=neg+r.cfg*(pos-neg)
                den=x-float(sigma)*eps
                x=float(ratio)*x+float(factor)*(float(ca)*den+float(cb)*old)
                old=den
            t.mps.synchronize();timing['diffusion_time']=time.perf_counter()-s
            if save_intermediates:
                diagnostic_start=time.perf_counter()
                np.save(str(save_intermediates)+'-conditioning.npy',c.float().cpu().numpy());np.save(str(save_intermediates)+'-latent.npy',x.cpu().numpy())
                diagnostic_time=time.perf_counter()-diagnostic_start
            s=time.perf_counter();out=self.vae.decode(x/self.vae.config.scaling_factor).sample;t.mps.synchronize();timing['vae_time']=time.perf_counter()-s
            s=time.perf_counter();out=out.cpu().numpy().transpose(0,2,3,1);timing['readback_time']=time.perf_counter()-s
        s=time.perf_counter();image=image_array(out[0]);timing['postprocess_time']=time.perf_counter()-s
        timing['stage_sum_time']=sum(timing.values());timing['seconds_per_step']=timing['diffusion_time']/r.steps
        timing['backend_allocated_mb']=t.mps.current_allocated_memory()/2**20
        timing['backend_driver_mb']=t.mps.driver_allocated_memory()/2**20
        timing['wall_time']=time.perf_counter()-start
        timing['diagnostic_time']=diagnostic_time
        timing['total_time']=timing['wall_time']-diagnostic_time
        return image,timing
