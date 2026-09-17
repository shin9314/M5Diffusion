import argparse,cProfile,csv,dataclasses,hashlib,json,pstats,time
from pathlib import Path
import numpy as np
from PIL import Image
from m5diffusion.engine.common import Request,ROOT
from m5diffusion.profiling.telemetry import Sampler,system_snapshot,thermal_snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--native-conv',action='store_true');p.add_argument('--backend',choices=['mlx','mps'],default='mlx');p.add_argument('--model',type=Path,default=ROOT/'models/sd15');p.add_argument('--width',type=int,default=512);p.add_argument('--height',type=int,default=512);p.add_argument('--steps',type=int,default=20);p.add_argument('--seed',type=int,default=12345);p.add_argument('--cfg',type=float,default=7);p.add_argument('--prompt',default=Request.prompt);p.add_argument('--negative-prompt',default='');p.add_argument('--repeat',type=int,default=2);p.add_argument('--compile',action='store_true');p.add_argument('--no-buffer-reuse',action='store_true');p.add_argument('--quantize',type=int,choices=[0,4,8],default=0);p.add_argument('--label',default='run');p.add_argument('--profile',action='store_true');p.add_argument('--attention',choices=['standard','padded'],default='standard');args=p.parse_args()
 r=Request(args.prompt,args.negative_prompt,args.seed,args.width,args.height,args.steps,args.cfg);r.validate()
 if args.repeat<1:raise ValueError('repeat must be positive')
 if args.backend=='mps' and (args.quantize or args.compile or args.attention!='standard'):p.error('quantization, compilation and padded attention are MLX-only options')
 precision='UNet+CLIP fp16; VAE+latents fp32'
 if args.quantize:precision+=f'; UNet+CLIP Linear weights affine {args.quantize}-bit group32 (convolutions unchanged)' 
 output=ROOT/'benchmark/results';output.mkdir(parents=True,exist_ok=True)
 run_id=time.strftime('%Y%m%d-%H%M%S')+'-'+args.label
 before=system_snapshot()
 if args.backend=='mps':
  from m5diffusion.engine.torch_backend import TorchEngine
  engine=TorchEngine(args.model)
 else:
  from m5diffusion.engine.mlx_backend import MLXEngine
  engine=MLXEngine(args.model,compiled=args.compile,buffer_reuse=not args.no_buffer_reuse,quantize=args.quantize,attention=args.attention,convolution=not args.native_conv)
 print(json.dumps({'model_load_time':engine.load_time,'backend':args.backend}),flush=True)
 rows=[]
 for i in range(args.repeat):
  prefix=output/f'{run_id}-{i}'
  thermal_before=thermal_snapshot()
  with Sampler() as telemetry:
   profiler=cProfile.Profile() if args.profile else None
   if profiler:profiler.enable()
   image,timing=engine.generate(r,save_intermediates=prefix if i==0 else None)
   if profiler:
    profiler.disable();profiler.dump_stats(str(prefix)+'.prof')
    with open(str(prefix)+'-cpu-profile.txt','w') as f:pstats.Stats(profiler,stream=f).strip_dirs().sort_stats('cumulative').print_stats(10)
  if not np.isfinite(image).all() or image.std()<.01:raise RuntimeError('Invalid generated image')
  s=time.perf_counter();Image.fromarray(np.rint(image*255).astype(np.uint8)).save(str(prefix)+'.png');png_time=time.perf_counter()-s
  row={'run':i,'cold':i==0,'backend':args.backend,'compiled':args.compile,'attention':args.attention,'convolution':'native' if args.native_conv else 'im2col','quantize':args.quantize,'buffer_reuse':not args.no_buffer_reuse,'thermal_before':thermal_before,'thermal_after':thermal_snapshot(),**timing,**telemetry.summary(),'png_time':png_time,'image':prefix.name+'.png'};rows.append(row);print(json.dumps(row),flush=True)
  result={'request':dataclasses.asdict(r),'model':str(args.model),'sampler':'DPM++ 2M','schedule':'Karras rho=7','rng':'NumPy PCG64 float32 NCHW shared bytes','precision':precision,'timing_schema':'generation-wall-v2','model_load_time':engine.load_time,'before':before,'after':system_snapshot(),'runs':rows,'metric_notes':{'rss':'Process RSS includes shared pages; do not add GPU allocations to RSS','gpu_peak':'MLX peak allocation is not total system unified-memory peak; MPS peak unavailable','unavailable_metrics':'Temperature, power and frequency are null; GPU utilization is whole-device driver telemetry when available; no privileged collector used','timing':'GPU synchronized at stage boundaries. total_time is generation wall time minus measured diagnostic .npy transfer/save overhead; includes input preparation, latent initialization, postprocessing and telemetry queries. wall_time includes diagnostic overhead. stage_sum_time sums named stages. Model load and PNG saving are separate. First run includes kernel warmup.'}}
  (output/f'{run_id}.json').write_text(json.dumps(result,indent=2))
  with (output/f'{run_id}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
