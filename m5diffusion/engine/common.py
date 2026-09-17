from dataclasses import dataclass,asdict
from pathlib import Path
import json,hashlib,time
import math
from numbers import Integral, Real
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
@dataclass(frozen=True)
class Request:
    prompt:str='a small cozy cabin beside a lake, mountains in the background, golden morning light, landscape photography'
    negative_prompt:str=''
    seed:int=12345
    width:int=512
    height:int=512
    steps:int=20
    cfg:float=7.0
    def validate(self):
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (self.width, self.height, self.steps, self.seed)):
            raise ValueError('dimensions, steps and seed must be integers')
        if self.seed < 0:raise ValueError('seed must be nonnegative')
        if not isinstance(self.prompt, str) or not isinstance(self.negative_prompt, str):raise ValueError('prompts must be strings')
        if isinstance(self.cfg, bool) or not isinstance(self.cfg, Real) or not math.isfinite(self.cfg):raise ValueError('CFG must be a finite number')
        if self.width%64 or self.height%64 or not 64<=self.width<=1024 or not 64<=self.height<=1024:raise ValueError('dimensions must be multiples of 64 between 64 and 1024')
        if not 2<=self.steps<=100:raise ValueError('steps must be 2..100')
        if not 1<=self.cfg<=20:raise ValueError('CFG must be 1..20')

def inputs(model,r):
    from transformers import CLIPTokenizer
    from m5diffusion.scheduler.dpm import initial_noise,schedule
    r.validate()
    tok=CLIPTokenizer.from_pretrained(str(model/'tokenizer'),local_files_only=True)
    ids=tok([r.negative_prompt,r.prompt],padding='max_length',max_length=77,truncation=True,return_tensors='np').input_ids
    noise=initial_noise(r.seed,r.width,r.height)
    config=json.loads((model/'scheduler/scheduler_config.json').read_text())
    sigmas,ts,coeff=schedule(r.steps,config)
    return ids,noise,sigmas,ts,coeff

def image_array(x):return np.clip(x/2+0.5,0,1)
