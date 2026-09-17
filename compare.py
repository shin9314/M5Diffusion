import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
p=argparse.ArgumentParser();p.add_argument('a',type=Path);p.add_argument('b',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
x=np.asarray(Image.open(a.a).convert('RGB'),dtype=np.float32)/255;y=np.asarray(Image.open(a.b).convert('RGB'),dtype=np.float32)/255
r={'a':str(a.a),'b':str(a.b),'psnr':float(peak_signal_noise_ratio(x,y,data_range=1)),'ssim':float(structural_similarity(x,y,data_range=1,channel_axis=-1)),'pixel_mae':float(np.abs(x-y).mean()),'pixel_max_abs':float(np.abs(x-y).max())}
for part in ['conditioning','latent']:
 pa=a.a.with_name(a.a.stem+'-'+part+'.npy');pb=a.b.with_name(a.b.stem+'-'+part+'.npy')
 if pa.exists() and pb.exists():
  u=np.load(pa);v=np.load(pb);r[part]={'rmse':float(np.sqrt(np.mean((u-v)**2))),'max_abs':float(np.max(np.abs(u-v))),'relative_l2':float(np.linalg.norm(u-v)/max(np.linalg.norm(u),1e-12))}
print(json.dumps(r,indent=2))
if a.output:a.output.write_text(json.dumps(r,indent=2))
