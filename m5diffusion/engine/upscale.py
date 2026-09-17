"""Local Real-ESRGAN SRVGG upscaling; no Torch/GPU initialization at import.

Network topology follows Xintao Wang's BSD-3-Clause Real-ESRGAN SRVGGNetCompact.
See docs/third-party/Real-ESRGAN-LICENSE.txt and docs/upscale.md.
"""
from pathlib import Path
import gc
import hashlib
import math
import time
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[2]
MODEL_PATH=ROOT/'models/upscale/realesr-general-x4v3.pth'
MODEL_SHA256='8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292'
MAX_INPUT_PIXELS=4_000_000
MAX_OUTPUT_PIXELS=16_000_000
MAX_OUTPUT_SIDE=4096
TILE_SIZE=192
# 34 successive 3x3 convolutions have a 69x69 low-resolution receptive field.
TILE_PAD=34


def validate_image(image,scale):
    """Validate limits without importing Torch or touching a GPU."""
    if not isinstance(image,Image.Image):raise ValueError('画像ファイルを指定してください。')
    if not isinstance(scale,int) or isinstance(scale,bool) or scale not in (2,4):raise ValueError('拡大率は2倍または4倍を選択してください。')
    width,height=image.size
    if width<1 or height<1:raise ValueError('画像のサイズが不正です。')
    if width*height>MAX_INPUT_PIXELS:raise ValueError('入力画像は約400万画素以下にしてください。')
    if width*height*scale*scale>MAX_OUTPUT_PIXELS:raise ValueError('拡大後は約1600万画素以下にしてください。画像を小さくするか2倍を選んでください。')
    if max(width,height)*scale>MAX_OUTPUT_SIDE:raise ValueError('拡大後の縦・横は4096ピクセル以下にしてください。')


def build_model(num_feat=64,num_conv=32):
    """Construct the upstream-compatible network; imports Torch only on demand."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    class SRVGGNetCompact(nn.Module):
        def __init__(self):
            super().__init__();self.receptive_radius=num_conv+2
            layers=[nn.Conv2d(3,num_feat,3,1,1),nn.PReLU(num_feat)]
            for _ in range(num_conv):layers.extend([nn.Conv2d(num_feat,num_feat,3,1,1),nn.PReLU(num_feat)])
            layers.append(nn.Conv2d(num_feat,3*4*4,3,1,1))
            self.body=nn.ModuleList(layers);self.upsampler=nn.PixelShuffle(4)
        def forward(self,x):
            out=x
            for layer in self.body:out=layer(out)
            return self.upsampler(out)+F.interpolate(x,scale_factor=4,mode='nearest')
    return SRVGGNetCompact()


def _load_model():
    import torch
    if not MODEL_PATH.is_file():raise RuntimeError('AI拡大モデルが見つかりません。realesr-general-x4v3.pthを復元してください。')
    with MODEL_PATH.open('rb') as file:digest=hashlib.file_digest(file,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(file.read()).hexdigest()
    if digest!=MODEL_SHA256:raise RuntimeError('AI拡大モデルの検証に失敗しました。公式モデルを再配置してください。')
    checkpoint=torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
    weights=checkpoint.get('params_ema',checkpoint.get('params'))
    if weights is None:raise RuntimeError('AI拡大モデルの重み形式が不正です。')
    model=build_model();model.load_state_dict(weights,strict=True);model.eval();model.requires_grad_(False)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    dtype=torch.float16 if device.type=='mps' else torch.float32
    try:model=model.to(device=device,dtype=dtype)
    except Exception:
        model=None;gc.collect()
        if device.type=='mps':torch.mps.empty_cache()
        raise
    return model,device,dtype


def _run_tiled(rgb,model,device,dtype,tile_size=TILE_SIZE,tile_pad=TILE_PAD,progress=None):
    """Crop overlapping input halos, infer, and keep disjoint output cores.

    Halo >= network receptive radius makes each retained pixel independent of
    artificial internal tile boundaries. True image edges retain zero padding.
    Only one tile and its activations occupy GPU memory at a time.
    """
    import torch
    if tile_pad<model.receptive_radius:raise ValueError('Tile halo must cover the complete network receptive radius.')
    height,width=rgb.shape[:2];output=np.empty((height*4,width*4,3),dtype=np.uint8)
    total=math.ceil(width/tile_size)*math.ceil(height/tile_size);done=0
    with torch.inference_mode():
        for y in range(0,height,tile_size):
            for x in range(0,width,tile_size):
                right=min(x+tile_size,width);bottom=min(y+tile_size,height)
                x0=max(x-tile_pad,0);y0=max(y-tile_pad,0);x1=min(right+tile_pad,width);y1=min(bottom+tile_pad,height)
                patch=np.ascontiguousarray(rgb[y0:y1,x0:x1].transpose(2,0,1))
                tensor=torch.from_numpy(patch).unsqueeze(0).to(device=device,dtype=dtype)/255.
                prediction=model(tensor)
                core=prediction[0,:,4*(y-y0):4*(bottom-y0),4*(x-x0):4*(right-x0)]
                cpu=core.detach().float().cpu().numpy().transpose(1,2,0)
                if not np.isfinite(cpu).all():raise RuntimeError('AI拡大処理で不正な値が発生しました。')
                output[y*4:bottom*4,x*4:right*4]=np.rint(np.clip(cpu,0,1)*255).astype(np.uint8)
                del tensor,prediction,core,cpu
                done+=1
                if progress:progress(.08+.84*done/total,f'AIで細部を復元中… {done}/{total}')
    return output,total


def upscale_image(image,scale,progress=None):
    """Return (PIL output, metadata); callback(fraction_0_to_1, message).

    4x is native neural output. 2x runs the same 4x model then downsamples once
    with Lanczos. Alpha is independently resized and retained, never discarded.
    Model and GPU allocations are released after each job, including failures.
    """
    start=time.perf_counter();validate_image(image,scale)
    image=ImageOps.exif_transpose(image);validate_image(image,scale)
    alpha_present='A' in image.getbands() or 'transparency' in image.info
    rgba=image.convert('RGBA') if alpha_present else None
    alpha=rgba.getchannel('A') if rgba is not None else None
    rgb=np.asarray((rgba or image).convert('RGB'),dtype=np.uint8)
    model=None;device=None
    import torch
    try:
        if progress:progress(.01,'AI拡大モデルを読み込み中…')
        load_start=time.perf_counter();model,device,dtype=_load_model();load_seconds=time.perf_counter()-load_start
        if progress:progress(.08,'AIで細部を復元します…')
        infer_start=time.perf_counter();output,tiles=_run_tiled(rgb,model,device,dtype,progress=progress);inference_seconds=time.perf_counter()-infer_start
        result=Image.fromarray(output)
        size=(image.width*scale,image.height*scale)
        if scale==2:result=result.resize(size,Image.Resampling.LANCZOS)
        if alpha is not None:result.putalpha(alpha.resize(size,Image.Resampling.LANCZOS))
        if image.mode in ('RGB','RGBA') and 'icc_profile' in image.info:result.info['icc_profile']=image.info['icc_profile']
        metadata={'model':'realesr-general-x4v3','model_sha256':MODEL_SHA256,'scale':scale,'native_scale':4,'input_size':list(image.size),'output_size':list(result.size),'device':str(device),'precision':'fp16' if dtype==torch.float16 else 'fp32','tile_size':TILE_SIZE,'tile_pad':TILE_PAD,'tiles':tiles,'model_load_seconds':load_seconds,'inference_seconds':inference_seconds,'alpha_preserved':alpha_present,'alpha_method':'Lanczos interpolation' if alpha_present else None,'method':'Real-ESRGAN neural 4x'+(' + Lanczos downsample to 2x' if scale==2 else '')}
    finally:
        model=None;gc.collect()
        if device is not None and device.type=='mps':
            torch.mps.synchronize();torch.mps.empty_cache()
    metadata['total_seconds']=time.perf_counter()-start
    if progress:progress(1.,'AI拡大が完了しました')
    return result,metadata
