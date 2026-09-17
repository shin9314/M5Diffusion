import os
import subprocess
import sys
import numpy as np
import pytest
from PIL import Image
from m5diffusion.engine import upscale


def test_validation_no_torch_import():
    subprocess.run([sys.executable,'-c','import sys; from m5diffusion.engine.upscale import validate_image; from PIL import Image; validate_image(Image.new("RGB",(32,32)),4); assert "torch" not in sys.modules'],check=True)

@pytest.mark.parametrize('size,scale',[((2048,2048),4),((2049,10),2),((1025,1024),4),((10,10),3),((10,10),True)])
def test_limits(size,scale):
    with pytest.raises(ValueError):upscale.validate_image(Image.new('RGB',size),scale)


def test_official_architecture_and_weights():
    import torch
    model=upscale.build_model();checkpoint=torch.load(upscale.MODEL_PATH,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint.get('params_ema',checkpoint.get('params')),strict=True)
    assert sum(isinstance(m,torch.nn.Conv2d) for m in model.modules())==34
    assert model.receptive_radius==34
    assert len(model.body)==67


def test_tiled_matches_full_cpu():
    import torch
    torch.manual_seed(9);model=upscale.build_model(num_feat=4,num_conv=2).eval()
    rgb=np.random.default_rng(4).integers(0,256,size=(17,21,3),dtype=np.uint8)
    with torch.inference_mode():full=model(torch.from_numpy(rgb.transpose(2,0,1).copy()).unsqueeze(0).float()/255.)
    reference=np.rint(np.clip(full[0].numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
    tiled,count=upscale._run_tiled(rgb,model,torch.device('cpu'),torch.float32,tile_size=8,tile_pad=4)
    assert count==9
    np.testing.assert_allclose(tiled.astype(float),reference.astype(float),atol=1)
    with pytest.raises(ValueError):upscale._run_tiled(rgb,model,torch.device('cpu'),torch.float32,tile_size=8,tile_pad=3)

@pytest.mark.parametrize('scale',[2,4])
def test_alpha_progress_and_sizes(monkeypatch,scale):
    import torch
    model=upscale.build_model(num_feat=4,num_conv=0).eval()
    monkeypatch.setattr(upscale,'_load_model',lambda:(model,torch.device('cpu'),torch.float32))
    image=Image.new('RGBA',(7,5),(80,90,100,128));alpha=np.arange(35,dtype=np.uint8).reshape(5,7)*7;image.putalpha(Image.fromarray(alpha))
    progress=[];output,metadata=upscale.upscale_image(image,scale,lambda fraction,message:progress.append((fraction,message)))
    assert output.mode=='RGBA' and output.size==(7*scale,5*scale)
    np.testing.assert_array_equal(np.array(output.getchannel('A')),np.array(image.getchannel('A').resize(output.size,Image.Resampling.LANCZOS)))
    assert metadata['alpha_preserved'] and metadata['native_scale']==4
    assert progress[-1][0]==1 and all(a[0]<=b[0] for a,b in zip(progress,progress[1:]))

@pytest.mark.skipif(os.environ.get('M5DIFFUSION_GPU_TESTS')!='1',reason='exclusive GPU opt-in required')
def test_real_model_gpu():
    image=Image.fromarray(np.random.default_rng(8).integers(0,256,(32,40,3),dtype=np.uint8))
    output,metadata=upscale.upscale_image(image,4)
    assert output.size==(160,128) and metadata['device']=='mps'
    assert np.std(np.array(output))>1
    assert not np.array_equal(np.array(output),np.array(image.resize(output.size,Image.Resampling.LANCZOS)))
