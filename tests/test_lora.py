import os
from types import SimpleNamespace
import numpy as np
import pytest
from safetensors.numpy import save_file
from m5diffusion.engine.lora import LoRAError,parse_state,normalize_key,delta_weight,ldm_alias,inspect_lora,read_adapter

TARGET='down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q'
TARGETS={('unet',TARGET):(3,4),('text_encoder','text_model.encoder.layers.0.self_attn.q_proj'):(3,4)}

@pytest.mark.parametrize('prefix,suffix',[('unet.'+TARGET,'.lora_A.weight'),('lora_unet_'+TARGET.replace('.','_'),'.lora_down.weight'),(ldm_alias(TARGET),'.lora_down.weight'),('unet.'+TARGET,'.lora_A.default.weight')])
def test_standard_formats(prefix,suffix):
    other=suffix.replace('lora_A','lora_B').replace('lora_down','lora_up')
    state={prefix+suffix:np.ones((2,4)),prefix+other:np.ones((3,2)),prefix+'.alpha':np.array(1)}
    result=parse_state(state,TARGETS,scale=.5)
    np.testing.assert_allclose(result[('unet',TARGET)],.5)

@pytest.mark.parametrize('key,expected',[('unet.x.attn1.processor.to_q_lora.down.weight',('unet.x.attn1.to_q','down')),('text_encoder.x.self_attn.to_q_lora.up.weight',('text_encoder.x.self_attn.q_proj','up')),('text_encoder.x.mlp.fc1.lora_linear_layer.down.weight',('text_encoder.x.mlp.fc1','down'))])
def test_legacy_names(key,expected):assert normalize_key(key)==expected

def test_ldm_resnet_and_samplers():
    assert ldm_alias('down_blocks.1.resnets.1.conv1')=='lora_unet_input_blocks_5_0_in_layers_2'
    assert ldm_alias('mid_block.resnets.1.conv2')=='lora_unet_middle_block_2_out_layers_3'
    assert ldm_alias('down_blocks.0.downsamplers.0.conv')=='lora_unet_input_blocks_3_0_op'
    assert ldm_alias('up_blocks.0.upsamplers.0.conv')=='lora_unet_output_blocks_2_1_conv'
    assert ldm_alias('up_blocks.1.upsamplers.0.conv')=='lora_unet_output_blocks_5_2_conv'

@pytest.mark.parametrize('orientation',[0,1])
def test_conv(orientation):
    rng=np.random.default_rng(5)
    down=rng.normal(size=(2,4,3,3) if orientation==0 else (2,4,1,1))
    up=rng.normal(size=(3,2,1,1) if orientation==0 else (3,2,3,3))
    actual=delta_weight(down,up,(3,4,3,3))
    expected=np.zeros((3,4,3,3))
    for o in range(3):
        for i in range(4):
            for r in range(2):expected[o,i]+=up[o,r]*down[r,i]
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)

@pytest.mark.parametrize('extra',[{'unet.x.dora_scale':np.ones(1)},{'lora_te2_x.lora_down.weight':np.ones((2,4))},{'unet.no_such_layer.lora_A.weight':np.ones((2,4))}])
def test_unknown_never_ignored(extra):
    with pytest.raises(LoRAError):parse_state(extra,TARGETS)

def test_bad_pair_shape_metadata_nonfinite():
    base='unet.'+TARGET
    good={base+'.lora_A.weight':np.ones((2,4)),base+'.lora_B.weight':np.ones((3,2))}
    for state,kwargs in [(dict(list(good.items())[:1]),{}),({**good,base+'.alpha':np.array([1,2])},{}),(good,{'metadata':{'ss_base_model_version':'sdxl_base_v1-0'}}),(good,{'scale':float('nan')}),({**good,base+'.lora_B.weight':np.full((3,2),np.nan)},{}),({**good,base+'.lora_B.weight':np.ones((5,2))},{})]:
        with pytest.raises(LoRAError):parse_state(state,TARGETS,**kwargs)

def test_file_inspection_and_loading(tmp_path):
    path=tmp_path/'small.safetensors';base='unet.'+TARGET
    save_file({base+'.lora_A.weight':np.ones((2,4),np.float32),base+'.lora_B.weight':np.ones((3,2),np.float32)},str(path))
    assert inspect_lora(path)['modules']==1
    np.testing.assert_allclose(read_adapter(path,TARGETS,1.)[('unet',TARGET)],2.)

@pytest.mark.skipif(os.environ.get('M5DIFFUSION_GPU_TESTS')!='1',reason='explicit exclusive GPU slot required')
def test_tiny_merge_restore_multiscale_and_compile(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    from m5diffusion.engine.lora import LoRAManager
    from m5diffusion.engine.mlx_backend import MLXEngine  # installs vendor import path
    from stable_diffusion.model_io import map_unet_weights
    unet=nn.Module();unet.query_proj=nn.Linear(4,3,bias=False)
    text=nn.Module();text.placeholder=nn.Linear(1,1,bias=False)
    engine=SimpleNamespace(mx=mx,unet=unet,text=text,quantize=0,model=tmp_path,compiled=True)
    def refresh():engine.forward=mx.compile(lambda x:engine.unet.query_proj(x))
    engine._refresh_step=refresh;refresh()
    mgr=LoRAManager(engine);mgr.targets={('unet','to_q'):(3,4)}
    before=unet.query_proj.weight;mx.eval(before);x=mx.ones((1,4));baseline=engine.forward(x);mx.eval(baseline)
    paths=[]
    for index,value in enumerate([1.,2.]):
        path=tmp_path/f'adapter{index}.safetensors'
        save_file({'unet.to_q.lora_A.weight':np.ones((2,4),np.float32)*value,'unet.to_q.lora_B.weight':np.ones((3,2),np.float32)},str(path));paths.append(path)
    mgr.set([(paths[0],.5),(paths[1],.25)])
    np.testing.assert_allclose(np.array(unet.query_proj.weight),np.array(before)+2.,atol=1e-6)
    np.testing.assert_allclose(np.array(engine.forward(x)),np.array(baseline)+8.,atol=1e-5)
    mgr.set([(paths[0],1.)]);np.testing.assert_allclose(np.array(unet.query_proj.weight),np.array(before)+2.,atol=1e-6)
    bad=tmp_path/'bad.safetensors';save_file({'unet.to_q.lora_A.weight':np.ones((2,5),np.float32),'unet.to_q.lora_B.weight':np.ones((3,2),np.float32)},str(bad))
    with pytest.raises(LoRAError):mgr.set([(bad,1.)])
    np.testing.assert_allclose(np.array(unet.query_proj.weight),np.array(before)+2.,atol=1e-6)
    mgr.set([]);np.testing.assert_array_equal(np.array(unet.query_proj.weight),np.array(before));assert mgr.originals=={}
    np.testing.assert_allclose(np.array(engine.forward(x)),np.array(baseline),atol=1e-6)
    # GEGLU delta uses the same mapper as the base model and splits rows exactly.
    pieces=map_unet_weights('ff.net.0.proj.weight',mx.array(np.arange(24).reshape(6,4),dtype=mx.float32))
    assert [name for name,_ in pieces]==['linear1.weight','linear2.weight']
    np.testing.assert_array_equal(np.array(pieces[0][1]),np.arange(24).reshape(6,4)[:3])
