import io
import numpy as np
import pytest
from safetensors.numpy import save
from m5diffusion.models.library import Library, SOURCE_HASH

def adapter():
 return save({'lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight':np.ones((2,4),np.float32),'lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight':np.ones((4,2),np.float32)})

def test_atomic_dedup_and_fresh_catalog(tmp_path):
 lib=Library(tmp_path);blob=adapter()
 first=lib.ingest(io.BytesIO(blob),len(blob),'lora','one.safetensors')
 second=lib.ingest(io.BytesIO(blob),len(blob),'lora','two.safetensors')
 assert first['id']==second['id']
 assert len(list(lib.loras.glob('*.safetensors')))==1
 assert Library(tmp_path).catalog()['loras'][0]['id']==first['id']
 assert lib.resolve_lora(first['id']).read_bytes()==blob
 assert not list(lib.loras.glob('.upload-*'))

def test_interrupted_upload_leaves_nothing(tmp_path):
 lib=Library(tmp_path)
 with pytest.raises(ValueError,match='途中'):lib.ingest(io.BytesIO(b'bad'),100,'model','test.safetensors')
 assert not list(lib.checkpoints.iterdir())

@pytest.mark.parametrize('name',['../test.safetensors','a/b.safetensors','test.ckpt','a\\b.safetensors'])
def test_unsafe_filename(tmp_path,name):
 with pytest.raises(ValueError):Library(tmp_path).ingest(io.BytesIO(b'x'),1,'model',name)

def test_wrong_format_and_incomplete_models_rejected(tmp_path):
 lib=Library(tmp_path)
 for blob in (b'pickle data',save({'model.diffusion_model.input_blocks.0.0.weight':np.zeros((1,),np.float32)})):
  with pytest.raises(ValueError):lib.ingest(io.BytesIO(blob),len(blob),'model','test.safetensors')
 assert not list(lib.checkpoints.iterdir())

def test_sdxl_rejected(tmp_path):
 lib=Library(tmp_path);blob=save({'conditioner.foo':np.zeros(1,np.float32)})
 with pytest.raises(ValueError,match='SDXL'):lib.ingest(io.BytesIO(blob),len(blob),'model','sdxl.safetensors')

def test_local_import_symlink_and_traversal_rejected(tmp_path):
 lib=Library(tmp_path/'app');source=tmp_path/'source.safetensors';source.write_bytes(adapter())
 link=tmp_path/'link.safetensors';link.symlink_to(source)
 with pytest.raises(ValueError):lib.import_local(str(link),'lora')
 with pytest.raises(ValueError):lib.import_local(str(tmp_path/'x/../source.safetensors'),'lora')
 entry=lib.import_local(str(source),'lora');assert lib.resolve_lora(entry['id']).read_bytes()==source.read_bytes()
 with pytest.raises(ValueError):lib.resolve_model('../sd15')

def test_real_reference_import_reuses_existing_model_without_copy():
 from m5diffusion.engine.common import ROOT
 import os
 from pathlib import Path
 reference=os.environ.get('M5DIFFUSION_REFERENCE_CHECKPOINT')
 if not reference:pytest.skip('set M5DIFFUSION_REFERENCE_CHECKPOINT for optional real-checkpoint reuse verification')
 source=Path(reference)
 if not source.is_file():pytest.skip('configured reference checkpoint unavailable')
 lib=Library(ROOT)
 entry=lib.import_local(str(source),'model')
 assert entry['id']=='sd15'
 assert lib.resolve_model('sd15')==ROOT/'models/sd15'
 assert not (lib.checkpoints/(SOURCE_HASH+'.safetensors')).exists()


def test_catalog_reports_invalid_manual_drop(tmp_path):
 lib=Library(tmp_path)
 (lib.checkpoints/'broken.safetensors').write_bytes(b'bad')
 data=lib.catalog()
 assert data['models']==[]
 assert data['errors'][0]['name']=='broken.safetensors'
 assert data['errors'][0]['error']

def test_existing_dedupe_symlink_rejected(tmp_path):
 import hashlib
 lib=Library(tmp_path/'app');blob=adapter()
 source=tmp_path/'original.safetensors';source.write_bytes(blob)
 (lib.loras/(hashlib.sha256(blob).hexdigest()+'.safetensors')).symlink_to(source)
 with pytest.raises(ValueError,match='シンボリック'):lib.ingest(io.BytesIO(blob),len(blob),'lora','safe.safetensors')
 assert source.read_bytes()==blob


def test_standard_te1_alias_is_not_automatically_sdxl(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file
    from m5diffusion.models.library import validate_safetensors
    path=tmp_path/'te1.safetensors'
    prefix='lora_te1_text_model_encoder_layers_0_self_attn_q_proj'
    save_file({prefix+'.lora_down.weight':np.zeros((1,768),dtype=np.float32),prefix+'.lora_up.weight':np.zeros((768,1),dtype=np.float32)},str(path))
    validate_safetensors(path,'lora')
