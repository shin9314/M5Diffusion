"""Strict, reversible standard SD1.x LoRA weight merging.

CPU parsing and validation precede any model mutation. No tensor is ignored.
Kohya aliases are resolved against real checkpoint keys, rather than guessing
where underscore-separated module names contain literal underscores.
"""
from pathlib import Path
import json
import math
import re
import numpy as np
from safetensors import safe_open


class LoRAError(ValueError):
    pass


def target_index(model):
    model=Path(model)
    config=json.loads((model/'unet/config.json').read_text())
    if config.get('cross_attention_dim') != 768 or config.get('addition_embed_type'):
        raise LoRAError('LoRA対応はSD1.x（CLIP 768次元）のみです。SDXL/SD2は非対応です。')
    targets={}
    for component,filename in [('unet','diffusion_pytorch_model.safetensors'),('text_encoder','model.safetensors')]:
        path=model/component/filename
        if not path.is_file():raise LoRAError(f'ベースモデルの重みが見つかりません: {path}')
        with safe_open(path,framework='np') as file:
            for key in file.keys():
                if key.endswith('.weight'):
                    shape=tuple(file.get_slice(key).get_shape())
                    if len(shape) in (2,4):targets[(component,key[:-7])]=shape
    return targets


def ldm_alias(key):
    """SD1.x two-resblock LDM naming, inverse of Diffusers checkpoint mapping."""
    match=re.match(r'(down_blocks|up_blocks)\.(\d+)\.(resnets|attentions)\.(\d+)\.(.+)',key)
    resnet=False
    if match:
        direction,block,kind,layer,rest=match.groups();block=int(block);layer=int(layer)
        prefix=('input_blocks' if direction=='down_blocks' else 'output_blocks')
        index=3*block+layer+(1 if direction=='down_blocks' else 0)
        key=f'{prefix}.{index}.{0 if kind=="resnets" else 1}.{rest}';resnet=kind=='resnets'
    else:
        match=re.match(r'mid_block\.(resnets|attentions)\.(\d+)\.(.+)',key)
        if match:
            kind,layer,rest=match.groups();index=1 if kind=='attentions' else int(layer)*2
            key=f'middle_block.{index}.{rest}';resnet=kind=='resnets'
        else:
            match=re.match(r'(down_blocks|up_blocks)\.(\d+)\.(downsamplers|upsamplers)\.0\.conv$',key)
            if match:
                direction,block,_=match.groups();block=int(block)
                key=f'input_blocks.{3*block+3}.0.op' if direction=='down_blocks' else f'output_blocks.{3*block+2}.{1 if block==0 else 2}.conv'
            else:
                key={'conv_in':'input_blocks.0.0','conv_out':'out.2','time_embedding.linear_1':'time_embed.0','time_embedding.linear_2':'time_embed.2'}.get(key,key)
    if resnet:
        for old,new in [('conv1','in_layers.2'),('conv2','out_layers.3'),('time_emb_proj','emb_layers.1'),('conv_shortcut','skip_connection')]:
            if key.endswith('.'+old):key=key[:-len(old)]+new
    return 'lora_unet_'+key.replace('.','_')


def aliases_for(targets):
    aliases={}
    for target in targets:
        component,key=target
        names=[f'{component}.{key}']
        prefixes=['lora_unet_'] if component=='unet' else ['lora_te_','lora_te1_']
        names += [p+key.replace('.','_') for p in prefixes]
        if component=='unet':names.append(ldm_alias(key))
        for name in names:
            if name in aliases and aliases[name]!=target:raise LoRAError(f'曖昧なLoRA名: {name}')
            aliases[name]=target
    return aliases


def normalize_key(key):
    key=key.replace('.processor.','.')
    for prefix in ('unet.base_model.model.','text_encoder.base_model.model.'):
        if key.startswith(prefix):key=prefix.split('.')[0]+'.'+key[len(prefix):]
    # PEFT adapter names are accepted only for the conventional default slot.
    key=key.replace('.lora_A.default.weight','.lora_A.weight').replace('.lora_B.default.weight','.lora_B.weight')
    for projection in ('q','k','v','out'):
        target=('out_proj' if projection=='out' else projection+'_proj') if key.startswith('text_encoder.') else ('to_out.0' if projection=='out' else 'to_'+projection)
        key=key.replace('.to_'+projection+'_lora.', '.'+target+'.lora.')
    key=key.replace('.lora_linear_layer.','.lora.')
    for suffix,kind in [('.lora_down.weight','down'),('.lora_up.weight','up'),('.lora_A.weight','down'),('.lora_B.weight','up'),('.lora.down.weight','down'),('.lora.up.weight','up'),('.alpha','alpha')]:
        if key.endswith(suffix):return key[:-len(suffix)],kind
    raise LoRAError(f'非対応または不明なLoRAテンソル: {key}（LoHa/LoKr/DoRA/バイアスは非対応）')


def delta_weight(down,up,target_shape):
    down=np.asarray(down,dtype=np.float32);up=np.asarray(up,dtype=np.float32)
    if down.ndim not in (2,4) or up.ndim not in (2,4):raise LoRAError('LoRAの重みは2次元または4次元が必要です。')
    if not np.isfinite(down).all() or not np.isfinite(up).all():raise LoRAError('LoRAにNaN/Infが含まれています。')
    if down.shape[0]<1 or down.shape[0]!=up.shape[1]:raise LoRAError('LoRA rankの形状が一致しません。')
    if down.ndim==up.ndim==2:
        result=np.einsum('or,ri->oi',up,down,optimize=False)
    else:
        if down.ndim==2:down=down[:,:,None,None]
        if up.ndim==2:up=up[:,:,None,None]
        if up.shape[2:]==(1,1):result=np.einsum('or,rihw->oihw',up[:,:,0,0],down,optimize=False)
        elif down.shape[2:]==(1,1):result=np.einsum('orhw,ri->oihw',up,down[:,:,0,0],optimize=False)
        else:raise LoRAError('畳み込みLoRAはdownまたはupの一方が1×1である必要があります。')
    # Linear LoRA on a 1x1 convolution (or vice versa) is equivalent.
    if result.ndim==2 and len(target_shape)==4 and tuple(target_shape[2:])==(1,1):result=result[:,:,None,None]
    if result.ndim==4 and result.shape[2:]==(1,1) and len(target_shape)==2:result=result[:,:,0,0]
    if tuple(result.shape)!=tuple(target_shape):raise LoRAError(f'LoRA形状{result.shape}がベース重み{target_shape}と一致しません。SDXL/SD2など別系列は使えません。')
    if not np.isfinite(result).all():raise LoRAError('LoRAの積が有限値になりません。')
    return result


def parse_state(state,targets,scale=1.,metadata=None):
    if not math.isfinite(float(scale)):raise LoRAError('LoRA強度は有限の数値が必要です。')
    metadata=metadata or {}
    signature=' '.join(str(metadata.get(k,'')) for k in ('ss_base_model_version','modelspec.architecture','ss_network_module')).lower()
    if any(tag in signature for tag in ('sdxl','stable-diffusion-xl','stable-diffusion-v2','sd_v2','flux','lycoris','loha','lokr','dora')):
        raise LoRAError('このLoRA形式/モデル系列には対応していません。SD1.x標準LoRAを指定してください。')
    aliases=aliases_for(targets);groups={}
    if not state:raise LoRAError('LoRAファイルに重みがありません。')
    for key,value in state.items():
        name,kind=normalize_key(key)
        if name not in aliases:raise LoRAError(f'ベースモデルに存在しないLoRA対象: {name}')
        target=aliases[name];parts=groups.setdefault(target,{})
        if kind in parts:raise LoRAError(f'LoRAの重みが重複しています: {name}/{kind}')
        parts[kind]=np.asarray(value)
    deltas={}
    for target,parts in groups.items():
        if 'down' not in parts or 'up' not in parts:raise LoRAError(f'LoRAのdown/upペアが不足しています: {target}')
        down,up=parts['down'],parts['up']
        if down.ndim not in (2,4) or not down.shape[0]:raise LoRAError(f'不正なLoRA rank: {target}')
        rank=down.shape[0];alpha=parts.get('alpha',np.array(rank))
        if alpha.size!=1 or not np.isfinite(alpha).all():raise LoRAError(f'不正なLoRA alpha: {target}')
        multiplier=float(scale)*float(alpha.reshape(-1)[0])/rank
        delta=delta_weight(down,up,targets[target])*multiplier
        if not np.isfinite(delta).all():raise LoRAError('LoRA強度を適用するとNaN/Infが発生します。')
        deltas[target]=delta
    return deltas


def inspect_lora(path):
    """CPU-only format/finite/rank validation; base-specific checks occur on apply.

    Return {format, modules, tensors, metadata}; raise LoRAError on unsupported
    tensors. This does not certify compatibility with a particular checkpoint.
    """
    path=Path(path)
    if path.suffix.lower()!='.safetensors':raise LoRAError('LoRAは.safetensors形式のみ対応しています。')
    groups={}
    with safe_open(path,framework='pt',device='cpu') as file:
        metadata=file.metadata() or {};keys=list(file.keys())
        signature=' '.join(str(metadata.get(k,'')) for k in ('ss_base_model_version','modelspec.architecture','ss_network_module')).lower()
        if any(tag in signature for tag in ('sdxl','stable-diffusion-xl','stable-diffusion-v2','sd_v2','flux','lycoris','loha','lokr','dora')):raise LoRAError('SD1.x標準LoRAのみ対応しています。')
        for key in keys:
            name,kind=normalize_key(key)
            if not name.startswith(('lora_unet_','lora_te_','lora_te1_','unet.','text_encoder.')):raise LoRAError(f'非対応のLoRA対象: {name}')
            value=file.get_tensor(key).float().numpy()
            if not np.isfinite(value).all():raise LoRAError(f'LoRAにNaN/Infがあります: {key}')
            parts=groups.setdefault(name,{})
            if kind in parts:raise LoRAError(f'重複したLoRAテンソル: {key}')
            parts[kind]=value.shape
        if not groups:raise LoRAError('LoRAの重みがありません。')
        for name,parts in groups.items():
            if 'up' not in parts or 'down' not in parts:raise LoRAError(f'LoRA down/upペアが不足: {name}')
            down,up=parts['down'],parts['up']
            if len(down) not in (2,4) or len(up) not in (2,4) or down[0]<1 or down[0]!=up[1]:raise LoRAError(f'不正なLoRA rank: {name}')
            if len(down)==len(up)==4 and down[2:]!=(1,1) and up[2:]!=(1,1):raise LoRAError('畳み込みLoRAの一方は1×1が必要です。')
            if 'alpha' in parts and np.prod(parts['alpha'])!=1:raise LoRAError(f'不正なLoRA alpha: {name}')
    return {'format':'standard-sd1-lora','modules':len(groups),'tensors':len(keys),'metadata':metadata}


def read_adapter(path,targets,scale):
    path=Path(path)
    if path.suffix.lower()!='.safetensors':raise LoRAError('LoRAは.safetensors形式のみ対応しています。')
    # Torch is used only for safe CPU BF16 decoding; no GPU tensors are created.
    with safe_open(path,framework='pt',device='cpu') as file:
        metadata=file.metadata() or {}
        state={key:file.get_tensor(key).float().numpy() for key in file.keys()}
    return parse_state(state,targets,scale,metadata)


class LoRAManager:
    def __init__(self,engine):
        self.engine=engine;self.originals={};self.active=[];self.targets=None

    def set(self,adapters):
        adapters=[(str(Path(path).resolve()),float(scale)) for path,scale in adapters]
        if adapters==self.active:return
        if getattr(self.engine,'quantize',0):raise LoRAError('量子化モデルへのLoRA適用は非対応です。標準FP16モデルを使用してください。')
        if not adapters and not self.originals:self.active=[];return
        if self.targets is None:self.targets=target_index(self.engine.model)
        combined={}
        for path,scale in adapters:
            for target,delta in read_adapter(path,self.targets,scale).items():
                combined[target]=combined.get(target,0)+delta
        mx=self.engine.mx
        from mlx.utils import tree_flatten
        from stable_diffusion.model_io import map_unet_weights,map_clip_text_encoder_weights
        modules={'unet':self.engine.unet,'text_encoder':self.engine.text}
        current={component:dict(tree_flatten(module.parameters())) for component,module in modules.items()}
        mapped={}
        for (component,key),delta in combined.items():
            mapper=map_unet_weights if component=='unet' else map_clip_text_encoder_weights
            for name,value in mapper(key+'.weight',mx.array(delta)):
                target=(component,name)
                if name not in current[component] or tuple(current[component][name].shape)!=tuple(value.shape):raise LoRAError(f'MLXモデルのLoRA対象/形状が一致しません: {target}')
                if target in mapped:raise LoRAError(f'MLX変換後のLoRA対象が重複しています: {target}')
                mapped[target]=value
        # Keep originals only for currently targeted parameters. Prepare all
        # candidate values before mutating either module (including removals).
        originals={target:self.originals.get(target,current[target[0]][target[1]]) for target in mapped}
        updates=dict(self.originals)
        for target,value in mapped.items():
            original=originals[target]
            updates[target]=(original.astype(mx.float32)+value).astype(original.dtype)
        mx.eval(list(updates.values()))
        for value in updates.values():
            if not bool(mx.all(mx.isfinite(value)).item()):raise LoRAError('LoRA適用後の重みがFP16範囲を超えています。強度を下げてください。')
        rollback={target:current[target[0]][target[1]] for target in updates}
        try:
            for component,module in modules.items():module.load_weights([(name,value) for (part,name),value in updates.items() if part==component],strict=False)
        except Exception:
            for component,module in modules.items():module.load_weights([(name,value) for (part,name),value in rollback.items() if part==component],strict=False)
            raise
        self.originals=originals;self.active=adapters
        self.engine._refresh_step()
