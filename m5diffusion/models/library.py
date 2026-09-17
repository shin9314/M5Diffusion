"""Local-only SD1.5 library. No pickle loaders or automatic downloads."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from m5diffusion.engine.common import ROOT

LIMITS = {'model': 8 * 2**30, 'lora': 2 * 2**30}
CHUNK = 4 * 2**20
SOURCE_HASH = '6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa'

def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()

def safe_path(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError('シンボリックリンクは取り込めません')
    if not path.is_file():
        raise ValueError('通常のローカルファイルを指定してください')
    return path

def validate_safetensors(path, kind):
    from safetensors import safe_open
    try:
        with safe_open(str(path), framework='numpy') as f:
            shapes = {k: list(f.get_slice(k).get_shape()) for k in f.keys()}
            metadata = f.metadata() or {}
    except Exception as exc:
        raise ValueError('正常な safetensors ファイルではありません') from exc
    if not shapes:
        raise ValueError('重みが空です')
    if any(k.startswith(('conditioner.', 'double_blocks.', 'single_blocks.', 'lora_te2_')) for k in shapes) or any(x in str(metadata).lower() for x in ['sdxl', 'flux', 'stable-diffusion-xl']):
        raise ValueError('SDXL / FLUX は未対応です。SD1.5 を指定してください')
    if kind == 'model':
        schema = json.loads(Path(__file__).with_name('sd15_checkpoint_schema.json').read_text())
        missing = [k for k, v in schema.items() if shapes.get(k) != v]
        if missing:
            raise ValueError('完全な SD1.5 checkpoint が必要です（UNet/VAE/CLIP の不足または形状不一致: ' + missing[0] + '）')
        for key in shapes:
            if key.startswith(('model.diffusion_model.', 'first_stage_model.', 'cond_stage_model.')) and key not in schema and not key.endswith('position_ids'):
                raise ValueError('非対応のモデル構造です: ' + key)
    else:
        from m5diffusion.engine.lora import inspect_lora
        inspect_lora(path)
    return metadata

class Library:
    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        self.lock = threading.RLock()
        self._fingerprints = {}
        self._entries = {}
        self.checkpoints = self.root / 'models/checkpoints'
        self.loras = self.root / 'models/loras'
        self.converted = self.root / 'models/converted'
        for path in (self.checkpoints, self.loras, self.converted):
            if any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('Library directory must not be a symlink')
            path.mkdir(parents=True, exist_ok=True)

    def _entry(self, path, kind, digest=None, name=None):
        stat = path.stat()
        if not 0 < stat.st_size <= LIMITS[kind]:raise ValueError('ファイルサイズ上限を超えています')
        fingerprint = (str(path), stat.st_size, stat.st_mtime_ns)
        if fingerprint not in self._fingerprints:
            validate_safetensors(path, kind)
            self._fingerprints[fingerprint] = digest or file_hash(path)
        digest = self._fingerprints[fingerprint]
        ident = kind + '-' + digest
        sidecar = path.with_suffix('.json')
        display = name or path.stem
        if not name and sidecar.is_file() and not sidecar.is_symlink():
            try:display = json.loads(sidecar.read_text()).get('name', display)
            except (ValueError, OSError):pass
        entry = {'id': ident, 'name': display, 'kind': kind, 'format': 'safetensors', 'size': stat.st_size, 'sha256': digest}
        self._entries[ident] = (path, entry)
        return entry

    def catalog(self):
        with self.lock:
            self._entries = {}; models = []; loras = []; errors = []
            base = self.root / 'models/sd15'
            if (base / 'model_index.json').is_file() and not base.is_symlink():
                entry = {'id': 'sd15', 'name': 'Stable Diffusion 1.5 (標準)', 'kind': 'model', 'format': 'diffusers'}
                self._entries['sd15'] = (base, entry); models.append(entry)
            for directory, kind, entries in [(self.checkpoints,'model',models), (self.loras,'lora',loras)]:
                for path in sorted(directory.glob('*.safetensors')):
                    try:
                        safe_path(path); entry = self._entry(path, kind)
                        if entry['id'] not in {e['id'] for e in entries}:entries.append(entry)
                    except (ValueError, OSError) as exc:errors.append({'name':path.name,'kind':kind,'error':str(exc)})
            # Converted cache entries are advertised even when source was moved away.
            for path in sorted(self.converted.glob('model-*')):
                try:
                    if path.is_symlink() or not (path/'library.json').is_file():continue
                    entry=json.loads((path/'library.json').read_text());ident=entry['id']
                    if re.fullmatch(r'model-[0-9a-f]{64}',ident) and ident not in self._entries and (path/'model_index.json').is_file():
                        self._entries[ident]=(path,entry);models.append(entry)
                except (OSError,ValueError,KeyError):continue
            return {'models':models,'loras':loras,'errors':errors}

    def _resolve(self, ident, kind):
        if not isinstance(ident,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',ident):raise ValueError('不正なライブラリ ID')
        self.catalog()
        if ident not in self._entries or self._entries[ident][1]['kind'] != kind:raise ValueError('ライブラリに見つかりません')
        return self._entries[ident]

    def resolve_lora(self, ident):
        with self.lock:return self._resolve(ident,'lora')[0]

    def resolve_model(self, ident):
        with self.lock:
            source,entry=self._resolve(ident,'model')
            if source.is_dir():return source
            destination=self.converted/ident
            if destination.is_symlink():raise ValueError('変換先のシンボリックリンクは使えません')
            if (destination/'model_index.json').is_file():return destination
            if shutil.disk_usage(self.converted).free < 6*2**30:raise ValueError('変換には少なくとも6 GiBの空き容量が必要です')
            temporary=Path(tempfile.mkdtemp(prefix='.convert-',dir=self.converted))
            try:
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':str(self.root/'cache/huggingface'),'TOKENIZERS_PARALLELISM':'false'}
                result=subprocess.run([sys.executable,'-m','m5diffusion.models.library','--convert',str(source),str(self.root/'models/sd15-config' if (self.root/'models/sd15-config').is_dir() else ROOT/'assets/sd15-config'),str(temporary)],cwd=self.root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
                if result.returncode:raise ValueError('ローカルモデル変換に失敗しました: '+result.stdout[-2000:])
                (temporary/'library.json').write_text(json.dumps(entry,ensure_ascii=False))
                temporary.rename(destination)
            finally:
                if temporary.exists():shutil.rmtree(temporary)
            return destination

    def ingest(self, file_stream, length, kind, name):
        if kind not in LIMITS:raise ValueError('kind must be model or lora')
        if type(length) is not int or not 0 < length <= LIMITS[kind]:raise ValueError('ファイルサイズ上限: モデル8 GiB / LoRA2 GiB')
        if not isinstance(name,str) or len(name)>200 or Path(name).name!=name or '/' in name or '\\' in name or not name.lower().endswith('.safetensors'):raise ValueError('単純な .safetensors ファイル名が必要です。.ckpt / pickle は使えません')
        directory=self.checkpoints if kind=='model' else self.loras
        if shutil.disk_usage(directory).free < length+256*2**20:raise ValueError('空き容量が不足しています')
        fd,tmp=tempfile.mkstemp(prefix='.upload-',suffix='.safetensors',dir=directory);temporary=Path(tmp)
        try:
            h=hashlib.sha256();remaining=length
            with os.fdopen(fd,'wb') as dest:
                while remaining:
                    chunk=file_stream.read(min(CHUNK,remaining))
                    if not chunk:raise ValueError('ファイル受信が途中で終了しました')
                    if len(chunk)>remaining:raise ValueError('宣言サイズを超えるファイルです')
                    dest.write(chunk);h.update(chunk);remaining-=len(chunk)
                dest.flush();os.fsync(dest.fileno())
            validate_safetensors(temporary,kind);digest=h.hexdigest()
            if kind=='model' and digest==SOURCE_HASH and (self.root/'models/sd15/model_index.json').is_file():
                return next(e for e in self.catalog()['models'] if e['id']=='sd15')
            with self.lock:
                destination=directory/(digest+'.safetensors')
                if destination.exists():
                    safe_path(destination);temporary.unlink()
                else:temporary.rename(destination)
                entry=self._entry(destination,kind,digest,name)
                if destination.with_suffix('.json').is_symlink():raise ValueError('不正なライブラリ metadata')
                destination.with_suffix('.json').write_text(json.dumps({'name':name},ensure_ascii=False))
                return entry
        finally:
            temporary.unlink(missing_ok=True)

    def import_local(self, path, kind):
        if kind not in LIMITS:raise ValueError('kind must be model or lora')
        raw=Path(path).expanduser()
        if not raw.is_absolute() or '..' in raw.parts:raise ValueError('絶対パスを指定してください')
        source=safe_path(raw)
        if not 0 < source.stat().st_size <= LIMITS[kind]:raise ValueError('ファイルサイズ上限を超えています')
        if source.suffix.lower()!='.safetensors':raise ValueError('.safetensors のみ対応しています')
        # Existing library files need no copy. Hash and validate before deduplication.
        validate_safetensors(source,kind);digest=file_hash(source)
        if kind=='model' and digest==SOURCE_HASH and (self.root/'models/sd15/model_index.json').is_file():
            return next(e for e in self.catalog()['models'] if e['id']=='sd15')
        existing=(self.checkpoints if kind=='model' else self.loras)/(digest+'.safetensors')
        if existing.is_file():
            safe_path(existing)
            return self._entry(existing,kind,digest,source.name)
        with source.open('rb') as stream:return self.ingest(stream,source.stat().st_size,kind,source.name)

def convert(source, config, destination):
    import torch
    from diffusers import StableDiffusionPipeline
    validate_safetensors(Path(source),'model')
    pipeline=StableDiffusionPipeline.from_single_file(source,config=config,local_files_only=True,safety_checker=None,torch_dtype=torch.float32)
    pipeline.save_pretrained(destination,safe_serialization=True)

if __name__=='__main__':
    if len(sys.argv)!=5 or sys.argv[1]!='--convert':raise SystemExit('Internal conversion command only')
    convert(*sys.argv[2:])
