"""Loopback-only library and UI. All model mutations run on one worker."""
import dataclasses
import gc
import io
import json
import logging
import errno
import math
import os
import queue
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from m5diffusion.engine.common import ROOT, Request


LOGGER = logging.getLogger('m5diffusion.ui')


def user_error(exc):
    """Keep paths, tracebacks and runtime internals in logs, not normal UI."""
    raw = str(exc)
    lower = raw.lower()
    if isinstance(exc, MemoryError) or any(s in lower for s in ('out of memory', 'memory limit', 'allocation failed', 'insufficient memory')):
        return 'メモリが不足しています。他のアプリを閉じ、M5Diffusion を起動し直してから再試行してください。'
    if isinstance(exc, PermissionError) or 'permission denied' in lower or 'read-only file system' in lower:
        return 'ファイルを保存・読み込みできません。アプリのデータフォルダと選択したファイルのアクセス権を確認してください。'
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return '空き容量が不足しています。Mac のストレージを空けてから再試行してください。'
    if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE:
        return '起動用ポート 7861 が使用中です。すでに開いている M5Diffusion を確認するか、使用中のアプリを終了してください。'
    if isinstance(exc, FileNotFoundError):
        return '必要なファイルが見つかりません。モデルを追加し直してください。改善しない場合はアプリを再インストールしてください。'
    if 'AI拡大モデル' in raw:
        return '高画質化モデルが未導入、または破損しています。「高画質化モデルを準備」から再取得してください。'
    if '変換に失敗' in raw or 'safetensor' in lower or 'checkpoint' in lower or 'state_dict' in lower:
        return 'モデルを読み込めません。破損していない完全な SD1.5 の .safetensors ファイルを追加し直してください。'
    if isinstance(exc, (ImportError, ModuleNotFoundError)) or any(s in lower for s in ('metal', 'mlx', 'command buffer', 'gpu')):
        return '画像処理機能を利用できません。Apple Silicon 対応の Mac と対応する macOS を確認し、アプリを再起動してください。改善しない場合は再インストールしてください。'
    # Validated Japanese guidance may pass through; never expose multiline diagnostics or paths.
    if isinstance(exc, ValueError) and re.search(r'[ぁ-んァ-ン一-龯]', raw) and not any(s in raw for s in ('Traceback', '\n', '/Users/', '/Library/', '/private/')):
        return raw[:350]
    if isinstance(exc, (ValueError, TypeError, UnicodeDecodeError)):
        return '入力内容を確認してください。サイズ・ステップ数・ファイル形式が対応範囲内か確認して再試行してください。'
    return '処理を完了できませんでした。アプリを再起動して再試行してください。詳しい原因はアプリのログに記録しています。'


def parse_selection(data):
    model = data.get('model', 'sd15')
    if not isinstance(model, str) or not re.fullmatch(r'(sd15|model-[a-f0-9]{64})', model):
        raise ValueError('対応するSD1.5モデルを選択してください')
    if data.get('lora'):
        raise ValueError('LoRAは選択欄から追加してください')
    loras = data.get('loras', [])
    if not isinstance(loras, list) or len(loras) > 4:
        raise ValueError('LoRAは同時に4個まで選択できます')
    result = []
    seen = set()
    for item in loras:
        if not isinstance(item, dict) or set(item) != {'id', 'scale'}:
            raise ValueError('LoRAの指定形式が正しくありません')
        identity, scale = item['id'], item['scale']
        if not isinstance(identity, str) or not re.fullmatch(r'lora-[a-f0-9]{64}', identity) or identity in seen:
            raise ValueError('LoRAの指定が不正、または重複しています')
        if type(scale) not in (int, float) or not math.isfinite(scale) or not -2 <= scale <= 2:
            raise ValueError('LoRAの強度は-2〜2で指定してください')
        seen.add(identity)
        result.append({'id': identity, 'scale': float(scale)})
    return model, result


def parse_request(data):
    if not isinstance(data, dict):
        raise ValueError('JSON object required')
    parse_selection(data)
    allowed = {f.name for f in dataclasses.fields(Request)} | {'model', 'lora', 'loras'}
    if set(data) - allowed:
        raise ValueError('Unknown request fields')
    args = {k: v for k, v in data.items() if k not in {'model', 'lora', 'loras'}}
    for name in ('seed', 'steps', 'width', 'height'):
        if name in args and type(args[name]) is not int:
            raise ValueError(f'{name} must be an integer')
    if 'cfg' in args and (type(args['cfg']) not in (int, float) or not math.isfinite(args['cfg'])):
        raise ValueError('CFG must be finite')
    for name in ('prompt', 'negative_prompt'):
        if name in args and (not isinstance(args[name], str) or len(args[name]) > 10000):
            raise ValueError(f'{name} must be text up to 10000 characters')
    r = Request(**args)
    r.validate()
    if not 0 <= r.seed <= 2**32 - 1:
        raise ValueError('Seed must be 0..4294967295')
    return r


def validate_upscale(image, scale):
    if type(scale) is not int or scale not in (2, 4):
        raise ValueError('拡大率は2倍または4倍を選んでください')
    w, h = image.size
    if w * h > 4_000_000 or w * h * scale * scale > 16_000_000 or max(w, h) * scale > 4096:
        raise ValueError('画像が大きすぎます。出力は最大4096px・1600万画素、入力は最大400万画素です。小さな画像または2倍を選んでください')


def decode_upscale(payload, scale):
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ValueError('PNG・JPEG・WebP画像を選んでください')
            if getattr(source, 'is_animated', False):
                raise ValueError('静止画像を選んでください')
            validate_upscale(source, scale)
            source.load()
            return ImageOps.exif_transpose(source).convert('RGBA' if 'A' in source.getbands() or 'transparency' in source.info else 'RGB')
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, OSError) as exc:
        raise ValueError('画像を読み込めません。PNG・JPEG・WebP画像を選び直してください') from exc


class Jobs:
    def __init__(self, factory=None, output=None, library=None):
        self.output = Path(output or ROOT / 'outputs')
        self.factory = factory
        self._library = library
        self.jobs = {}
        self.lock = threading.Lock()
        self.queue = queue.Queue(maxsize=3)
        self.worker = threading.Thread(target=self.run, daemon=True, name='mlx-generation')
        self.worker.start()

    @property
    def library(self):
        if self._library is None:
            from m5diffusion.models.library import Library
            with self.lock:
                if self._library is None:
                    self._library = Library(ROOT)
        return self._library

    @staticmethod
    def make_engine(model):
        from m5diffusion.engine.mlx_backend import MLXEngine
        return MLXEngine(model, compiled=os.environ.get('M5DIFFUSION_COMPILE', '1') == '1',
                         buffer_reuse=os.environ.get('M5_BUFFER_REUSE', '1') == '1',
                         quantize=int(os.environ.get('M5_QUANTIZE', '0')),
                         attention=os.environ.get('M5DIFFUSION_ATTENTION', 'padded'))

    def submit(self, request, model='sd15', loras=None):
        model, loras = parse_selection({'model': model, 'loras': loras or []})
        # Fake factories in CPU tests need no real library or model files.
        if self.factory is None or self._library is not None:
            catalog = self.library.catalog()
            if model not in {m['id'] for m in catalog['models']}:
                raise ValueError('モデルがありません。「モデルを追加」から SD1.5 のファイルを選ぶか、Mac 内のファイルのパスを指定してください。')
            if any(x['id'] not in {m['id'] for m in catalog['loras']} for x in loras):
                raise ValueError('LoRAが見つかりません。一覧を更新してください')
        job_id = uuid.uuid4().hex
        with self.lock:
            if self.queue.full():
                raise queue.Full
            if len(self.jobs) >= 100:
                for key, old in list(self.jobs.items()):
                    if old['status'] in ('done', 'error'):
                        del self.jobs[key]
                        break
            self.jobs[job_id] = {'id': job_id, 'status': 'queued', 'created': time.time()}
            self.queue.put_nowait((job_id, request, model, loras))
        return job_id

    def submit_upscale(self, image, scale):
        validate_upscale(image, scale)
        job_id = uuid.uuid4().hex
        with self.lock:
            if self.queue.full():
                raise queue.Full
            if len(self.jobs) >= 100:
                for key, old in list(self.jobs.items()):
                    if old['status'] in ('done', 'error'):
                        del self.jobs[key]
                        break
            self.jobs[job_id] = {'id': job_id, 'kind': 'upscale', 'status': 'queued', 'created': time.time(), 'progress': 0}
            self.queue.put_nowait((job_id, (image.copy(), scale), '__upscale__', []))
        return job_id

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def run(self):
        engine = None
        current_model = None
        current_loras = None
        while True:
            job_id, request, model, loras = self.queue.get()
            try:
                if model == '__upscale__':
                    if engine is not None:
                        del engine
                        engine = None
                        gc.collect()
                        if self.factory is None:
                            import mlx.core as mx
                            mx.clear_cache()
                    current_model = current_loras = None
                    from m5diffusion.engine.upscale import upscale_image
                    source, scale = request
                    self.update(job_id, status='running', started=time.time())
                    started = time.perf_counter()
                    image, metadata = upscale_image(source, scale, progress=lambda fraction, message: self.update(job_id, progress=round(fraction * 100), message=message))
                    self.output.mkdir(parents=True, exist_ok=True)
                    image.save(self.output / f'{job_id}.png')
                    result = {'kind': 'upscale', 'model': 'Real-ESRGAN general x4v3', 'scale': scale,
                              'input_size': list(source.size), 'output_size': list(image.size),
                              'timing': {'total_time': time.perf_counter() - started}, 'details': metadata,
                              'image': f'/outputs/{job_id}.png'}
                    (self.output / f'{job_id}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
                    self.update(job_id, status='done', progress=100, **result)
                    continue
                if engine is None or current_model != model:
                    self.update(job_id, status='converting')
                    # Release previous weights before conversion/model loading to cap memory.
                    if engine is not None:
                        del engine
                        engine = None
                        gc.collect()
                        if self.factory is None:
                            import mlx.core as mx
                            mx.clear_cache()
                    current_model = current_loras = None
                    model_path = self.library.resolve_model(model) if self.factory is None else None
                    self.update(job_id, status='loading')
                    engine = self.make_engine(model_path) if self.factory is None else self.factory()
                    current_model = model
                if current_loras != loras:
                    self.update(job_id, status='adapting')
                    if hasattr(engine, 'set_loras'):
                        resolved = [(self.library.resolve_lora(x['id']), x['scale']) for x in loras]
                        engine.set_loras(resolved)
                    elif loras:
                        raise ValueError('このエンジンではLoRAを使用できません')
                    current_loras = list(loras)
                self.update(job_id, status='running', started=time.time())
                image, timing = engine.generate(request)
                import numpy as np
                from PIL import Image
                if not np.isfinite(image).all() or image.std() < .01:
                    raise RuntimeError('Generated image is invalid')
                self.output.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(self.output / f'{job_id}.png')
                result = {'request': dataclasses.asdict(request), 'model': model, 'loras': loras, 'backend': 'mlx',
                          'sampler': 'DPM++ 2M / Karras', 'timing': timing,
                          'model_load_time': engine.load_time, 'image': f'/outputs/{job_id}.png'}
                (self.output / f'{job_id}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
                self.update(job_id, status='done', **result)
            except Exception as exc:
                LOGGER.exception('Job failed: %s', job_id)
                self.update(job_id, status='error', error=user_error(exc))
            finally:
                self.queue.task_done()


def upscaler_ready():
    import hashlib
    from m5diffusion.engine.upscale import MODEL_SHA256
    path = ROOT / 'models/upscale/realesr-general-x4v3.pth'
    try:
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256
    except OSError:
        return False


def handler_for(jobs):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, kind='application/json; charset=utf-8'):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def trusted(self):
            port = self.server.server_port
            hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
            return self.headers.get('Host') in hosts and (not self.headers.get('Origin') or self.headers['Origin'] in {f'http://{h}' for h in hosts})

        def do_GET(self):
            try:
                self.get_response()
            except Exception as exc:
                LOGGER.exception('Could not serve UI request')
                self.send(500, {'error': user_error(exc)})

        def get_response(self):
            if not self.trusted():
                return self.send(403, {'error': 'Local origin required'})
            path = urlparse(self.path).path
            if path == '/':
                return self.send(200, Path(__file__).with_name('index.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/health':
                return self.send(200, {'app': 'M5Diffusion', 'backend': 'mlx', 'model': 'sd15', 'lora': True, 'upscaler_ready': upscaler_ready()})
            if path == '/api/library':
                try:
                    catalog = jobs.library.catalog()
                    for item in catalog.get('errors', []):
                        item['error'] = user_error(ValueError(item['error']))
                    catalog['needs_model'] = not bool(catalog['models'])
                    return self.send(200, catalog)
                except (ValueError, OSError) as exc:
                    return self.send(400, {'error': user_error(exc)})
            if path.startswith('/api/jobs/'):
                result = jobs.get(path.rsplit('/', 1)[-1])
                return self.send(200 if result else 404, result or {'error': 'Job not found'})
            if path.startswith('/outputs/'):
                name = path.rsplit('/', 1)[-1]
                stem, _, ext = name.partition('.')
                if len(stem) == 32 and all(c in '0123456789abcdef' for c in stem) and ext in ('png', 'json'):
                    file = jobs.output / name
                    if file.is_file():
                        return self.send(200, file.read_bytes(), 'image/png' if ext == 'png' else 'application/json')
            self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.trusted():
                return self.send(403, {'error': 'Local origin required'})
            parsed = urlparse(self.path)
            if parsed.path not in ('/api/jobs', '/api/upscale', '/api/setup/upscale', '/api/library/upload', '/api/library/import'):
                return self.send(404, {'error': 'Not found'})
            try:
                if self.headers.get('Transfer-Encoding'):
                    raise ValueError('Content-Length is required')
                if parsed.path == '/api/setup/upscale':
                    from m5diffusion.release_setup import install_upscale_model
                    install_upscale_model(root=ROOT)
                    return self.send(200, {'ready': True})
                length = int(self.headers.get('Content-Length', '0'))
                if parsed.path == '/api/upscale':
                    scale_values = parse_qs(parsed.query).get('scale', [])
                    if len(scale_values) != 1 or scale_values[0] not in ('2', '4'):
                        raise ValueError('拡大率は2倍または4倍を選んでください')
                    if not 0 < length <= 20 * 1024 * 1024:
                        raise ValueError('画像ファイルは20MB以下にしてください')
                    self.connection.settimeout(60)
                    payload = self.rfile.read(length)
                    if len(payload) != length:
                        raise ValueError('画像の転送が完了しませんでした。もう一度選んでください')
                    scale = int(scale_values[0])
                    image = decode_upscale(payload, scale)
                    return self.send(202, {'id': jobs.submit_upscale(image, scale)})
                if parsed.path == '/api/library/upload':
                    query = parse_qs(parsed.query)
                    kind, name = query.get('kind', [''])[0], query.get('name', [''])[0]
                    self.connection.settimeout(120)
                    return self.send(201, jobs.library.ingest(self.rfile, length, kind, name))
                if not 0 < length <= 65536:
                    raise ValueError('Request size must be 1..65536 bytes')
                data = json.loads(self.rfile.read(length))
                if parsed.path == '/api/library/import':
                    if not isinstance(data, dict) or set(data) != {'path', 'kind'} or not isinstance(data['path'], str):
                        raise ValueError('ファイルのパスと種類を指定してください')
                    return self.send(201, jobs.library.import_local(data['path'], data['kind']))
                request = parse_request(data)
                model, loras = parse_selection(data)
                job_id = jobs.submit(request, model, loras)
                self.send(202, {'id': job_id})
            except queue.Full:
                self.send(429, {'error': '処理待ちがいっぱいです。現在の画像が完了してから再試行してください。'})
            except (ValueError, TypeError, UnicodeDecodeError, OSError) as exc:
                LOGGER.warning('Request failed: %s', exc)
                self.send(400, {'error': user_error(exc)})
            except Exception as exc:
                LOGGER.exception('Unexpected request failure')
                self.send(500, {'error': user_error(exc)})
    return Handler


def main():
    try:
        server = ThreadingHTTPServer(('127.0.0.1', 7861), handler_for(Jobs()))
    except OSError as exc:
        print(user_error(exc), flush=True)
        raise SystemExit(1) from None
    print('M5Diffusion: http://127.0.0.1:7861 (model loads on first generation)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
