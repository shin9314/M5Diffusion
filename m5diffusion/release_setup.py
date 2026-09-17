"""Release preflight and explicit first-use model installation; no inference edits."""
import hashlib
import os
import platform
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

UPSCALER_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
UPSCALER_SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"


def preflight(root, *, machine=None, version=None, memory=None, free=None, check_mlx=True):
    root = Path(root)
    machine = machine or platform.machine()
    version = version or platform.mac_ver()[0]
    if machine != 'arm64':
        raise RuntimeError('Apple Silicon (Mシリーズ) のMacが必要です。Intel Macには対応していません。')
    if tuple(int(x) for x in version.split('.')[:2]) < (26, 0):
        raise RuntimeError('macOS 26以降が必要です。macOSを更新してください。')
    if memory is None:
        memory = int(subprocess.check_output(['/usr/sbin/sysctl', '-n', 'hw.memsize']))
    if memory < 16 * 2**30:
        raise RuntimeError('このアルファ版には16GB以上のメモリが必要です。24GBのM5で検証しています。')
    root.mkdir(parents=True, exist_ok=True)
    if free is None:
        free = shutil.disk_usage(root).free
    if free < 8 * 2**30:
        raise RuntimeError('空き容量が不足しています。8GB以上の空きを確保してください。モデル追加には別途容量が必要です。')
    try:
        with tempfile.TemporaryFile(dir=root):
            pass
    except OSError as exc:
        raise RuntimeError('アプリの保存先に書き込めません。ユーザーフォルダのアクセス権と空き容量を確認してください。') from exc
    if check_mlx:
        try:
            import mlx.core as mx
            if not mx.metal.is_available():
                raise RuntimeError('Metal unavailable')
            value = mx.array([1.0]) + 1
            mx.eval(value)
        except Exception as exc:
            raise RuntimeError('画像処理の準備ができませんでした。Macを再起動してください。改善しない場合はログを添えて報告してください。') from exc
    return {'memory_gib': memory / 2**30, 'free_gib': free / 2**30,
            'model_present': (root / 'models/sd15/model_index.json').is_file() or any((root / 'models/checkpoints').glob('*.safetensors')),
            'validated_hardware': 'M5 Air 24GB only'}


def install_upscale_model(root):
    destination = Path(root) / 'models/upscale/realesr-general-x4v3.pth'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == UPSCALER_SHA256:
        return {'ready': True}
    fd, temporary = tempfile.mkstemp(prefix='.download-', dir=destination.parent)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(UPSCALER_URL, timeout=30) as response:
            total = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > 8 * 1024 * 1024:
                    raise ValueError('Unexpected model download size')
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != UPSCALER_SHA256:
            raise ValueError('Model integrity verification failed')
        os.replace(temporary, destination)
        return {'ready': True}
    except Exception as exc:
        raise RuntimeError('高画質化モデルを準備できませんでした。インターネット接続と空き容量を確認し、もう一度お試しください。') from exc
    finally:
        Path(temporary).unlink(missing_ok=True)
