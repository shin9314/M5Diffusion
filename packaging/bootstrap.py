"""Private app setup and owned server lifecycle. Executed by bundled Python."""
import json, os, shutil, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path


def prepare(resources, destination):
    payload = resources / 'payload'
    destination.mkdir(parents=True, exist_ok=True)
    # Copy only application files. Never overwrite user models, outputs or cache.
    for name in ('m5diffusion', 'vendor', 'serve.py'):
        source = payload / name
        if source.is_dir():
            shutil.copytree(source, destination / name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination / name)
    shutil.copytree(payload / 'models/sd15-config', destination / 'models/sd15-config', dirs_exist_ok=True)
    sys.path.insert(0, str(destination))
    from m5diffusion.release_setup import preflight
    return preflight(destination)


def main():
    resources = Path(__file__).resolve().parent
    destination = Path(os.environ.get('M5DIFFUSION_DATA_DIR', str(Path.home() / 'Library/Application Support/M5Diffusion/current')))
    child = None
    try:
        print('STATUS:初回チェックを実行しています…', flush=True)
        info = prepare(resources, destination)
        (destination/'work').mkdir(exist_ok=True)
        (destination/'work/preflight.json').write_text(json.dumps(info, indent=2))
        if '--check-only' in sys.argv:
            print('READY:check-only', flush=True)
            return 0
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', 7861))
            except OSError:
                raise RuntimeError('ポート7861は別のアプリで使用中です。起動済みのM5Diffusionを終了して、もう一度開いてください。')
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONNOUSERSITE':'1', 'TOKENIZERS_PARALLELISM':'false', 'HF_HOME':str(destination/'cache/huggingface')}
        with (destination/'work/server.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-B', str(destination/'serve.py')], cwd=destination, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        def stop(*_):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError('画面を起動できませんでした。アプリを開き直してください。詳細はアプリのログで確認できます。')
            try:
                with urllib.request.urlopen('http://127.0.0.1:7861/api/health', timeout=.4) as response:
                    if json.load(response).get('app') == 'M5Diffusion':
                        print('READY:http://127.0.0.1:7861/', flush=True)
                        return child.wait()
            except (OSError, ValueError):
                time.sleep(.1)
        raise RuntimeError('画面の起動が時間内に終わりませんでした。アプリを開き直してください。')
    except Exception as exc:
        if isinstance(exc, PermissionError):
            message = '保存先に書き込めません。ユーザーフォルダのアクセス権を確認してください。'
        elif isinstance(exc, RuntimeError):
            message = str(exc)
        else:
            message = 'アプリの準備に失敗しました。空き容量を確認し、アプリを入れ直してください。'
        print('ERROR:' + message, flush=True)
        return 1
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL)

if __name__ == '__main__':
    raise SystemExit(main())
