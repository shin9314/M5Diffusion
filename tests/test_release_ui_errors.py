import errno
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
import numpy as np
import pytest
from m5diffusion.ui import server
from m5diffusion.models.library import Library

@pytest.mark.parametrize('error,expected',[(MemoryError('alloc'),'メモリ'),(RuntimeError('Metal out of memory'),'メモリ'),(PermissionError('/Users/private/secret'),'アクセス権'),(OSError(errno.ENOSPC,'disk'),'空き容量'),(OSError(errno.EADDRINUSE,'bind'),'7861'),(RuntimeError('mlx stream failed'),'画像処理'),(ValueError('ローカルモデル変換に失敗しました: Traceback\nsecret'),'モデル'),(ValueError('bad safetensors header'),'モデル')])
def test_friendly_error(error,expected):
    text=server.user_error(error)
    assert expected in text
    assert 'Traceback' not in text and '/Users/' not in text and 'secret' not in text


def test_worker_survives_oom(tmp_path):
    class Engine:
        load_time=0
        calls=0
        def generate(self,request):
            self.calls+=1
            if self.calls==1:raise MemoryError('simulated out of memory')
            return np.arange(12,dtype=np.float32).reshape(2,2,3)/12,{'total_time':0}
    jobs=server.Jobs(factory=Engine,output=tmp_path)
    first=jobs.submit(server.Request());jobs.queue.join()
    assert jobs.get(first)['status']=='error'
    second=jobs.submit(server.Request());jobs.queue.join()
    assert jobs.get(second)['status']=='done'


def test_empty_library_and_corrupt_model(tmp_path):
    library=Library(tmp_path)
    assert library.catalog()['models']==[]
    bad=library.checkpoints/'broken.safetensors';bad.write_bytes(b'not model weights')
    catalog=library.catalog()
    assert catalog['models']==[] and len(catalog['errors'])==1
    jobs=server.Jobs(library=library)
    with pytest.raises(ValueError,match='モデルを追加'):jobs.submit(server.Request())


def test_nonimage_is_friendly():
    with pytest.raises(ValueError,match='画像を読み込めません'):
        server.decode_upscale(b'not an image',2)


def test_missing_ui_and_permission_api_do_not_disconnect(tmp_path,monkeypatch):
    class Denied:
        def catalog(self):raise PermissionError('/Users/private/library')
    jobs=server.Jobs(library=Denied())
    http=ThreadingHTTPServer(('127.0.0.1',0),server.handler_for(jobs))
    thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f'http://127.0.0.1:{http.server_port}/api/library')
        assert 'アクセス権' in json.load(exc.value)['error']
        monkeypatch.setattr(server,'__file__',str(tmp_path/'absent.py'))
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f'http://127.0.0.1:{http.server_port}/')
        assert exc.value.code==500
        assert '必要なファイル' in json.load(exc.value)['error']
    finally:http.shutdown();http.server_close()


def test_port_in_use_friendly(monkeypatch,capsys):
    def occupied(*args,**kwargs):raise OSError(errno.EADDRINUSE,'Address in use')
    monkeypatch.setattr(server,'ThreadingHTTPServer',occupied)
    with pytest.raises(SystemExit):server.main()
    assert '7861' in capsys.readouterr().out


def test_upscale_setup_api_explicit_only(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    calls=[]
    monkeypatch.setitem(sys.modules,'m5diffusion.release_setup',SimpleNamespace(install_upscale_model=lambda root:calls.append(root)))
    monkeypatch.setattr(server,'upscaler_ready',lambda:False)
    http=ThreadingHTTPServer(('127.0.0.1',0),server.handler_for(server.Jobs(factory=lambda:None,output=tmp_path)))
    threading.Thread(target=http.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{http.server_port}'
    try:
        assert json.load(urllib.request.urlopen(base+'/api/health'))['upscaler_ready'] is False
        assert calls==[]
        request=urllib.request.Request(base+'/api/setup/upscale',method='POST')
        assert json.load(urllib.request.urlopen(request))=={'ready':True}
        assert calls==[server.ROOT]
    finally:http.shutdown();http.server_close()
