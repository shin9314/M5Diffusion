"""Release startup/download failures without real downloads, models or GPU work."""
import hashlib
import importlib.util
import io
import sys
import types
from pathlib import Path

import pytest
from m5diffusion import release_setup as release


def preflight(tmp_path,**overrides):
    options=dict(machine='arm64',version='26.0',memory=24*2**30,free=20*2**30,check_mlx=False)
    options.update(overrides)
    return release.preflight(tmp_path,**options)


@pytest.mark.parametrize('options,text',[
    ({'machine':'x86_64'},'Apple Silicon'),
    ({'version':'25.0'},'macOS 26'),
    ({'memory':8*2**30},'16GB'),
    ({'free':2*2**30},'8GB'),
])
def test_preflight_explains_unsupported_environment(tmp_path,options,text):
    with pytest.raises(RuntimeError,match=text):preflight(tmp_path,**options)


def test_missing_model_is_nonfatal(tmp_path):
    result=preflight(tmp_path)
    assert result['model_present'] is False and result['memory_gib']==24
    (tmp_path/'models/sd15').mkdir(parents=True)
    (tmp_path/'models/sd15/model_index.json').write_text('{}')
    assert preflight(tmp_path)['model_present'] is True


def test_preflight_write_denied_has_friendly_error(tmp_path,monkeypatch):
    def denied(*args,**kwargs):raise PermissionError('technical permission details')
    monkeypatch.setattr(release.tempfile,'TemporaryFile',denied)
    with pytest.raises(RuntimeError,match='書き込めません') as exc:preflight(tmp_path)
    assert 'technical permission details' not in str(exc.value)


def test_mlx_initialization_failure_is_friendly_and_gpu_free(tmp_path,monkeypatch):
    mlx=types.ModuleType('mlx');core=types.ModuleType('mlx.core')
    core.metal=types.SimpleNamespace(is_available=lambda:False);mlx.core=core
    monkeypatch.setitem(sys.modules,'mlx',mlx);monkeypatch.setitem(sys.modules,'mlx.core',core)
    with pytest.raises(RuntimeError,match='画像処理の準備'):preflight(tmp_path,check_mlx=True)


@pytest.mark.parametrize('body',[b'corrupted',b'x'*(8*1024*1024+1)])
def test_download_failure_cleans_temporary_and_preserves_existing(tmp_path,monkeypatch,body):
    destination=tmp_path/'models/upscale/realesr-general-x4v3.pth'
    destination.parent.mkdir(parents=True);destination.write_bytes(b'previous file')
    monkeypatch.setattr(release.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(body))
    with pytest.raises(RuntimeError,match='準備できませんでした'):release.install_upscale_model(tmp_path)
    assert destination.read_bytes()==b'previous file'
    assert not list(destination.parent.glob('.download-*'))


def test_verified_download_is_atomic_and_cached(tmp_path,monkeypatch):
    body=b'valid test model';calls=[]
    monkeypatch.setattr(release,'UPSCALER_SHA256',hashlib.sha256(body).hexdigest())
    def response(*a,**k):calls.append(1);return io.BytesIO(body)
    monkeypatch.setattr(release.urllib.request,'urlopen',response)
    assert release.install_upscale_model(tmp_path)=={'ready':True}
    assert release.install_upscale_model(tmp_path)=={'ready':True}
    destination=tmp_path/'models/upscale/realesr-general-x4v3.pth'
    assert destination.read_bytes()==body and len(calls)==1
    assert not list(destination.parent.glob('.download-*'))


@pytest.fixture
def bootstrap():
    path=Path(__file__).resolve().parents[1]/'packaging/bootstrap.py'
    spec=importlib.util.spec_from_file_location('release_bootstrap_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_prepare_preserves_user_models_outputs_and_reports_missing_base(tmp_path,monkeypatch,bootstrap):
    resources=tmp_path/'resources';payload=resources/'payload';destination=tmp_path/'user-data'
    for folder in ['m5diffusion','vendor','models/sd15-config']:(payload/folder).mkdir(parents=True)
    (payload/'serve.py').write_text('# fixture')
    (payload/'models/sd15-config/model_index.json').write_text('{}')
    for file in ('models/custom.safetensors','outputs/user.png','cache/private'):
        p=destination/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'keep')
    monkeypatch.setattr(sys,'path',sys.path.copy())
    monkeypatch.setattr(release,'preflight',lambda root:{'model_present':False})
    assert bootstrap.prepare(resources,destination)['model_present'] is False
    for file in ('models/custom.safetensors','outputs/user.png','cache/private'):assert (destination/file).read_bytes()==b'keep'
    assert (destination/'serve.py').exists()


def setup_main(tmp_path,monkeypatch,bootstrap):
    destination=tmp_path/'data';destination.mkdir()
    monkeypatch.setenv('M5DIFFUSION_DATA_DIR',str(destination))
    monkeypatch.setattr(sys,'argv',['bootstrap.py'])
    monkeypatch.setattr(bootstrap,'prepare',lambda *a:{'model_present':False})
    return destination


def test_busy_port_friendly_and_does_not_start_or_kill_server(tmp_path,monkeypatch,bootstrap,capsys):
    setup_main(tmp_path,monkeypatch,bootstrap)
    class Socket:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def bind(self,*a):raise OSError('busy')
    monkeypatch.setattr(bootstrap.socket,'socket',Socket)
    def forbidden(*a,**k):raise AssertionError('Must not touch existing server')
    monkeypatch.setattr(bootstrap.subprocess,'Popen',forbidden)
    monkeypatch.setattr(bootstrap.os,'killpg',forbidden)
    assert bootstrap.main()==1
    assert 'ポート7861' in capsys.readouterr().out


def test_server_early_exit_has_no_traceback(tmp_path,monkeypatch,bootstrap,capsys):
    setup_main(tmp_path,monkeypatch,bootstrap)
    class Socket:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def bind(self,*a):pass
    monkeypatch.setattr(bootstrap.socket,'socket',Socket)
    monkeypatch.setattr(bootstrap.subprocess,'Popen',lambda *a,**k:types.SimpleNamespace(poll=lambda:2))
    monkeypatch.setattr(bootstrap.signal,'signal',lambda *a:None)
    assert bootstrap.main()==1
    text=capsys.readouterr().out
    assert '画面を起動できませんでした' in text and 'Traceback' not in text


def test_check_only_missing_model_does_not_launch_server(tmp_path,monkeypatch,bootstrap,capsys):
    destination=setup_main(tmp_path,monkeypatch,bootstrap)
    monkeypatch.setattr(sys,'argv',['bootstrap.py','--check-only'])
    assert bootstrap.main()==0
    assert 'READY:check-only' in capsys.readouterr().out
    assert (destination/'work/preflight.json').exists()
