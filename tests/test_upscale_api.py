"""CPU-only upscale API contract: real image decoding, mocked AI inference."""
import contextlib
import io
import json
import queue
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image
import pytest
from m5diffusion.ui.server import Jobs, handler_for, parse_request


def encoded_image(format='PNG',size=(12,8)):
    stream=io.BytesIO()
    Image.new('RGB',size,(100,120,140)).save(stream,format=format)
    return stream.getvalue()


def fake_upscale(image,scale,progress=None):
    if progress:progress(.5,'CPU test fixture')
    output=image.resize((image.width*scale,image.height*scale))
    return output,{'model':'test-upscaler','scale':scale,'input_size':list(image.size),'output_size':list(output.size),'device':'cpu','precision':'float32','total_seconds':.01}


@pytest.fixture
def upscaler(monkeypatch):
    import m5diffusion.engine.upscale as module
    monkeypatch.setattr(module,'upscale_image',fake_upscale)
    return module


@contextlib.contextmanager
def running_server(jobs):
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(jobs))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:yield f'http://127.0.0.1:{server.server_port}'
    finally:server.shutdown();server.server_close()


def post(base,body,scale='2',headers=None):
    request=urllib.request.Request(base+'/api/upscale?scale='+scale,data=body,headers={'Content-Type':'image/png',**(headers or {})})
    return urllib.request.urlopen(request,timeout=5)


def forbidden_diffusion():
    raise AssertionError('Upscaling must not load a diffusion model')


@pytest.mark.parametrize('format',['PNG','JPEG','WEBP'])
def test_raw_upload_outputs_png_and_metadata_without_diffusion(tmp_path,upscaler,format):
    jobs=Jobs(factory=forbidden_diffusion,output=tmp_path)
    with running_server(jobs) as base:
        with post(base,encoded_image(format)) as response:
            assert response.status==202
            identity=json.load(response)['id']
        jobs.queue.join()
        result=jobs.get(identity)
        assert result['status']=='done',result
        with Image.open(tmp_path/f'{identity}.png') as image:
            assert image.size==(24,16) and image.format=='PNG'
        metadata=json.loads((tmp_path/f'{identity}.json').read_text())
        assert metadata['image']==f'/outputs/{identity}.png'
        assert metadata['kind']=='upscale' and metadata['scale']==2
        assert metadata['input_size']==[12,8] and metadata['output_size']==[24,16]
        assert metadata['details']['model']=='test-upscaler'
        with urllib.request.urlopen(base+metadata['image']) as response:
            assert response.headers['Content-Type']=='image/png'
            assert response.read().startswith(b'\x89PNG')
        with urllib.request.urlopen(base+'/api/jobs/'+identity) as response:
            assert json.load(response)['status']=='done'


@pytest.mark.parametrize('scale',['0','1','3','5','2.5','nan','bad'])
def test_invalid_scale_is_client_error(tmp_path,upscaler,scale):
    jobs=Jobs(factory=forbidden_diffusion,output=tmp_path)
    with running_server(jobs) as base:
        with pytest.raises(urllib.error.HTTPError) as exc:post(base,encoded_image(),scale)
        assert exc.value.code==400
    assert not list(tmp_path.glob('*.png'))


def test_corruption_size_and_origin_are_rejected_without_model(tmp_path,upscaler):
    jobs=Jobs(factory=forbidden_diffusion,output=tmp_path)
    with running_server(jobs) as base:
        for body,headers,expected in [
            (b'not an image',{},400),
            (b'',{'Content-Length':str(20*2**20+1)},400),
            (encoded_image(),{'Origin':'https://example.com'},403),
        ]:
            with pytest.raises(urllib.error.HTTPError) as exc:post(base,body,headers=headers)
            assert exc.value.code==expected
    assert jobs.jobs=={}


def test_decoded_pixel_limit_rejected_before_queue(tmp_path,upscaler):
    jobs=Jobs(factory=forbidden_diffusion,output=tmp_path)
    with running_server(jobs) as base:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(base,encoded_image(size=(2049,2048)))
        assert exc.value.code==400
    assert jobs.jobs=={}


def test_queue_full_maps_to_429(tmp_path,upscaler,monkeypatch):
    jobs=Jobs(factory=forbidden_diffusion,output=tmp_path)
    def full(*args,**kwargs):raise queue.Full
    monkeypatch.setattr(jobs,'submit_upscale',full)
    with running_server(jobs) as base:
        with pytest.raises(urllib.error.HTTPError) as exc:post(base,encoded_image())
        assert exc.value.code==429


def test_upscale_error_does_not_poison_next_job(tmp_path,upscaler,monkeypatch,caplog):
    calls=[]
    def fail_once(image,scale,progress=None):
        calls.append(threading.get_ident())
        if len(calls)==1:raise RuntimeError('Simulated model failure')
        return fake_upscale(image,scale,progress)
    monkeypatch.setattr(upscaler,'upscale_image',fail_once)
    jobs=Jobs(factory=forbidden_diffusion,output=tmp_path)
    ids=[jobs.submit_upscale(Image.new('RGB',(12,8)),2) for _ in range(2)]
    jobs.queue.join()
    assert jobs.get(ids[0])['status']=='error'
    assert jobs.get(ids[0])['error']
    assert 'Simulated model failure' not in jobs.get(ids[0])['error']
    assert 'Traceback' not in jobs.get(ids[0])['error']
    assert 'Simulated model failure' in caplog.text
    assert jobs.get(ids[1])['status']=='done'
    assert len(set(calls))==1


def test_upscale_and_diffusion_share_serial_worker_and_backpressure(tmp_path,upscaler,monkeypatch):
    events=[];entered=threading.Event();release=threading.Event()
    def blocking_upscale(image,scale,progress=None):
        events.append(('upscale',threading.get_ident()))
        if len(events)==1:
            entered.set()
            if not release.wait(5):raise RuntimeError('test synchronization timed out')
        return fake_upscale(image,scale,progress)
    class Engine:
        load_time=.01
        def __init__(self):events.append(('load',threading.get_ident()))
        def generate(self,request):
            events.append(('generate',threading.get_ident()))
            return np.arange(12,dtype=np.float32).reshape(2,2,3)/12,{'total_time':.01}
    monkeypatch.setattr(upscaler,'upscale_image',blocking_upscale)
    jobs=Jobs(factory=Engine,output=tmp_path)
    first=jobs.submit_upscale(Image.new('RGB',(12,8)),2)
    try:
        assert entered.wait(5)
        second=jobs.submit(parse_request({'steps':2}))
        third=jobs.submit_upscale(Image.new('RGB',(12,8)),4)
        fourth=jobs.submit(parse_request({'steps':2}))
        with pytest.raises(queue.Full):jobs.submit_upscale(Image.new('RGB',(12,8)),2)
        assert [x[0] for x in events]==['upscale']
    finally:release.set()
    jobs.queue.join()
    assert [x[0] for x in events]==['upscale','load','generate','upscale','load','generate']
    assert len({x[1] for x in events})==1
    assert all(jobs.get(i)['status']=='done' for i in (first,second,third,fourth))


def test_decode_preserves_alpha_and_applies_exif_orientation():
    from m5diffusion.ui.server import decode_upscale
    rgba=Image.new('RGBA',(12,8),(100,120,140,75))
    stream=io.BytesIO();rgba.save(stream,format='PNG')
    decoded=decode_upscale(stream.getvalue(),2)
    assert decoded.mode=='RGBA' and decoded.getpixel((0,0))[3]==75
    stream=io.BytesIO();exif=Image.Exif();exif[274]=6
    Image.new('RGB',(12,8)).save(stream,format='JPEG',exif=exif)
    assert decode_upscale(stream.getvalue(),2).size==(8,12)


def test_animated_and_non_image_formats_rejected():
    from m5diffusion.ui.server import decode_upscale
    frames=[Image.new('RGB',(12,8),color) for color in ('red','blue')]
    for format in ('GIF','WEBP'):
        stream=io.BytesIO()
        frames[0].save(stream,format=format,save_all=True,append_images=frames[1:],duration=100,loop=0)
        with pytest.raises(ValueError):decode_upscale(stream.getvalue(),2)
