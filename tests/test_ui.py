import json
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from m5diffusion.ui.server import Jobs, handler_for, parse_request

@pytest.mark.parametrize('data', [{'width':513}, {'steps':1}, {'cfg':float('nan')}, {'seed':-1}, {'steps':3.5}, {'prompt':[]}, {'model':'sdxl'}, {'lora':'x'}, {'wrong':1}])
def test_invalid_input(data):
    with pytest.raises(ValueError):
        parse_request(data)

def test_serial_reuse_and_persistence(tmp_path):
    calls=[]
    class Engine:
        load_time=.01
        def __init__(self):
            calls.append(('init',threading.get_ident()))
        def generate(self, request):
            calls.append(('generate',threading.get_ident()))
            return np.arange(12,dtype=np.float32).reshape(2,2,3)/12, {'total_time':.01}
    jobs=Jobs(factory=Engine,output=tmp_path)
    ids=[jobs.submit(parse_request({'steps':2})) for _ in range(2)]
    jobs.queue.join()
    assert [c[0] for c in calls]==['init','generate','generate']
    assert len({c[1] for c in calls})==1
    for job_id in ids:
        assert jobs.get(job_id)['status']=='done'
        assert (tmp_path/f'{job_id}.png').exists()
        assert json.loads((tmp_path/f'{job_id}.json').read_text())['request']['steps']==2

def test_loopback_api_without_loading_model(tmp_path):
    def forbidden():
        raise AssertionError('Model must not load in health/validation tests')
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(Jobs(factory=forbidden,output=tmp_path)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        assert json.load(urllib.request.urlopen(base+'/api/health'))['app']=='M5Diffusion'
        assert 'LoRA'.encode() in urllib.request.urlopen(base+'/').read()
        req=urllib.request.Request(base+'/api/jobs',data=b'{"steps":0}',headers={'Content-Type':'application/json'})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code==400
        req=urllib.request.Request(base+'/api/jobs',data=b'{}',headers={'Origin':'https://example.com'})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code==403
    finally:
        server.shutdown();server.server_close()

@pytest.mark.parametrize('extra', [
    {'loras':'x'}, {'loras':[{'id':'../x','scale':1}]},
    {'loras':[{'id':'lora-'+'a'*64,'scale':float('nan')}]},
    {'loras':[{'id':'lora-'+'a'*64,'scale':True}]},
    {'loras':[{'id':'lora-'+'a'*64,'scale':3}]},
    {'loras':[{'id':'lora-'+'a'*64,'scale':1}]*2},
])
def test_invalid_adapter_selection(extra):
    with pytest.raises(ValueError):parse_request(extra)


def test_lora_switch_without_base_reload(tmp_path):
    calls=[]
    lid='lora-'+'a'*64
    class Library:
        def catalog(self):return {'models':[{'id':'sd15'}], 'loras':[{'id':lid}]}
        def resolve_lora(self, identity):return tmp_path/'style.safetensors'
    class Engine:
        load_time=.01
        def __init__(self):calls.append(('init',threading.get_ident()))
        def set_loras(self, values):calls.append(('loras',list(values)))
        def generate(self, request):
            calls.append(('generate',threading.get_ident()))
            return np.arange(12,dtype=np.float32).reshape(2,2,3)/12,{'total_time':.01}
    jobs=Jobs(factory=Engine,output=tmp_path,library=Library())
    ids=[]
    for selected in [[],[{'id':lid,'scale':.7}],[]]:
        ids.append(jobs.submit(parse_request({}),loras=selected));jobs.queue.join()
    assert sum(x[0]=='init' for x in calls)==1
    assert [x[1] for x in calls if x[0]=='loras']==[[],[(tmp_path/'style.safetensors',.7)],[]]
    assert all(jobs.get(i)['status']=='done' for i in ids)
    metadata=json.loads((tmp_path/f'{ids[1]}.json').read_text())
    assert metadata['loras']==[{'id':lid,'scale':.7}]
    with pytest.raises(ValueError):jobs.submit(parse_request({}),loras=[{'id':'lora-'+'b'*64,'scale':1}])


def test_library_http_stream_and_local_import(tmp_path):
    import io
    class Library:
        def catalog(self):return {'models':[{'id':'sd15','name':'Stable Diffusion 1.5'}],'loras':[]}
        def ingest(self, stream, length, kind, name):
            assert stream.read(length)==b'binary-test'
            assert kind=='lora' and name=='style.safetensors'
            return {'id':'lora-'+'a'*64,'name':name}
        def import_local(self,path,kind):
            assert path=='/tmp/style.safetensors' and kind=='lora'
            return {'id':'lora-'+'a'*64}
    jobs=Jobs(factory=lambda:None,output=tmp_path,library=Library())
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(jobs))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        assert json.load(urllib.request.urlopen(base+'/api/library'))['models'][0]['id']=='sd15'
        req=urllib.request.Request(base+'/api/library/upload?kind=lora&name=style.safetensors',data=b'binary-test',headers={'Content-Type':'application/octet-stream'})
        with urllib.request.urlopen(req) as r:assert r.status==201 and json.load(r)['name']=='style.safetensors'
        req=urllib.request.Request(base+'/api/library/import',data=json.dumps({'path':'/tmp/style.safetensors','kind':'lora'}).encode(),headers={'Content-Type':'application/json'})
        assert json.load(urllib.request.urlopen(req))['id']=='lora-'+'a'*64
        req=urllib.request.Request(base+'/api/library/import',data=b'{}',headers={'Origin':'https://example.com'})
        with pytest.raises(urllib.error.HTTPError) as exc:urllib.request.urlopen(req)
        assert exc.value.code==403
    finally:server.shutdown();server.server_close()
