import json
import sys
import pytest
from benchmark_controlled import descriptive,session_order,summarize_result,matched_conditions,cooldown,run_variant
from m5diffusion.profiling import telemetry

def fixture_result():
 return {'request':{'seed':12345},'model':'sd15','runs':[{'run':i,'cold':i==0,'total_time':float(1000 if i==0 else i),'diffusion_time':float(i+1),'seconds_per_step':.5,'vae_time':1.,'thermal_before':{'thermal_state':None},'thermal_after':{'thermal_state':1}} for i in range(11)]}

def test_stats_and_warmup_exclusion():
 result=summarize_result(fixture_result())['metrics']['total_time']
 assert result['n']==10 and result['median']==5.5 and result['max']==10
 assert result['p25']==3.25 and result['p75']==7.75
 assert result['std']==pytest.approx(3.027650354)
 assert descriptive([3])['std']==0

@pytest.mark.parametrize('values',[[],[float('nan')],[float('inf')],[-1]])
def test_invalid_metrics(values):
 with pytest.raises(ValueError):descriptive(values)

def test_counterbalanced_order_and_required_counts():
 assert session_order(3)==[('A','B'),('B','A'),('A','B')]
 with pytest.raises(ValueError):session_order(2)
 data=fixture_result();data['runs'].pop()
 with pytest.raises(ValueError):summarize_result(data)
 data=fixture_result();del data['runs'][3]['thermal_after']
 with pytest.raises(ValueError):summarize_result(data)
 data=fixture_result();data['runs'][3]['run']=2
 with pytest.raises(ValueError):summarize_result(data)

def test_cooldown_actual_elapsed_and_snapshots():
 t=[0.];calls=[]
 def sleep(seconds):calls.append(seconds);t[0]+=seconds
 value=cooldown(60,snapshot=lambda:{'thermal_state':None},sleeper=sleep,clock=lambda:t[0])
 assert value['elapsed_seconds']>=60 and len(calls)==60
 assert value['after']['thermal_state'] is None
 with pytest.raises(ValueError):cooldown(59)

def test_protocol_mismatch():
 a=fixture_result();b=fixture_result();b['request']['seed']=0
 with pytest.raises(ValueError,match='request'):matched_conditions(a,b)
 b=fixture_result();b['precision']='int8'
 with pytest.raises(ValueError,match='precision'):matched_conditions(a,b)

def test_missing_thermal_probe_is_null(monkeypatch):
 monkeypatch.setattr(telemetry,'_read_command',lambda *a,**k:None)
 data=telemetry.thermal_snapshot()
 assert data['thermal_state'] is None and data['gpu_temperature_c'] is None
 assert data['gpu_frequency_mhz'] is None and data['power_w'] is None

def test_thermal_state_not_confused_with_temperature(monkeypatch):
 monkeypatch.setattr(telemetry,'_read_command',lambda *a,**k:json.dumps({'thermal_state':2}))
 data=telemetry.thermal_snapshot()
 assert data['thermal_state_name']=='serious' and data['temperature_c'] is None
 monkeypatch.setattr(telemetry,'_read_command',lambda *a,**k:json.dumps({'thermal_state':False}))
 assert telemetry.thermal_snapshot()['thermal_state'] is None

def test_variant_subprocess_collects_exact_artifact_without_shell(tmp_path):
 (tmp_path/'benchmark/results').mkdir(parents=True)
 (tmp_path/'logs').mkdir()
 script=tmp_path/'fake.py'
 script.write_text("import sys,json,pathlib\nlabel=sys.argv[sys.argv.index('--label')+1]\nn=int(sys.argv[sys.argv.index('--repeat')+1])\npathlib.Path('benchmark/results/stamp-'+label+'.json').write_text(json.dumps({'runs':list(range(n))}))\nprint(json.dumps({'run':0,'total_time':1}))\n")
 path,data,argv=run_variant([sys.executable,str(script)],'test-A',11,tmp_path,tmp_path/'logs')
 assert len(data['runs'])==11 and path.name=='stamp-test-A.json'
 assert argv[-4:]==['--repeat','11','--label','test-A']
 with pytest.raises(ValueError):run_variant([sys.executable,'--repeat','2'],'bad',11,tmp_path,tmp_path/'logs')


def test_provenance_ignores_output_json_but_detects_source_changes(tmp_path):
 import subprocess
 from benchmark_controlled import provenance
 def git(*args):
  return subprocess.run(['git',*args],cwd=tmp_path,check=True,capture_output=True)
 git('init')
 (tmp_path/'benchmark.py').write_text('value = 1\n')
 (tmp_path/'m5diffusion').mkdir()
 (tmp_path/'m5diffusion/config.json').write_text('{"x":1}')
 git('add','.')
 git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture')
 original=provenance(tmp_path)['source_sha256']
 for directory in ['benchmark/controlled/run','benchmark/results','outputs','work','models','cache']:
  dest=tmp_path/directory;dest.mkdir(parents=True,exist_ok=True)
  (dest/'summary.json').write_text('{"new":true}')
  (dest/'temporary.py').write_text('output = 1')
 assert provenance(tmp_path)['source_sha256']==original
 git('add','benchmark')
 assert provenance(tmp_path)['source_sha256']==original
 (tmp_path/'benchmark.py').write_text('value = 2\n')
 assert provenance(tmp_path)['source_sha256']!=original
 (tmp_path/'benchmark.py').write_text('value = 1\n')
 (tmp_path/'m5diffusion/config.json').write_text('{"x":2}')
 assert provenance(tmp_path)['source_sha256']!=original


def test_complete_three_pair_protocol_without_gpu(tmp_path,monkeypatch):
 import benchmark_controlled as runner
 import csv
 monkeypatch.setattr(runner,'ROOT',tmp_path)
 monkeypatch.setattr(sys,'argv',['benchmark_controlled.py','--name','cpu-test'])
 monkeypatch.setattr(runner,'provenance',lambda root:{'source_sha256':'unchanged','git_sha':'test'})
 monkeypatch.setattr(runner,'model_hashes',lambda model:{'weights.safetensors':'abc'})
 monkeypatch.setattr(runner,'system_snapshot',lambda:{'thermal_state':None})
 pauses=[];order=[]
 def pause(seconds):
  pauses.append(seconds)
  return {'elapsed_seconds':seconds,'before':{'thermal_state':None},'after':{'thermal_state':None}}
 def variant(command,label,repeat,root,output):
  order.append(label.rsplit('-',1)[1]);data=fixture_result()
  data['model']=str(tmp_path/'models/sd15')
  return output/(label+'.json'),data,command
 monkeypatch.setattr(runner,'cooldown',pause)
 monkeypatch.setattr(runner,'run_variant',variant)
 runner.main()
 path=next((tmp_path/'benchmark/controlled').glob('*/summary.json'))
 report=json.loads(path.read_text())
 assert report['status']=='complete'
 assert order==['A','B','B','A','A','B']
 assert pauses==[60]*6
 assert len(report['runs'])==6 and len(report['comparisons'])==3
 assert all(r['summary']['metrics']['total_time']['n']==10 for r in report['runs'])
 assert report['paired_ratio_summary']['diffusion_time']['median']==1
 with path.with_suffix('.csv').open() as stream:rows=list(csv.DictReader(stream))
 assert len(rows)==24 and {r['metric'] for r in rows}==set(runner.METRICS)


def test_source_change_during_final_variant_is_rejected(tmp_path,monkeypatch):
 import benchmark_controlled as runner
 monkeypatch.setattr(runner,'ROOT',tmp_path)
 monkeypatch.setattr(sys,'argv',['benchmark_controlled.py','--name','change-test'])
 calls=[0]
 monkeypatch.setattr(runner,'provenance',lambda root:{'source_sha256':'before' if calls[0]<6 else 'changed'})
 monkeypatch.setattr(runner,'model_hashes',lambda model:{'weights':'same'})
 monkeypatch.setattr(runner,'system_snapshot',lambda:{})
 monkeypatch.setattr(runner,'cooldown',lambda seconds:{'elapsed_seconds':seconds})
 def variant(command,label,repeat,root,output):
  calls[0]+=1;data=fixture_result();data['model']=str(tmp_path/'models/sd15')
  return output/(label+'.json'),data,command
 monkeypatch.setattr(runner,'run_variant',variant)
 with pytest.raises(RuntimeError,match='while variant'):runner.main()
 report=json.loads(next((tmp_path/'benchmark/controlled').glob('*/summary.json')).read_text())
 assert report['status']=='failed' and len(report['runs'])==5
