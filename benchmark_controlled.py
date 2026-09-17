"""Counterbalanced, cooldown-separated benchmark sessions. Never parallelizes GPU work."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from m5diffusion.engine.common import ROOT
from m5diffusion.profiling.telemetry import system_snapshot, thermal_snapshot

METRICS = ('total_time', 'diffusion_time', 'seconds_per_step', 'vae_time')
CONDITIONS = ('request','model','sampler','schedule','rng','precision','timing_schema')

def percentile(values, fraction):
    values=sorted(values);position=(len(values)-1)*fraction
    lo=math.floor(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)

def descriptive(values):
    if not values or any(not math.isfinite(x) or x<0 for x in values):raise ValueError('Finite nonnegative timing samples required')
    return {'n':len(values),'median':statistics.median(values),'mean':statistics.mean(values),'min':min(values),'max':max(values),'std':statistics.stdev(values) if len(values)>1 else 0.,'p25':percentile(values,.25),'p75':percentile(values,.75)}

def session_order(sessions):
    if sessions<3:raise ValueError('At least three paired sessions required')
    return [('A','B') if i%2==0 else ('B','A') for i in range(sessions)]

def summarize_result(data, repeat=11):
    rows=data.get('runs',[])
    if len(rows)!=repeat or repeat<11:raise ValueError('Each variant must finish at least eleven images')
    if not rows[0].get('cold') or any(r.get('cold') for r in rows[1:]):raise ValueError('Exactly the first image must be warmup')
    if [r.get('run') for r in rows]!=list(range(repeat)):raise ValueError('Missing or repeated image indices')
    for row in rows:
        if 'thermal_before' not in row or 'thermal_after' not in row:raise ValueError('Per-image thermal snapshots required; update benchmark.py')
    warm=rows[1:]
    metrics={k:descriptive([r[k] for r in warm]) for k in METRICS}
    early=statistics.median(r['diffusion_time'] for r in warm[:5])
    late=statistics.median(r['diffusion_time'] for r in warm[-5:])
    return {'warmup_excluded':1,'metrics':metrics,'diffusion_late_vs_early_percent':100*(late/early-1) if early else None,'degradation_note':'Later/earlier timing change; not causally attributed to temperature.'}

def matched_conditions(a,b):
    mismatch=[k for k in CONDITIONS if a.get(k)!=b.get(k)]
    if mismatch:raise ValueError('Unmatched benchmark conditions: '+', '.join(mismatch))

def provenance(root):
    def git(*args):
        return subprocess.run(['git',*args],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    files=git('ls-files','--cached','--others','--exclude-standard').splitlines()
    h=hashlib.sha256()
    for name in sorted(set(files)):
        path=root/name
        relative=Path(name)
        source_tree=relative.parts[0] in ('m5diffusion','vendor','tests')
        root_source=len(relative.parts)==1 and (path.suffix=='.py' or relative.name.startswith('requirements') or relative.name in ('pyproject.toml','setup.cfg','pytest.ini'))
        allowed_suffix=path.suffix in ('.py','.metal','.m','.mm','.h','.hpp','.cpp','.json','.txt','.toml','.cfg','.ini')
        if (source_tree or root_source) and allowed_suffix and path.is_file() and not path.is_symlink():
            h.update(name.encode());h.update(b'\0');h.update(path.read_bytes());h.update(b'\0')
    return {'git_sha':git('rev-parse','HEAD'),'git_status':git('status','--porcelain'),'source_sha256':h.hexdigest(),'python':sys.version,'argv':sys.argv}

def model_hashes(path):
    files=sorted(path.rglob('*.safetensors'))+sorted(path.rglob('*.json'))+sorted(path.rglob('*.txt'))
    output={}
    for file in files:
        if file.is_symlink():raise ValueError('Model provenance does not follow symlinks')
        h=hashlib.sha256()
        with file.open('rb') as source:
            for block in iter(lambda:source.read(8*2**20),b''):h.update(block)
        output[str(file.relative_to(path))]=h.hexdigest()
    if not output:raise ValueError('No model files found for provenance')
    return output

def cooldown(seconds, snapshot=thermal_snapshot, sleeper=time.sleep, clock=time.monotonic):
    if seconds<60:raise ValueError('Cooldown must be at least sixty seconds')
    start=clock();before=snapshot()
    while clock()-start<seconds:sleeper(min(1.,seconds-(clock()-start)))
    return {'minimum_seconds':seconds,'elapsed_seconds':clock()-start,'before':before,'after':snapshot(),'note':'Fixed minimum cooldown does not prove equal chip temperature.'}

def atomic_json(path,data):
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def run_variant(command, label, repeat, root, output):
    if not command or not all(isinstance(x,str) for x in command):raise ValueError('Variant command must be a JSON argv list')
    if '--repeat' in command or '--label' in command:raise ValueError('Runner owns --repeat and --label')
    argv=command+['--repeat',str(repeat),'--label',label]
    with (output/(label+'.log')).open('w') as log:
        process=subprocess.Popen(argv,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                log.write(line);log.flush()
                try:
                    row=json.loads(line)
                    if 'run' in row:print(json.dumps({'label':label,'image':row['run'],'total':row.get('total_time'),'diffusion':row.get('diffusion_time')}),flush=True)
                except (ValueError,TypeError):pass
            code=process.wait()
        except BaseException:
            process.terminate();process.wait(timeout=20);raise
    if code:raise RuntimeError(f'{label} failed with exit {code}; see its log')
    matches=list((root/'benchmark/results').glob('*-'+label+'.json'))
    if len(matches)!=1:raise ValueError('Expected exactly one benchmark result for unique label')
    return matches[0],json.loads(matches[0].read_text()),argv

def write_csv(path,runs):
    columns=['session','variant','metric','n','median','mean','min','max','std','p25','p75']
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for run in runs:
            for metric,values in run['summary']['metrics'].items():writer.writerow({'session':run['session'],'variant':run['variant'],'metric':metric,**values})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sessions',type=int,default=3);p.add_argument('--repeat',type=int,default=11);p.add_argument('--cooldown',type=float,default=60)
    p.add_argument('--a-command',help='JSON argv list; runner appends --repeat and --label')
    p.add_argument('--b-command',help='JSON argv list; runner appends --repeat and --label')
    p.add_argument('--name',default='controlled');p.add_argument('--model',type=Path,default=ROOT/'models/sd15')
    args=p.parse_args();order=session_order(args.sessions)
    if args.repeat<11 or args.cooldown<60:p.error('Require repeat>=11 and cooldown>=60 seconds')
    if not args.name.replace('-','').replace('_','').isalnum():p.error('name must be simple letters/digits')
    commands={'A':json.loads(args.a_command) if args.a_command else [sys.executable,'benchmark.py','--backend','mps','--model',str(args.model)],'B':json.loads(args.b_command) if args.b_command else [sys.executable,'benchmark.py','--backend','mlx','--attention','padded','--compile','--model',str(args.model)]}
    uid=time.strftime('%Y%m%d-%H%M%S')+'-'+args.name+'-'+str(os.getpid());output=ROOT/'benchmark/controlled'/uid;output.mkdir(parents=True)
    report={'protocol':'counterbalanced-v1','created':time.time(),'order':order,'repeat':args.repeat,'cooldown_min_seconds':args.cooldown,'provenance':provenance(ROOT),'model':str(args.model),'model_files_sha256':model_hashes(args.model),'commands':commands,'before':system_snapshot(),'runs':[],'comparisons':[],'status':'running','notes':['Each pair has two variants; AB/BA/AB gives three independent process starts per variant.','For three pairs the starting order is nearly, not perfectly, balanced.','Warm first image excluded. std is sample standard deviation; quartiles use linear interpolation.','Within-process images are correlated. Per-session paired ratios are descriptive, not independent per-image confidence intervals.','Thermal state is coarse; unavailable temperature/frequency/power stay null. No claim cooldown equalizes temperature.']}
    result_path=output/'summary.json';atomic_json(result_path,report);reference=None
    try:
        for session,pair in enumerate(order,1):
            pair_runs={}
            for variant in pair:
                print(json.dumps({'event':'cooldown_begin','session':session,'variant':variant,'minimum_seconds':args.cooldown}),flush=True)
                pause=cooldown(args.cooldown)
                print(json.dumps({'event':'cooldown_end','session':session,'variant':variant,**pause}),flush=True)
                if provenance(ROOT)['source_sha256']!=report['provenance']['source_sha256']:raise RuntimeError('Source changed during controlled benchmark')
                label=uid+f'-s{session}-{variant}'
                path,data,command=run_variant(commands[variant],label,args.repeat,ROOT,output)
                if provenance(ROOT)['source_sha256']!=report['provenance']['source_sha256']:raise RuntimeError('Source changed while variant was running')
                if Path(data.get('model','')).resolve()!=args.model.resolve():raise ValueError('Variant model differs from hashed provenance model')
                if reference is not None:matched_conditions(reference,data)
                else:reference=data;report['conditions']={k:data.get(k) for k in CONDITIONS}
                summary=summarize_result(data,args.repeat)
                entry={'session':session,'variant':variant,'raw_file':str(path),'command':command,'cooldown':pause,'summary':summary,'images':data['runs']}
                report['runs'].append(entry);pair_runs[variant]=summary
                atomic_json(result_path,report);write_csv(output/'summary.csv',report['runs'])
            report['comparisons'].append({'session':session,'A_div_B_median_speedup':{k:pair_runs['A']['metrics'][k]['median']/pair_runs['B']['metrics'][k]['median'] for k in METRICS}})
        if model_hashes(args.model)!=report['model_files_sha256']:raise RuntimeError('Model files changed during measurement')
        report['paired_ratio_summary']={k:descriptive([r['A_div_B_median_speedup'][k] for r in report['comparisons']]) for k in METRICS}
        report['status']='complete';report['after']=system_snapshot()
    except BaseException as exc:
        report['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed';report['error']=str(exc);raise
    finally:atomic_json(result_path,report)
    print(str(result_path),flush=True)

if __name__=='__main__':main()
