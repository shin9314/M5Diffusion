"""Host telemetry, with low-frequency read-only GPU registry sampling.

GPU utilization is driver-reported whole-device utilization, not process-specific.
Swap values are occupied system-wide swap, not disk I/O or this process's swap.
Growth is measured relative to entry; pre-existing occupied swap may persist.
"""
import json,subprocess,time,threading,plistlib
import psutil
from m5diffusion.engine.common import ROOT


def parse_gpu_utilization(data):
    """Parse ioreg XML; unavailable or invalid counters remain None, never zero."""
    try:
        nodes=plistlib.loads(data)
        pending=list(nodes) if isinstance(nodes,list) else [nodes]
        values=[]
        while pending:
            node=pending.pop()
            if not isinstance(node,dict):continue
            value=node.get('PerformanceStatistics',{}).get('Device Utilization %')
            if isinstance(value,(int,float)) and not isinstance(value,bool) and 0<=value<=100:
                values.append(float(value))
            pending.extend(node.get('IORegistryEntryChildren',[]))
        return max(values) if values else None
    except (ValueError,TypeError,plistlib.InvalidFileException,AttributeError):
        return None


def gpu_utilization():
    try:
        result=subprocess.run(['/usr/sbin/ioreg','-a','-r','-c','AGXAccelerator','-d','1'],capture_output=True,timeout=.8)
        return parse_gpu_utilization(result.stdout) if result.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):
        return None


def _read_command(command, timeout=1.5):
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
        return result.stdout.strip() if result.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):
        return None


def thermal_snapshot():
    """Coarse NSProcessInfo thermal state; unavailable sensors are explicit nulls.

    No sudo, powermetrics, private SMC assumptions, or nominal frequencies
    misrepresented as live CPU/GPU frequency.
    """
    result={'timestamp':time.time(),'thermal_state':None,'thermal_state_name':None,
            'thermal_pressure':None,'soc_temperature_c':None,'gpu_temperature_c':None,
            'cpu_temperature_c':None,'temperature_c':None,'power_w':None,
            'gpu_frequency_mhz':None,'cpu_frequency_mhz':None}
    raw=_read_command([str(ROOT/'work/metal-probe')])
    try:
        state=json.loads(raw)['thermal_state'] if raw else None
        if type(state) is int and 0<=state<=3:
            result['thermal_state']=state
            result['thermal_state_name']=['nominal','fair','serious','critical'][state]
    except (ValueError,TypeError,KeyError):
        pass
    result['power_source_raw']=_read_command(['/usr/bin/pmset','-g','batt'])
    result['thermal_source']='NSProcessInfo.thermalState via local Metal probe' if result['thermal_state'] is not None else 'unavailable'
    result['sensor_note']='Nonprivileged GPU registry exposes utilization, not temperature/frequency/power on this device; absent sensors are null.'
    return result


def system_snapshot():
    v=psutil.virtual_memory();s=psutil.swap_memory()
    result={'system_memory_used_mb':v.used/2**20,'system_memory_available_mb':v.available/2**20,
            'swap_mb':s.used/2**20,'gpu_utilization_percent':gpu_utilization(),**thermal_snapshot()}
    result['memory_pressure_raw']=_read_command(['memory_pressure','-Q'])
    result['thermal_pressure_raw']=_read_command(['/usr/bin/pmset','-g','therm'])
    return result

class Sampler:
    def __enter__(self):
        self.stop=threading.Event();self.samples=[];self.p=psutil.Process();self.p.cpu_percent()
        self.initial_swap_mb=psutil.swap_memory().used/2**20
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start();return self
    def run(self):
        next_gpu=0.
        while not self.stop.is_set():
            now=time.monotonic();gpu=None
            # CPU/RSS retain 250ms sampling; query GPU only once per second.
            if now>=next_gpu:
                gpu=gpu_utilization();next_gpu=now+1.
            self.samples.append({'t':time.time(),'rss_mb':self.p.memory_info().rss/2**20,'cpu_percent':self.p.cpu_percent(),'swap_mb':psutil.swap_memory().used/2**20,'gpu_utilization_percent':gpu})
            self.stop.wait(.25)
    def __exit__(self,*a):self.stop.set();self.thread.join()
    def summary(self):
        gpu=[x['gpu_utilization_percent'] for x in self.samples if x.get('gpu_utilization_percent') is not None]
        peak_swap=max((x['swap_mb'] for x in self.samples),default=self.initial_swap_mb)
        return {'peak_rss_mb':max((x['rss_mb'] for x in self.samples),default=None),'cpu_percent_mean':sum(x['cpu_percent'] for x in self.samples)/len(self.samples) if self.samples else None,'peak_swap_mb':peak_swap,'initial_swap_mb':self.initial_swap_mb,'swap_growth_mb':max(0.,peak_swap-self.initial_swap_mb),'gpu_utilization_percent_mean':sum(gpu)/len(gpu) if gpu else None,'gpu_utilization_percent_peak':max(gpu) if gpu else None,'gpu_utilization_sample_count':len(gpu)}
