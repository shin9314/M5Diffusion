import numpy as np
from m5diffusion.scheduler.dpm import schedule,initial_noise

def test_noise_identical_and_layout_roundtrip():
 a=initial_noise(12345,512,512);b=initial_noise(12345,512,512)
 assert np.array_equal(a,b)
 assert np.array_equal(a,a.transpose(0,2,3,1).transpose(0,3,1,2))
def test_dpm_first_and_last_steps():
 config={'beta_start':.00085,'beta_end':.012,'num_train_timesteps':1000}
 s,t,k=schedule(20,config)
 assert len(t)==20 and s[-1]==0 and np.all(np.diff(s)<0)
 assert np.isfinite(k).all() and k[0,3]==1 and k[0,4]==0
 assert np.array_equal(k[-1,1:],np.array([0,1,1,0],np.float32))
def test_schedule_timestep_range():
 s,t,k=schedule(20,{'beta_start':.00085,'beta_end':.012,'num_train_timesteps':1000})
 assert abs(t[0]-999)<.001 and abs(t[-1])<.001
