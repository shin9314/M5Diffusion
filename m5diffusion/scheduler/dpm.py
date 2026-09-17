"""Shared FP32 Karras schedule and DPM++ 2M coefficients for both backends."""
import numpy as np

def schedule(steps, config):
    if steps < 2: raise ValueError('steps must be >= 2')
    betas=np.linspace(np.sqrt(config['beta_start']),np.sqrt(config['beta_end']),config['num_train_timesteps'],dtype=np.float64)**2
    a=np.cumprod(1-betas); train=np.sqrt((1-a)/a)
    rho=7.; ramp=np.linspace(0,1,steps)
    sigmas=(train[-1]**(1/rho)+ramp*(train[0]**(1/rho)-train[-1]**(1/rho)))**rho
    ts=np.interp(np.log(sigmas),np.log(train),np.arange(len(train))).astype(np.float32)
    sigmas=np.append(sigmas,0).astype(np.float32)
    coeff=[]
    for i in range(steps):
        sigma=float(sigmas[i]); nxt=float(sigmas[i+1])
        if nxt==0: coeff.append((sigma,0.,1.,1.,0.));continue
        h=np.log(sigma/nxt)
        ratio=nxt/sigma; factor=-np.expm1(-h)
        if i==0: ca,cb=1.,0.
        else:
            hp=np.log(float(sigmas[i-1])/sigma);r=hp/h
            ca,cb=1+1/(2*r),-1/(2*r)
        coeff.append((sigma,ratio,factor,ca,cb))
    return sigmas,ts,np.array(coeff,dtype=np.float32)

def initial_noise(seed,width,height):
    # Exact same underlying bytes; transpose is only a layout change for MLX.
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((1,4,height//8,width//8),dtype=np.float32)
