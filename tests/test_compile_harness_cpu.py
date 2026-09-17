"""NumPy stand-in validates isolated step algebra, without importing MLX or running GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def function(path,name):
    tree=ast.parse((ROOT/path).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    return compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec')


def test_scope_step_matches_engine_algebra():
    mx=SimpleNamespace(concatenate=np.concatenate,rsqrt=lambda x:1/np.sqrt(x),float16=np.float16,
                       float32=np.float32,split=np.split)
    rng=np.random.default_rng(12345)
    x=rng.normal(size=(1,4,4,4)).astype(np.float32);old=np.zeros_like(x)
    c=rng.normal(size=(2,3,4)).astype(np.float16);t=np.array([999,999],np.float32)
    def unet(u,t,encoder_x):
        # Synthetic model output varies by conditional batch and preserves shape.
        return (u*np.float16(.3)+encoder_x.mean(axis=(1,2))[:,None,None,None]).astype(np.float16)
    engine=SimpleNamespace(mx=mx,unet=unet)
    env={};exec(function('m5diffusion/engine/mlx_backend.py','_step'),env)
    reference=env['_step']
    def predict(u,t,c,cfg):
        neg,pos=np.split(unet(u,t,c).astype(np.float32),2,axis=0)
        return neg+cfg*(pos-neg)
    candidate_env={'mx':mx,'predict':predict};exec(function('bench_unet.py','step'),candidate_env)
    candidate=candidate_env['step'];xa=x.copy();oa=old.copy()
    for i in range(20):
        k=np.array([1/(i+1),.95,-.1,1.05,-.05],np.float32)
        x,old=reference(engine,x,old,c,t,k,np.float32(7))
        xa,oa=candidate(xa,oa,c,t,k,np.float32(7))
        np.testing.assert_array_equal(x,xa);np.testing.assert_array_equal(old,oa)


def test_multistep_preserves_sequential_dependency():
    def step(x,old,c,t,k,cfg):return x+k+old,t+x
    for block in [2,4,5,10,20]:
        env={'block':block,'step':step};exec(function('bench_unet.py','multi'),env)
        t=np.arange(block);k=np.arange(block)*.3;x=2.;old=0.
        for i in range(block):x,old=step(x,old,None,t[i],k[i],7)
        assert env['multi'](2.,0.,None,t,k,7)==(x,old)
