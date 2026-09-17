"""Profile-directed experiment: 3x3 conv via explicit patches and MLX GEMM."""
import argparse,json
from pathlib import Path
import mlx.core as mx
from m5diffusion.profiling.microbench import measure,error


def im2col_conv(x,w,stride=1,padding=1):
    b,h,ww,ci=x.shape;co,kh,kw,_=w.shape
    x=mx.contiguous(mx.pad(x,((0,0),(padding,padding),(padding,padding),(0,0))))
    hp,wp=x.shape[1:3];oh=(hp-kh)//stride+1;ow=(wp-kw)//stride+1
    patches=mx.as_strided(x,shape=(b,oh,ow,kh,kw,ci),strides=(hp*wp*ci,wp*ci*stride,ci*stride,wp*ci,ci,1))
    patches=patches.reshape(b*oh*ow,kh*kw*ci)
    return (patches @ w.reshape(co,-1).T).reshape(b,oh,ow,co)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('benchmark/optimization/conv-im2col.json'));a=p.parse_args()
    mx.set_cache_limit(2*2**30);mx.set_memory_limit(10*2**30)
    records={}
    for h,ci,co in [(32,1280,1280),(32,1920,640),(16,2560,1280),(64,960,320),(64,320,320)]:
        mx.random.seed(12345)
        x=mx.random.normal((2,h,h,ci)).astype(mx.float16);w=(mx.random.normal((co,3,3,ci))*.01).astype(mx.float16);mx.eval(x,w)
        native=lambda x,w:mx.conv2d(x,w,padding=1)
        ref=native(x,w);mx.eval(ref);row={}
        for name,fn in [('native',native),('im2col',im2col_conv),('compiled_im2col',mx.compile(im2col_conv))]:
            out=fn(x,w);mx.eval(out)
            row[name]={**measure(fn,(x,w),10),**error(out,ref)}
            print(h,ci,co,name,row[name]['median_ms'],row[name]['max_abs_error'],flush=True)
        records[f'{h}:{ci}:{co}']=row
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(records,indent=2))
        mx.clear_cache()
if __name__=='__main__':main()
