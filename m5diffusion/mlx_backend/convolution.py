"""Measured selective im2col convolution for low-resolution SD1.5 layers."""
import mlx.core as mx
import mlx.nn as nn

def im2col_conv(x,w,stride=1,padding=1):
    b,h,ww,ci=x.shape;co,kh,kw,_=w.shape
    x=mx.contiguous(mx.pad(x,((0,0),(padding,padding),(padding,padding),(0,0))))
    hp,wp=x.shape[1:3];oh=(hp-kh)//stride+1;ow=(wp-kw)//stride+1
    patches=mx.as_strided(x,shape=(b,oh,ow,kh,kw,ci),strides=(hp*wp*ci,wp*ci*stride,ci*stride,wp*ci,ci,1))
    patches=patches.reshape(b*oh*ow,kh*kw*ci)
    return (patches @ w.reshape(co,-1).T).reshape(b,oh,ow,co)

class ExperimentalConv(nn.Conv2d):
    def __call__(self,x):
        if max(x.shape[1:3]) <= 32 and x.shape[-1] >= 1280:
            y=im2col_conv(x,self.weight)
            return y+self.bias if 'bias' in self else y
        return super().__call__(x)

def install(unet):
    changed=[]
    for name,module in unet.named_modules():
        if isinstance(module,nn.Conv2d) and module.weight.shape[1:3]==(3,3) and module.stride==(1,1) and module.padding==(1,1) and module.dilation in (1,(1,1)) and module.groups==1:
            module.__class__=ExperimentalConv
            changed.append(name)
    return changed
