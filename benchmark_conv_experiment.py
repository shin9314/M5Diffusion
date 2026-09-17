"""Opt-in profile-directed convolution experiment; production untouched."""
import mlx.nn as nn
from bench_conv_im2col import im2col_conv
from m5diffusion.engine.mlx_backend import MLXEngine

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

original_init=MLXEngine.__init__
def experimental_init(self,*args,**kwargs):
    kwargs["convolution"]=False
    original_init(self,*args,**kwargs)
    install(self.unet)
    self._refresh_step()

if __name__=='__main__':
    MLXEngine.__init__=experimental_init
    from benchmark import main
    main()
