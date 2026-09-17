"""Temporary, synchronization-heavy diagnostic instrumentation; never production.

Each timed leaf consumes already evaluated inputs. Attention is atomic including
its projections, so nested Linear/SDPA work is not counted twice. Array-method
and arithmetic instrumentation is confined to vendor forward methods.
"""
import ast,inspect,textwrap,time,operator
from collections import defaultdict
import mlx.core as mx
import mlx.nn as nn


def arrays(value):
    if isinstance(value,mx.array):return [value]
    if isinstance(value,(list,tuple)):return [a for x in value for a in arrays(x)]
    if isinstance(value,dict):return [a for x in value.values() for a in arrays(x)]
    return []


class Rewriter(ast.NodeTransformer):
    def visit_BinOp(self,node):
        self.generic_visit(node)
        kind={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'truediv'}.get(type(node.op))
        if kind:return ast.copy_location(ast.Call(ast.Attribute(ast.Name('_instrument',ast.Load()),'binary',ast.Load()),[ast.Constant(kind),node.left,node.right],[]),node)
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if not isinstance(node.func,ast.Attribute):return node
        name=node.func.attr
        methods={'reshape':'reshape','flatten':'reshape','unflatten':'reshape','transpose':'transpose','astype':'cast'}
        if name in methods:
            return ast.copy_location(ast.Call(ast.Attribute(ast.Name('_instrument',ast.Load()),'method',ast.Load()),[ast.Constant(methods[name]),node.func.value,ast.Constant(name),*node.args],node.keywords),node)
        functions={'silu':'activation','gelu':'activation','concatenate':'copy','broadcast_to':'broadcast_view','pad':'copy'}
        if name in functions:
            return ast.copy_location(ast.Call(ast.Attribute(ast.Name('_instrument',ast.Load()),'call',ast.Load()),[ast.Constant(functions[name]),ast.Constant(name),node.func,*node.args],node.keywords),node)
        return node


class OperationProfiler:
    def __init__(self,engine):
        self.engine=engine;self.records=[];self.names={id(m):name for name,m in engine.unet.named_modules()};self.active=False;self.restores=[];self.input_eval_ms=0.
    def call(self,category,name,fn,*args,**kwargs):
        if self.active:return fn(*args,**kwargs)
        inputs=arrays(args)+arrays(kwargs)
        if not inputs:return fn(*args,**kwargs)
        t=time.perf_counter();mx.eval(inputs);mx.synchronize();self.input_eval_ms+=(time.perf_counter()-t)*1000
        t=time.perf_counter();self.active=True
        try:
            out=fn(*args,**kwargs);dispatch=(time.perf_counter()-t)*1000
            mx.eval(arrays(out));mx.synchronize();elapsed=(time.perf_counter()-t)*1000
        finally:self.active=False
        self.records.append({'category':category,'name':name,'synchronized_wall_ms':elapsed,'python_graph_construction_ms':dispatch,'input_shapes':[list(x.shape) for x in inputs],'output_shapes':[list(x.shape) for x in arrays(out)]})
        return out
    def method(self,category,value,name,*args,**kwargs):
        return self.call(category,name,lambda value,*a,**kw:getattr(value,name)(*a,**kw),value,*args,**kwargs)
    def binary(self,kind,left,right):
        return self.call('residual' if kind in ('add','sub') else 'elementwise',kind,getattr(operator,kind),left,right)
    def patch(self,obj,name,value):
        self.restores.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    def __enter__(self):
        import stable_diffusion.unet as unet
        from m5diffusion.mlx_backend.attention import PaddedAttention
        for cls,category in [(nn.Conv2d,'convolution'),(nn.Linear,'linear'),(nn.GroupNorm,'normalization'),(nn.LayerNorm,'normalization'),(nn.MultiHeadAttention,'attention'),(PaddedAttention,'attention'),(nn.SinusoidalPositionalEncoding,'timestep_embedding')]:
            original=cls.__call__
            def wrapped(module,*args,_original=original,_category=category,**kwargs):
                name=self.names.get(id(module),type(module).__name__)
                category=_category
                if category=='attention':category='self_attention' if name.endswith('attn1') else 'cross_attention'
                if name.startswith('time_embedding'):category='timestep_embedding'
                return self.call(category,name,lambda *a,**kw:_original(module,*a,**kw),*args,**kwargs)
            self.patch(cls,'__call__',wrapped)
        for cls in [unet.TimestepEmbedding,unet.TransformerBlock,unet.Transformer2D,unet.ResnetBlock2D,unet.UNetBlock2D,unet.UNetModel]:
            original=cls.__call__;tree=ast.parse(textwrap.dedent(inspect.getsource(original)));tree=ast.fix_missing_locations(Rewriter().visit(tree));env=dict(original.__globals__,_instrument=self)
            exec(compile(tree,inspect.getsourcefile(original),'exec'),env);self.patch(cls,'__call__',env['__call__'])
        original=unet.upsample_nearest;tree=ast.fix_missing_locations(Rewriter().visit(ast.parse(textwrap.dedent(inspect.getsource(original)))));env=dict(original.__globals__,_instrument=self);exec(compile(tree,inspect.getsourcefile(original),'exec'),env);self.patch(unet,'upsample_nearest',env['upsample_nearest'])
        for cls in [unet.TimestepEmbedding,unet.TransformerBlock,unet.Transformer2D,unet.ResnetBlock2D,unet.UNetBlock2D,unet.UNetModel]:cls.__call__.__globals__['upsample_nearest']=unet.upsample_nearest
        return self
    def __exit__(self,*args):
        for obj,name,original in reversed(self.restores):setattr(obj,name,original)
    def summarize(self):
        grouped=defaultdict(lambda:{'calls':0,'synchronized_wall_ms':0.,'python_graph_construction_ms':0.})
        for row in self.records:
            summary=grouped[row['category']];summary['calls']+=1
            for key in ('synchronized_wall_ms','python_graph_construction_ms'):summary[key]+=row[key]
        operations={}
        for row in self.records:
            key=(row['category'],row['name'],str(row['input_shapes']))
            if key not in operations:operations[key]={**row,'calls':0,'total_synchronized_wall_ms':0.,'total_graph_construction_ms':0.}
            item=operations[key];item['calls']+=1;item['total_synchronized_wall_ms']+=row['synchronized_wall_ms'];item['total_graph_construction_ms']+=row['python_graph_construction_ms']
        for item in operations.values():
            item['mean_synchronized_wall_ms']=item['total_synchronized_wall_ms']/item['calls']
            item.pop('synchronized_wall_ms');item.pop('python_graph_construction_ms')
        for category in ('convolution','self_attention','cross_attention','normalization','activation','residual','timestep_embedding','CFG','scheduler','reshape','transpose','copy','cast'):grouped[category]
        total=sum(row['synchronized_wall_ms'] for row in self.records)
        for summary in grouped.values():summary['instrumented_share_percent']=100*summary['synchronized_wall_ms']/total if total else 0
        return {'categories':dict(grouped),'top20':sorted(operations.values(),key=lambda x:x['total_synchronized_wall_ms'],reverse=True)[:20],'operations':self.records,'input_eval_boundary_ms':self.input_eval_ms,'recorded_operation_wall_ms':total}
