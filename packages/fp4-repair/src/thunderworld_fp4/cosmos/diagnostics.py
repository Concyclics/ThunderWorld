"""No events, traces or GPU reads in off mode. Collection finishes after timing."""
from contextlib import nullcontext
import torch


class Diagnostics:
    def __init__(self, mode, capacity):
        self.mode = mode
        self.events = [] if mode == 'off' else [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(capacity)]
        self.stage_events = [] if mode == 'off' else [
            (torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            for _ in range(256)]
        self.reset()

    def reset(self):
        self.count = 0
        self.trace = []
        self.stages = []

    def stage(self,name,fn,*args,**kwargs):
        if self.mode=='off':return fn(*args,**kwargs)
        if len(self.stages)>=len(self.stage_events):raise RuntimeError('Debug stage capacity exceeded')
        i=len(self.stages);self.stages.append(name)
        a,b=self.stage_events[i];a.record()
        result=fn(*args,**kwargs);b.record()
        return result

    def begin(self):
        if self.mode != 'off': self.events[self.count][0].record()
        return torch.cuda.nvtx.range('Transformer') if self.mode == 'trace' else nullcontext()

    def end(self, output):
        if self.mode == 'off': return
        self.events[self.count][1].record()
        if self.mode == 'trace':
            from .trees import clone_tree
            # A fixed prefix avoids per-step linspace/index producer kernels.
            def sketch(x):
                if isinstance(x, torch.Tensor): return x.flatten()[:256].clone()
                if isinstance(x, (tuple, list)): return [sketch(v) for v in x]
                if isinstance(x, dict): return {k: sketch(v) for k,v in x.items()}
                return None
            self.trace.append(sketch(output))
        self.count += 1

    def finish(self):
        if self.mode == 'off': return {'mode': 'off'}
        # Caller has synchronized at the request boundary.
        result = {'mode': self.mode, 'transformer_step_ms': [
            a.elapsed_time(b) for a,b in self.events[:self.count]]}
        result['stages']=[dict(name=name,milliseconds=self.stage_events[i][0].elapsed_time(self.stage_events[i][1]))
                          for i,name in enumerate(self.stages)]
        if self.mode == 'trace':
            def host(x):
                if isinstance(x, torch.Tensor): return x.float().cpu().tolist()
                if isinstance(x, list): return [host(v) for v in x]
                if isinstance(x, dict): return {k:host(v) for k,v in x.items()}
                return x
            result['trajectory_sketches'] = host(self.trace)
        return result
