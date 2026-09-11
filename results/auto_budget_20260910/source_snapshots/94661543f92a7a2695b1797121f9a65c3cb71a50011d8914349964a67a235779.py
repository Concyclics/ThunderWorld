"""Capture held-current BF16 linear inputs/outputs at four denoising phases."""
from pathlib import Path
import json
import torch


class Capture:
    def __init__(self,transformer,study,directory):
        self.study=study;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.case=None;self.task=None;self.seed=None;self.rows=[];self.counter=0;self.enabled=False
        for i,layer in enumerate(transformer.layers):
            for name,child in layer.mlp_moe_gen.named_children():
                if isinstance(child,torch.nn.Linear):child.register_forward_hook(self.hook(f'{i}.{name}'))

    def hook(self,name):
        def callback(module,args,output):
            s=self.study
            probes=[1,s.steps//3,2*s.steps//3,s.steps-2]
            if not self.enabled or s.mode!='bf16' or s.step not in probes:return
            x=args[0].reshape(-1,module.in_features);y=output.reshape(-1,module.out_features)
            # Contiguous 64-row probe; local proxy only, not full-layer quality.
            start=min((s.branch+int(name.split('.')[0]))%max(1,(len(x)+127)//128)*128,max(0,len(x)-64))
            rec=dict(name=name,case=self.case,task=self.task,seed=self.seed,step=s.step,steps=s.steps,
                     branch=s.branch,m=len(x),n=module.out_features,k=module.in_features,start=start)
            filename=f'{self.counter:05d}.pt';self.counter+=1
            torch.save(dict(x=x[start:start+64].detach().cpu().contiguous(),y=y[start:start+64].detach().cpu().contiguous(),metadata=rec),self.directory/filename)
            self.rows.append(dict(file=filename,**rec))
        return callback

    def save(self,model_revision):
        (self.directory/'manifest.json').write_text(json.dumps(dict(complete=True,model_revision=model_revision,rows=self.rows,
            scope='BF16 local 64-row contiguous probes at four steps; proxy calibration, not end-to-end quality; evaluation uses separate seeds/cases'),indent=2)+'\n')
