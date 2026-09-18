"""Build pinned official SageAttention3 and compare complete attention outputs.

The wrapper copies K because upstream preprocessing subtracts its mean in place.
GQA is explicitly expanded; its duplicate packing work is not hidden.
"""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from baselines import sha,write,git


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('sources','capture','cutlass','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    lock=json.loads(a.sources.read_text());entry=lock['sources']['sageattention'];source=Path(entry['path'])
    if git(source,'rev-parse','HEAD')!=entry['commit']:raise ValueError('Upstream changed')
    work=a.output/'source';shutil.copytree(source/'sageattention3_blackwell',work,ignore=shutil.ignore_patterns('build','*.so','__pycache__'))
    cutlass=work/'csrc/cutlass';cutlass.parent.mkdir(exist_ok=True);cutlass.symlink_to(a.cutlass.resolve(),target_is_directory=True)
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        resource=select_shared_device()
        build=dict(complete=False,source_commit=entry['commit'],cutlass_commit=git(a.cutlass,'rev-parse','HEAD'),resource=resource,
            source_lock_sha256=sha(a.sources),patches=[],command=[sys.executable,'setup.py','build_ext','--inplace'])
        write(a.output/'build.json',build)
        with (a.output/'build.log').open('w') as log:
            subprocess.run(build['command'],cwd=work,env={**os.environ,'MAX_JOBS':'4','TORCH_CUDA_ARCH_LIST':'12.0a'},stdout=log,stderr=subprocess.STDOUT,check=True)
        build.update(complete=True,libraries={p.name:sha(p) for p in work.glob('*.so')});write(a.output/'build.json',build)
        sys.path.insert(0,str(work))
        import torch
        from sageattn3 import sageattn3_blackwell
        from thunderworld_fp4 import fp4_attention,RepairConfig
        torch.set_num_threads(8)
        capture=json.loads((a.capture/'manifest.json').read_text())
        result=dict(complete=False,performance_eligible=False,rows=[],resource=resource,build_sha256=sha(a.output/'build.json'),
            scope='Official native SageAttention3 plus explicit GQA expansion/copy; full attention output quality only. '
                  'No simulated timings; packing, mean correction and GQA duplication are part of this wrapper.')
        with torch.inference_mode():
            for row in capture['rows']:
                if row['kind']!='attention' or row['name'] not in ('0','14','27'):continue
                if sha(a.capture/row['file'])!=row['sha256']:raise ValueError('Corrupt panel')
                panel=torch.load(a.capture/row['file'],weights_only=True)
                q,k,v=(panel[key].cuda() for key in ('q','k','v'));gold=panel['y'].cuda().float()
                ratio=q.shape[2]//k.shape[2]
                original_k=k.clone()
                pred=sageattn3_blackwell(q.transpose(1,2).contiguous(),
                    k.repeat_interleave(ratio,2).transpose(1,2).contiguous(),
                    v.repeat_interleave(ratio,2).transpose(1,2).contiguous()).transpose(1,2).float()
                if not torch.equal(k,original_k) or not torch.isfinite(pred).all():raise ValueError('Invalid Sage wrapper output')
                methods={'sageattention3':pred,'fp4_base':fp4_attention(q,k,v,RepairConfig(coverage=0.,query_tile=32,transpose_v=True)).float(),
                    'fp4_full_three':fp4_attention(q,k,v,RepairConfig(coverage=1.,terms=3,query_tile=32,transpose_v=True,sparse_pack=True)).float()}
                errors={name:float((out-gold).square().sum()/gold.square().sum().clamp_min(1e-20)) for name,out in methods.items()}
                result['rows'].append(dict(layer=row['name'],seed=row['seed'],phase=row['phase'],branch=row['branch'],nmse=errors,
                    gqa_replication=ratio,kv_replication_bf16_bytes=2*ratio*(k.numel()+v.numel()),base_connections_preserved=True))
                write(a.output/'probe.json',result)
            result['complete']=True;write(a.output/'probe.json',result)


if __name__=='__main__':main()
