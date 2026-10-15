"""Capture exact BF16 AR prefixes without running any denoising GEMM."""
import argparse
import gc
import json
from pathlib import Path
import time

from baselines import sha, write
from super_cached_runtime import PREFIX_FIELDS, describe_prefix, digest, placement, prefix_cpu


class PrefixesCaptured(Exception):
    pass


def collect(pipe, case):
    """Use the actual pipeline input assembly; abort after its first CFG pair.

    The conditional dummy velocity is used only to reach the unconditional
    transformer call. No denoising step or decoded output is ever produced.
    """
    import torch
    from thunderworld_fp4.cosmos.runner import CosmosRunner
    from thunderworld_fp4.cosmos.scheduler import HostScheduleUniPC
    holder = object.__new__(CosmosRunner)
    holder.pipe = pipe
    entries = []
    original = pipe.transformer.forward
    def capture(**kwargs):
        entries.append(prefix_cpu(kwargs))
        if len(entries) == 2:
            raise PrefixesCaptured()
        return [torch.zeros_like(v) for v in kwargs['vision_tokens']], None, None
    pipe.transformer.forward = capture
    pipe.scheduler = HostScheduleUniPC.from_config(pipe.scheduler.config, flow_shift=3., use_karras_sigmas=False)
    try:
        with torch.inference_mode():
            try:
                pipe(**holder._args(case, 6099, 2))
            except PrefixesCaptured:
                pass
    finally:
        pipe.transformer.forward = original
    if len(entries) != 2:
        raise ValueError('Expected exactly two CFG prefixes')
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--cases', nargs='+')
    p.add_argument('--case-file', type=Path)
    p.add_argument('--smoke-only', action='store_true')
    a = p.parse_args()
    if a.case_file:
        if a.cases:raise ValueError('Choose cases or case-file')
        a.cases=json.loads(a.case_file.read_text())
    from super_quality_campaign import check_campaign
    check_campaign(a.campaign)
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    from thunderworld_fp4.cosmos.runner import model_identity
    source = a.campaign/'super/final/manifest.json'
    study = json.loads(source.read_text())
    smoke = json.loads((a.campaign/'super/smoke/manifest.json').read_text())
    cases = {e['case']['id']: e for e in smoke['cases']}
    if not a.smoke_only:
        for e in study['cases']:
            if a.cases is None or e['case']['id'] in a.cases:
                cases[e['case']['id']] = e
    bank = a.bank
    bank.mkdir(parents=True, exist_ok=True)
    path = bank/'manifest.json'
    identity = model_identity(study['model'])
    state = json.loads(path.read_text()) if path.exists() else dict(complete=False,
        model_identity=identity, source_manifest_sha256=sha(source), layers=64,
        prefixes={}, cases={}, source_sha256=sha(__file__),
        runtime_sha256=sha(Path(__file__).with_name('super_cached_runtime.py')),
        scope='Unchanged BF16 AR arithmetic with actual pipeline prefix assembly; no DM or generation output reused')
    if state['model_identity'] != identity or state['source_manifest_sha256'] != sha(source):
        raise ValueError('Changed prefix bank model or corpus')
    if state['source_sha256'] != sha(__file__) or state['runtime_sha256'] != sha(Path(__file__).with_name('super_cached_runtime.py')):
        raise ValueError('Changed prefix preparation code')
    pending = [e for cid,e in cases.items() if cid not in state['cases']]
    if not pending:
        print('All requested prefixes already prepared')
        return
    state['complete'] = False
    write(path,state)
    with shared_worker():
        resource = select_shared_device(72*1024)
        import torch
        import diffusers.utils.import_utils as imports
        imports._xformers_available = False
        from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer
        from thunderworld_fp4.cosmos.ar_cache import prefill
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        start = time.perf_counter()
        transformer, info = Cosmos3OmniTransformer.from_pretrained(str(Path(study['model'])/'transformer'),
            torch_dtype=torch.bfloat16,local_files_only=True,output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError(info)
        pipe = Cosmos3OmniPipeline.from_pretrained(study['model'],transformer=transformer,
            sound_tokenizer=None,safety_checker=None,enable_safety_checker=False,
            torch_dtype=torch.bfloat16,local_files_only=True)
        pipe.transformer.set_attention_backend('_native_flash')
        pipe.vae.enable_tiling();pipe.set_progress_bar_config(disable=True)
        with placement('ar'):
            pipe.to('cuda')
        state['resource'] = resource
        state['load_seconds'] = time.perf_counter()-start
        state['placement'] = pipe._super_cached_placement
        for entry in pending:
            case = entry['case'];cid=case['id'];start=time.perf_counter()
            if sha(case['image']) != entry['image_sha256']:
                raise ValueError('Changed prefix source image')
            prefixes = collect(pipe,case)
            keys = []
            for prefix in prefixes:
                descriptor = describe_prefix(prefix);key=digest(descriptor);keys.append(key)
                if key in state['prefixes']:
                    continue
                inputs = {n: v.to('cuda') if isinstance(v,torch.Tensor) else v for n,v in prefix.items()}
                with torch.inference_mode():
                    for _ in range(2):prefill(pipe.transformer,inputs,validate=False)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        rows = prefill(pipe.transformer,inputs,validate=False)
                    graph.replay();torch.cuda.synchronize()
                    cpu = [{n: row[n].cpu() for n in ('k','v','out')} for row in rows]
                target = bank/'prefixes'/(key+'.pt');target.parent.mkdir(exist_ok=True)
                temp=target.with_suffix('.partial')
                torch.save(dict(model_identity=identity,descriptor=descriptor,rows=cpu),temp)
                temp.replace(target)
                state['prefixes'][key]=dict(file=str(target.relative_to(bank)),sha256=sha(target),bytes=target.stat().st_size,
                    und_len=prefix['und_len'],sequence_length=prefix['sequence_length'])
                del graph, rows, cpu, inputs
                gc.collect();torch.cuda.empty_cache()
            state['cases'][cid]=dict(prefix_keys=keys,case_sha256=digest(case),image_sha256=entry['image_sha256'],
                seconds=time.perf_counter()-start)
            write(path,state)
            print(json.dumps(dict(case=cid,completed_cases=len(state['cases']),prefixes=len(state['prefixes']),
                seconds=state['cases'][cid]['seconds'])),flush=True)
        state['complete']=True
        state['total_storage_bytes']=sum(e['bytes'] for e in state['prefixes'].values())
        write(path,state)


if __name__=='__main__':main()
