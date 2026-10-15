import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines import sha, write
from super_cached_runtime import PrefixBank, cache_class, describe_prefix, digest, place_partition, prefix_cpu


def small_prefix(offset=0):
    import torch
    return dict(input_ids=torch.arange(4)+offset, text_indexes=torch.arange(4),
        position_ids=torch.arange(12),und_len=4,sequence_length=12,
        vision_sequence_indexes=torch.arange(4,12),sound_sequence_indexes=None,action_sequence_indexes=None)


def test_keys_cover_values_positions_and_modality_boundaries():
    p=small_prefix();key=digest(describe_prefix(prefix_cpu(p)))
    assert key!=digest(describe_prefix(prefix_cpu(small_prefix(1))))
    p['position_ids'][0]+=1
    assert key!=digest(describe_prefix(prefix_cpu(p)))
    p['vision_sequence_indexes'][0]=0
    with pytest.raises(ValueError,match='text-only'):prefix_cpu(p)


def test_missing_prefix_and_changed_model_are_rejected(tmp_path):
    write(tmp_path/'manifest.json',dict(model_identity={'revision':'fixed'},layers=2,prefixes={}))
    with pytest.raises(ValueError,match='model identity'):PrefixBank(tmp_path,{'revision':'changed'})
    bank=PrefixBank(tmp_path,{'revision':'fixed'})
    with pytest.raises(ValueError,match='Unprepared'):bank.lookup(small_prefix())


def test_corrupt_cache_is_rejected(tmp_path):
    p=small_prefix();key=digest(describe_prefix(prefix_cpu(p)))
    (tmp_path/'bad.pt').write_bytes(b'broken')
    write(tmp_path/'manifest.json',dict(model_identity={},layers=2,
        prefixes={key:dict(file='bad.pt',sha256='incorrect')}))
    with pytest.raises(ValueError,match='Corrupt'):PrefixBank(tmp_path,{}).lookup(p)


def test_pending_output_aliases_are_restart_safe(tmp_path):
    from super_cached_campaign import make_aliases
    task=dict(id='paibench__0__bf16',method='bf16',cases=['scene'])
    make_aliases(tmp_path,task)
    make_aliases(tmp_path,task)  # log target does not exist until the child starts
    log=tmp_path/'tasks'/task['id']/'generation/scene/generation.log'
    assert log.is_symlink()
    assert log.resolve()==tmp_path/('generate_'+task['id']+'.log')


def test_drain_reconciliation_preserves_completed_receipts(tmp_path):
    from super_cached_campaign import reconcile
    tasks=[dict(id='one',method='bf16',cases=['a']),dict(id='two',method='fp8',cases=['b'])]
    write(tmp_path/'generation_tasks.json',dict(tasks=tasks))
    write(tmp_path/'generation_queue.json',dict(complete=False,completed={}))
    for t in tasks:
        write(tmp_path/'tasks'/t['id']/'generation/progress.json',dict(complete=t['id']=='one',
            identity=dict(cases=t['cases'],method=t['method'])))
    _,q=reconcile(tmp_path)
    assert set(q['completed'])=={'one'}
    first=tmp_path/'tasks/one/generation/progress.json'
    before=first.read_bytes()
    reconcile(tmp_path)
    assert first.read_bytes()==before
    first.write_text('{}')
    with pytest.raises(ValueError,match='Changed completed'):reconcile(tmp_path)


def test_offline_ar_exact_and_graph_buffers_update(tmp_path, monkeypatch):
    import torch
    if not torch.cuda.is_available():pytest.skip('requires isolated SM120 GPU')
    from diffusers import Cosmos3OmniTransformer
    from thunderworld_fp4.cosmos.ar_cache import ARCache
    from thunderworld_fp4.cosmos.engine import JointEngine
    from thunderworld_fp4.cosmos.settings import RunConfig
    from thunderworld_fp4.cosmos.tuning import LaunchTable
    torch.manual_seed(1003);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model=Cosmos3OmniTransformer(hidden_size=256,intermediate_size=512,
        num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=2,
        head_dim=128,vocab_size=1024,use_und_k_norm_for_gen=True).bfloat16().eval()
    cached=copy.deepcopy(model)
    model.to('cuda');model.set_attention_backend('_native_flash');cached.set_attention_backend('_native_flash')
    pipe=SimpleNamespace(transformer=cached,components={'transformer':cached})
    place_partition(pipe,'dm')
    assert cached.layers[0].mlp.up_proj.weight.device.type=='meta'
    assert cached.layers[0].mlp_moe_gen.up_proj.weight.is_cuda
    probe=copy.deepcopy(model).cpu()
    probe.register_parameter('device_probe',torch.nn.Parameter(torch.ones(4,dtype=torch.bfloat16)))
    place_partition(SimpleNamespace(transformer=probe,components={'transformer':probe}),'ar')
    assert next(probe.parameters()).is_cuda
    assert probe.layers[0].mlp_moe_gen.up_proj.weight.device.type=='meta'
    del probe
    cfg=RunConfig(precision='bf16',attention_backend='bf16',execution='eager',resource_mode='shared-calibration')
    reference=JointEngine(model,cfg,LaunchTable());live=ARCache(reference)
    reference.warming=True;reference.branches=2;reference.call=0
    manifest=dict(complete=True,model_identity={},layers=2,prefixes={})
    cases=[]
    with torch.inference_mode():
        for m in (129,416):
            for branch in (0,1):
                for offset in (0,3):
                    und=17
                    kwargs=dict(input_ids=torch.arange(und,device='cuda')+offset+branch,
                        text_indexes=torch.arange(und,device='cuda'),position_ids=torch.arange(und+m,device='cuda'),
                        und_len=und,sequence_length=und+m,
                        vision_tokens=[torch.randn(1,48,1,2,2*m,device='cuda',dtype=torch.bfloat16)],
                        vision_token_shapes=[(1,1,m)],vision_sequence_indexes=torch.arange(und,und+m,device='cuda'),
                        vision_mse_loss_indexes=torch.arange(und,und+m,device='cuda'),vision_timesteps=torch.ones(m,device='cuda'),
                        vision_noisy_frame_indexes=[torch.tensor([0],device='cuda')])
                    reference.call=branch;live.begin_request('i2v');live.prepare_call(kwargs)
                    descriptor=describe_prefix(prefix_cpu(kwargs));key=digest(descriptor)
                    file=tmp_path/(key+'.pt')
                    torch.save(dict(model_identity={},descriptor=descriptor,
                        rows=[{k:v.cpu() for k,v in r.items()} for r in live.active]),file)
                    manifest['prefixes'][key]=dict(file=file.name,sha256=sha(file))
                    cases.append((branch,kwargs))
        write(tmp_path/'manifest.json',manifest)
        bank=PrefixBank(tmp_path,{})
        engine=JointEngine(cached,cfg,LaunchTable());offline=cache_class(ARCache,bank)(engine)
        engine.warming=True;engine.branches=2
        for branch,kwargs in cases+cases[:2]:
            reference.call=engine.call=branch
            live.begin_request('i2v');live.prepare_call(kwargs)
            offline.begin_request('i2v');offline.prepare_call(kwargs)
            assert len(bank.loaded)<=2
            for a,b in zip(live.active,offline.active):
                for key in a:assert torch.equal(a[key],b[key])
            assert torch.equal(reference.forward(**kwargs).sample[0],engine.forward(**kwargs).sample[0])
        # Capture A, replace the text with equal-shape prepared B, replay the
        # same graph. Cached AR buffers must update in place, not remain stale.
        branch,kwargs=cases[0];reference.call=engine.call=branch
        offline.begin_request('i2v');offline.prepare_call(kwargs)
        for _ in range(2):engine.forward(**kwargs)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):y=engine.forward(**kwargs).sample[0]
        kwargs['input_ids'].add_(3)
        offline.begin_request('i2v');offline.prepare_call(kwargs)
        live.begin_request('i2v');live.prepare_call(kwargs)
        graph.replay();torch.cuda.synchronize()
        assert torch.equal(y,reference.forward(**kwargs).sample[0])
