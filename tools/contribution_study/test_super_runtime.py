import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

import super_runtime as runtime


def test_placement_never_moves_generation_projections_to_ar_device():
    assert runtime.is_ar('layers.63.mlp.gate_proj.weight')
    assert runtime.is_ar('layers.63.input_layernorm')
    assert runtime.is_ar('layers.1.self_attn.to_k.weight')
    assert runtime.is_ar('layers.1.self_attn.k_norm_und_for_gen')
    for name in ('layers.63.mlp_moe_gen.gate_proj.weight',
                 'layers.63.input_layernorm_moe_gen',
                 'layers.0.self_attn.add_k_proj', 'embed_tokens', 'norm', 'norm_moe_gen'):
        assert not runtime.is_ar(name)


def test_device_pair_is_explicit_and_allowlisted(monkeypatch):
    monkeypatch.setenv('TW_SHARED_CALIBRATION_GPUS', '0,2')
    for invalid in ('', '0', '0,0', '0,1', '0,2,3'):
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', invalid)
        with pytest.raises(ValueError):
            runtime.physical_pair()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,2')
    assert runtime.physical_pair() == (0, 2)


def test_pair_lock_excludes_a_second_worker(monkeypatch, tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,2')
    monkeypatch.setenv('TW_SHARED_CALIBRATION_GPUS', '0,2')
    lock = tmp_path / 'worker.lock'
    with runtime.pair_worker(lock):
        with pytest.raises(RuntimeError, match='owns'):
            with runtime.pair_worker(lock):
                pass
    with runtime.pair_worker(lock):
        pass


def test_two_device_ar_and_dm_match_single_device_current_inputs():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        pytest.skip('requires an explicitly isolated pair of identical GPUs')
    from diffusers import Cosmos3OmniTransformer
    from thunderworld_fp4.cosmos.ar_cache import ARCache as OriginalCache
    from thunderworld_fp4.cosmos.engine import JointEngine
    from thunderworld_fp4.cosmos.settings import RunConfig
    from thunderworld_fp4.cosmos.tuning import LaunchTable
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(928)
    base = Cosmos3OmniTransformer(hidden_size=256, intermediate_size=512,
        num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
        head_dim=128, vocab_size=1024, use_und_k_norm_for_gen=True).bfloat16().eval()
    one = copy.deepcopy(base).to('cuda:0')
    two = copy.deepcopy(base)
    # Deliberately exercise the actual placement function used by large models.
    pipe = SimpleNamespace(transformer=two, components={'transformer': two})
    runtime.place_pipeline(pipe)
    one.set_attention_backend('_native_flash')
    two.set_attention_backend('_native_flash')
    cfg = RunConfig(precision='bf16', attention_backend='bf16', execution='eager',
                    resource_mode='shared-calibration')
    engine0 = JointEngine(one, cfg, LaunchTable())
    cache0 = OriginalCache(engine0)
    runtime.activate()
    from thunderworld_fp4.cosmos.ar_cache import ARCache as SplitCache
    engine1 = JointEngine(two, cfg, LaunchTable())
    cache1 = SplitCache(engine1)
    for engine in (engine0, engine1):
        engine.warming = True
        engine.branches = 2
        engine.call = 0
    with torch.inference_mode():
        for m in (129, 416):
            und = 17
            kwargs = dict(input_ids=torch.arange(und, device='cuda'),
                text_indexes=torch.arange(und, device='cuda'),
                position_ids=torch.arange(und + m, device='cuda'), und_len=und,
                sequence_length=und + m,
                vision_tokens=[torch.randn(1, 48, 1, 2, 2*m, device='cuda', dtype=torch.bfloat16)],
                vision_token_shapes=[(1, 1, m)],
                vision_sequence_indexes=torch.arange(und, und + m, device='cuda'),
                vision_mse_loss_indexes=torch.arange(und, und + m, device='cuda'),
                vision_timesteps=torch.ones(m, device='cuda'),
                vision_noisy_frame_indexes=[torch.tensor([0], device='cuda')])
            for branch in (0, 1):
                for engine in (engine0, engine1):
                    engine.call = branch
                previous = None
                for changed in (False, True, False):
                    kwargs['input_ids'] = torch.arange(und, device='cuda') + int(changed)
                    for cache in (cache0, cache1):
                        cache.begin_request('i2v')
                        cache.prepare_call(kwargs)
                    for a, b in zip(cache0.active, cache1.active):
                        for key in a:
                            assert torch.equal(a[key], b[key]), (m, branch, changed, key)
                    out0 = engine0.forward(**kwargs).sample[0]
                    out1 = engine1.forward(**kwargs).sample[0]
                    assert torch.equal(out0, out1), (m, branch, changed)
                    if previous is not None and changed:
                        assert not torch.equal(previous, out1), 'stale request input'
                    previous = out1.clone()
            # A captured DM forward only consumes cache buffers on CUDA0.
            static_output = None
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = engine1.forward(**kwargs).sample[0]
            kwargs['input_ids'].add_(3)
            cache1.begin_request('i2v')
            cache1.prepare_call(kwargs)
            graph.replay()
            result = static_output.clone()
            expected = engine1.forward(**kwargs).sample[0]
            assert torch.equal(result, expected), 'DM graph read stale remote AR cache'


def test_layer_local_teacher_interventions_preserve_fp4_arithmetic():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        pytest.skip('requires isolated SM120 GPUs')
    from thunderworld_fp4.cosmos.engine import RepairLinear as ResidentLinear
    from thunderworld_fp4.cosmos.tuning import LaunchTable
    from thunderworld_fp4.cosmos.settings import LinearSpec
    runtime.activate()
    from thunderworld_fp4.cosmos.linears import RepairLinear as LocalLinear
    source = torch.nn.Linear(512, 256, bias=False, device='cuda', dtype=torch.bfloat16)
    spec = LinearSpec(1, 1)
    controller = SimpleNamespace(warming=True, table=LaunchTable(),
        linear_spec=lambda *args: spec, record_linear=lambda *args: None)
    with torch.inference_mode():
        resident = ResidentLinear(source, controller, 'test', spec)
        local = LocalLinear(source, controller, 'test', spec, resident.asset)
        for m in (129, 416):
            for _ in range(2):
                x = torch.randn(m, 512, device='cuda', dtype=torch.bfloat16)
                assert torch.equal(resident(x), local(x))
                assert not local.ops and local.bound is None
