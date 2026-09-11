"""GPU integration checks for both FFN activations and the attention boundary."""
import pytest
import torch
import diffusers.utils.import_utils as import_utils

import_utils._xformers_available = False
from diffusers.models.transformers import transformer_cosmos3 as cosmos
from adapter import Controller, PreparedLinear, install_attention_dispatch
from graph import GraphForward
from types import SimpleNamespace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='SM120 CUDA required')


@pytest.mark.parametrize('activation', ['relu2', 'silu'])
@torch.inference_mode()
def test_ffn_roundtrip(activation):
    torch.manual_seed(5090)
    controller = Controller()
    mlp = cosmos.Cosmos3VLTextMLP(256, 384, activation).bfloat16()
    assert hasattr(mlp, 'gate_proj') == (activation == 'silu')
    for name, child in list(mlp.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(mlp, name, PreparedLinear(child, controller, name))
    mlp.cuda()
    x = torch.randn((129, 256), device='cuda', dtype=torch.bfloat16)
    controller.mode = 'ffn_original'
    original = mlp(x)
    controller.mode = 'ffn_optimized'
    optimized = mlp(x)
    assert (original.float()-optimized.float()).norm()/original.float().norm() < .004
    mlp.cpu().cuda()
    assert torch.equal(mlp(x), optimized)
    changed = mlp(x.mul(.7).add(.2))
    assert not torch.equal(changed, optimized)


@torch.inference_mode()
def test_attention_boundary(monkeypatch):
    controller = Controller()
    original = cosmos.dispatch_attention_fn
    # Register restoration before the adapter changes the module global.
    monkeypatch.setattr(cosmos, 'dispatch_attention_fn', original)
    install_attention_dispatch(controller)
    q = torch.randn((1, 33, 16, 128), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((1, 65, 8, 128), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    controller.mode = 'joint_original'
    a = cosmos.dispatch_attention_fn(q, k, v, is_causal=False, enable_gqa=True)
    controller.mode = 'joint_optimized'
    b = cosmos.dispatch_attention_fn(q, k, v, is_causal=False, enable_gqa=True)
    assert torch.equal(a, b)
    causal_kwargs = dict(is_causal=True, enable_gqa=True, backend='_native_flash')
    assert torch.equal(cosmos.dispatch_attention_fn(q, k[:, :33], v[:, :33], **causal_kwargs),
                       original(q, k[:, :33], v[:, :33], **causal_kwargs))
    with pytest.raises(ValueError, match='unmasked'):
        cosmos.dispatch_attention_fn(q, k, v, is_causal=False, attn_mask=torch.ones(33, 65, device='cuda'))


@torch.inference_mode()
def test_transformer_graph_current_input_and_owned_outputs():
    controller = SimpleNamespace(mode='bf16', calls={'ffn': 0, 'attention': 0}, tuning=True)
    w = torch.randn((128, 128), device='cuda', dtype=torch.bfloat16)
    def forward(tokens, timestep, scale):
        controller.calls['ffn'] += 1
        return ([tokens[0] @ w + timestep * scale], None, None)
    graph = GraphForward(forward, controller)
    x = torch.randn((33, 128), device='cuda', dtype=torch.bfloat16)
    t = torch.ones((), device='cuda', dtype=torch.bfloat16)
    old = graph(tokens=[x], timestep=t, scale=1)[0][0]
    saved = old.clone()
    assert controller.calls['ffn'] == 1
    x.mul_(.7)
    t.fill_(3)
    controller.tuning = False
    result = graph(tokens=[x], timestep=t, scale=1)[0][0]
    assert controller.calls['ffn'] == 2
    assert torch.equal(result, forward([x], t, 1)[0][0])
    assert torch.equal(old, saved) and not torch.equal(old, result)
    with pytest.raises(RuntimeError, match='Unwarmed'):
        graph(tokens=[x], timestep=t, scale=2)
    graph.clear()
