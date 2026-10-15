"""Optional exact pipeline shortcuts for the runner's single-image I2V path.

Install before graph warmup. All caches are instance owned; model weights and
pipeline topology must remain fixed. No temporal DM outputs are reused.
"""
import torch
import triton
import triton.language as tl
from .trees import signature


class FirstFrameConditioning:
    def __init__(self, pipe):
        self.enabled = False
        self.calls = 0
        self.original = pipe._encode_video
        self.ratio = int(pipe.vae.config.scale_factor_temporal)
        if self.ratio != 4 or type(pipe.vae).__name__ != 'AutoencoderKLWan':
            raise ValueError('First-frame shortcut requires the causal Wan codec with temporal ratio 4')
        pipe._encode_video = self.encode

    def begin_request(self, case):
        if case['task'] != 'i2v' or not case.get('image') or case.get('video'):
            raise ValueError('First-frame shortcut requires the runner single-image I2V flow')
        self.enabled = True

    def encode(self, x):
        if not self.enabled or x.ndim != 5 or x.shape[0] != 1 or (x.shape[2]-1) % self.ratio:
            raise ValueError('Unsupported first-frame conditioning request')
        first = self.original(x[:, :, :1].contiguous())
        if first.shape[2] != 1: raise RuntimeError('Unexpected causal first latent shape')
        out = first.new_zeros((*first.shape[:2], (x.shape[2]-1)//self.ratio+1, *first.shape[3:]))
        out[:, :, :1] = first
        self.calls += 1
        return out

    def receipt(self):
        return dict(enabled=True, calls=self.calls, scope='Encode first conditioning frame; preserve full latent/noise shape and RNG order; single-image I2V only')


@triton.jit
def rope_kernel(X, C, S, Y, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    d = i % D
    pair = i - d + (d + D//2) % D
    x = tl.load(X+i, i<N, 0).to(tl.float32)
    r = tl.load(X+pair, i<N, 0).to(tl.float32)
    r = tl.where(d < D//2, -r, r)
    j = i//(H*D)*D + d
    c = tl.load(C+j, i<N, 0).to(tl.float32)
    s = tl.load(S+j, i<N, 0).to(tl.float32)
    # Reference PyTorch materializes both BF16 products before BF16 addition.
    p = (x*c).to(tl.bfloat16).to(tl.float32)
    q = (r*s).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i, p+q, i<N)


def fused_rope(x, cos, sin):
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError('RoPE fusion requires contiguous BF16 tensors')
    y = torch.empty_like(x)
    rope_kernel[(triton.cdiv(x.numel(),512),)](x,cos,sin,y,x.numel(),x.shape[-2],x.shape[-1],512,enable_fp_fusion=False)
    return y


@triton.jit
def relu2_kernel(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.maximum(tl.load(X+i,i<N,0).to(tl.float32),0.)
    tl.store(Y+i,x*x,i<N)


def relu2(x):
    if x.dtype != torch.bfloat16 or not x.is_contiguous(): raise ValueError('Expected contiguous BF16 activation')
    out=torch.empty_like(x)
    relu2_kernel[(triton.cdiv(x.numel(),512),)](x,out,x.numel(),512,enable_fp_fusion=False)
    return out


def install_relu2(model):
    names=[]
    for name,module in model.named_modules():
        if name.endswith('mlp_moe_gen') and module.hidden_act=='relu2':
            def forward(x,_m=module): return _m.down_proj(relu2(_m.up_proj(x)))
            module.forward=forward;names.append(name)
    return names


class StaticRotary:
    """Compute full position tables once per request/CFG branch, outside DM graphs.

    Current position IDs always refresh the stable captured buffers. Only the
    static-position runner I2V contract permits reuse within a request.
    """
    def __init__(self, engine):
        self.engine=engine;self.model=engine.model;self.original=self.model.rotary_emb.forward
        self.slots={};self.seen=set();self.active=None;self.refreshes=0
        self.model.rotary_emb.forward=self.forward

    def begin_request(self,task):
        if task!='i2v':raise ValueError('Static rotary currently supports runner I2V only')
        self.seen.clear()

    def prepare_call(self,kw):
        pos=kw['position_ids'];key=(self.engine.call%self.engine.branches,signature(pos),torch.cuda.current_stream().cuda_stream)
        if key not in self.seen:
            if key not in self.slots and not self.engine.warming: raise RuntimeError('Unwarmed rotary shape')
            if key not in self.slots and len(self.slots)>=16:raise RuntimeError('Rotary cache capacity exceeded')
            value=self.original(position_ids=pos.unsqueeze(0) if pos.ndim==1 else pos.unsqueeze(1),device=pos.device,dtype=self.model.embed_tokens.weight.dtype)
            if key not in self.slots:self.slots[key]=tuple(v.clone() for v in value)
            else:
                for dst,src in zip(self.slots[key],value):dst.copy_(src)
            self.seen.add(key);self.refreshes+=1
        self.active=self.slots[key]

    def forward(self,position_ids,device,dtype):
        if self.active is None:raise RuntimeError('Prepare rotary before transformer/prefill')
        if self.active[0].device!=device or self.active[0].dtype!=dtype:raise ValueError('Rotary dtype/device mismatch')
        return self.active

    def receipt(self):
        return dict(enabled=True,slots=len(self.slots),refreshes=self.refreshes,scope='Current full position table per request and CFG branch; refresh included in latency')
