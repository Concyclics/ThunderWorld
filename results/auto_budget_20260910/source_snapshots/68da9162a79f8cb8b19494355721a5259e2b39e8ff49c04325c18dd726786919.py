"""Independent FFN/projection A-only repair with immutable packed W4 buffers.

This portable Triton backend is new packaging work, not the historical CUTLASS
producer binary. Its local correctness is tested separately; previous full-model
quality/speed numbers must not be attributed to this backend without a new run.
"""
import torch
import triton
from .config import LinearRepairConfig
from ._linear_kernels import scaled_amax, finish_amax, activation_rank, pack_activation, linear_gemm


def _pack(x, prescale, repair, full, union):
    m, k = x.shape
    mp = triton.cdiv(m, 16) * 16
    partial = torch.empty((triton.cdiv(m*k, 4096),), device=x.device, dtype=torch.float32)
    factor = torch.empty((), device=x.device, dtype=torch.float32)
    scaled_amax[(partial.numel(),)](x, prescale, partial, m, k, 4096, num_warps=4)
    finish_amax[(1,)](partial, factor, partial.numel(), triton.next_power_of_2(partial.numel()), num_warps=4)
    planes = 2 if repair else 1
    codes = torch.empty((planes, mp, k//2), device=x.device, dtype=torch.uint8)
    scales = torch.empty((planes, mp, k//16), device=x.device, dtype=torch.float8_e4m3fn)
    pack_activation[(triton.cdiv(m,16), k//128)](x, prescale, factor, union,
        codes, scales, m, k, mp, repair, full, num_warps=4)
    return codes, scales, factor, mp


class FP4Linear(torch.nn.Module):
    """W4A4(+A1W4) for FFN or projection layers, with independent configuration.

    Packed format: low nibble first, uint8 [N,K/2], un-swizzled E4M3 [N,K/16],
    scalar weight global scale. Weight = decoded_codes * scale * global_scale.
    Forward computes BF16-rounded (x * prescale) before activation quantization.
    N/K must be multiples of 128. Explicit masks index M128/N128/K128 tiles.
    """

    def __init__(self, weight_codes, weight_scales, weight_global_scale, *,
                 prescale=None, bias=None, config=None):
        super().__init__()
        if weight_codes.ndim != 2 or weight_codes.dtype != torch.uint8:
            raise ValueError("weight_codes must be uint8 [N,K/2]")
        n, half = weight_codes.shape
        self.in_features, self.out_features = half*2, n
        config = config or LinearRepairConfig()
        if not isinstance(config, LinearRepairConfig):
            raise TypeError("config must be LinearRepairConfig")
        config.output_counts(n, half*2)
        if weight_scales.shape != (n, half*2//16) or weight_scales.dtype != torch.float8_e4m3fn:
            raise ValueError("weight_scales must be un-swizzled E4M3 [N,K/16]")
        device = weight_codes.device
        if weight_scales.device != device:
            raise ValueError("weight codes/scales must be on the same device")
        factor = torch.as_tensor(weight_global_scale, device=device, dtype=torch.float32)
        if factor.numel() != 1 or not bool(torch.isfinite(factor).all()) or not bool((factor > 0).all()):
            raise ValueError("weight_global_scale must be one finite positive value")
        if not bool(torch.isfinite(weight_scales.float()).all()) or bool((weight_scales.float() < 0).any()):
            raise ValueError("weight scales must be finite and nonnegative")
        if prescale is None:
            prescale = torch.ones(self.in_features, device=device, dtype=torch.float32)
        if prescale.shape != (self.in_features,) or not bool(torch.isfinite(prescale).all()) or not bool((prescale > 0).all()):
            raise ValueError("prescale must be a finite positive [K] vector")
        if bias is not None and (bias.shape != (n,) or not bool(torch.isfinite(bias).all())):
            raise ValueError("bias must be a finite [N] vector")
        self.register_buffer("weight_codes", weight_codes.detach().contiguous().clone())
        self.register_buffer("weight_scales", weight_scales.detach().contiguous().clone())
        self.register_buffer("weight_global_scale", factor.reshape(()).detach().clone())
        self.register_buffer("prescale", prescale.detach().to(device=device, dtype=torch.float32).contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().to(device=device, dtype=torch.bfloat16).contiguous())
        self._config = config
        counts = config.output_counts(n, half*2)
        self._repair, self._full = any(counts), all(c == self.in_features//128 for c in counts)
        self.register_buffer("_counts", torch.tensor(counts, device=device, dtype=torch.int32), persistent=False)

    @property
    def config(self):
        """Immutable configuration; construct another module to change quotas."""
        return self._config

    @classmethod
    @torch.no_grad()
    def from_float(cls, weight, *, prescale=None, bias=None, config=None):
        """Offline round-to-nearest W4 preparation; this is not AWQ calibration.

        Optional prescale performs W'=W/prescale offline. For pre-calibrated AWQ
        assets use the packed constructor directly, preserving its exact W4.
        """
        if weight.ndim != 2 or not weight.is_cuda or weight.dtype != torch.bfloat16:
            raise ValueError("offline weight must be BF16 CUDA [N,K]")
        if torch.cuda.get_device_capability(weight.device) != (12,0):
            raise RuntimeError("offline NVFP4 preparation requires SM120")
        n,k=weight.shape
        (config or LinearRepairConfig()).output_counts(n,k)
        d=torch.ones(k,device=weight.device) if prescale is None else prescale.to(weight.device).float()
        if d.shape!=(k,) or not bool(torch.isfinite(d).all()) or not bool((d>0).all()):
            raise ValueError("prescale must be positive finite [K]")
        transformed=(weight.float()/d).bfloat16().contiguous()
        ones=torch.ones_like(d)
        with torch.cuda.device(weight.device):
            codes,scales,factor,_=_pack(transformed,ones,False,False,transformed)
        return cls(codes[0,:n],scales[0,:n],factor,prescale=d,bias=bias,config=config)

    def forward(self, x, *, repair_mask=None):
        if x.ndim < 2 or x.shape[-1] != self.in_features or x.dtype != torch.bfloat16:
            raise ValueError("expected BF16 input [..., K]")
        if not x.is_cuda or x.device != self.weight_codes.device:
            raise ValueError("activation and packed weights must share a CUDA device")
        if torch.cuda.get_device_capability(x.device) != (12,0):
            raise RuntimeError("FP4Linear requires SM120")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("FP4Linear is inference-only; use torch.inference_mode()")
        with torch.cuda.device(x.device):
            flat=x.reshape(-1,self.in_features).contiguous();m,k=flat.shape;n=self.out_features
            if m==0:
                raise ValueError("empty activation rows are unsupported")
            shape=(triton.cdiv(m,128),n//128,k//128)
            if repair_mask is not None:
                if repair_mask.shape!=shape or repair_mask.dtype!=torch.bool or repair_mask.device!=x.device or not repair_mask.is_contiguous():
                    raise ValueError(f"repair_mask must be contiguous CUDA bool {shape}")
                mask=repair_mask;repair,full=True,False
            elif self._repair and not self._full:
                ranks=torch.empty((shape[0],shape[2]),device=x.device,dtype=torch.int32)
                activation_rank[(shape[0],)](flat,self.prescale,ranks,m,k,triton.next_power_of_2(k//128),num_warps=4)
                mask=(ranks[:,None,:]<self._counts[None,:,None]).contiguous()
                repair,full=True,False
            else:
                repair,full=self._repair,self._full;mask=flat
            union=mask.any(dim=1).contiguous() if repair and not full else flat
            ac,sc,af,mp=_pack(flat,self.prescale,repair,full,union)
            out=torch.empty((m,n),device=x.device,dtype=x.dtype)
            linear_gemm[(triton.cdiv(m,self.config.query_tile),n//128)](
                ac,sc,self.weight_codes,self.weight_scales,af,self.weight_global_scale,
                self.bias if self.bias is not None else flat,mask,out,m,n,k,mp,
                self.bias is not None,repair,full,self.config.query_tile,
                num_warps=self.config.num_warps,num_stages=1)
            return out.reshape(*x.shape[:-1],n)
