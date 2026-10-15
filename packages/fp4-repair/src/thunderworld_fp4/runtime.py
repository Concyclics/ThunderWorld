"""Validated SM120 attention boundary and bounded, current-input graph cache."""
from collections import OrderedDict
from threading import RLock
import math
import torch
import triton
from .config import RepairConfig, RepairPolicy
from ._kernels import pack_inputs, select_tiles, select_tiles_head, streaming_attention, combine


def _validate(q, k, v):
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
        raise TypeError("q, k, v must be tensors")
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("expected [1, sequence, heads, 128] Q/K/V")
    if q.shape[0] != 1 or k.shape[0] != 1 or k.shape != v.shape:
        raise ValueError("batch size must be 1 and K/V shapes must match")
    if any(t.shape[-1] != 128 or t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("head dimension must be 128 with contiguous last dimension")
    if min(q.shape[1:3]) <= 0 or min(k.shape[1:3]) <= 0 or q.shape[2] % k.shape[2]:
        raise ValueError("nonempty sequences/heads and an integer GQA ratio are required")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError("Q/K/V must be BF16 inputs; FP4 quantization is online")
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError("Q/K/V must be on the same CUDA device")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("this native FP4 backend is validated for SM120 only")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError("inference-only backend: use torch.inference_mode()")


def _forward(q, k, v, config, repair_mask=None, counts_tensor=None, return_mask=False):
    _validate(q, k, v)
    if not isinstance(config, RepairConfig):
        raise TypeError("config must be RepairConfig")
    m, n, h, hk = q.shape[1], k.shape[1], q.shape[2], k.shape[2]
    nt = triton.cdiv(n, 128)
    counts = config.head_counts(h, n)
    pv_counts = config.pv_counts(h,n)
    separate = config.pv_head_quotas is not None or config.pv_terms is not None or config.pv_coverage is not None or bool(config.pv_head_budgets)
    pv_terms = config.pv_terms or config.terms
    mask_shape = (h, triton.cdiv(m, 128), nt)
    if repair_mask is not None:
        if (repair_mask.shape != mask_shape or repair_mask.dtype != torch.bool
                or repair_mask.device != q.device or not repair_mask.is_contiguous()):
            raise ValueError(f"repair_mask must be contiguous CUDA bool {mask_shape}")
        # True selects repair; False still executes every dense FP4 base product.
        flags = torch.where(repair_mask, config.terms, 1).to(torch.int32)
        repair, full, selection = True, False, False
    else:
        repair, full = any(counts), all(c == nt for c in counts)
        selection = repair and not full
        flags = torch.empty(mask_shape, device=q.device, dtype=torch.int32) if selection else q
    pv_repair,pv_full = any(pv_counts),all(c==nt for c in pv_counts)
    if separate and repair_mask is not None:
        pv_flags=torch.where(repair_mask,pv_terms,1).to(torch.int32)
        pv_repair,pv_full=True,False
    elif separate:
        selection = (repair and not full) or (pv_repair and not pv_full)
        flags = torch.empty(mask_shape,device=q.device,dtype=torch.int32) if selection else q
        pv_flags = torch.empty(mask_shape,device=q.device,dtype=torch.int32) if selection else q
    else:pv_flags=flags
    bm = config.query_tile or (16 if m <= 128 else 32)
    splits = min(config.splits, nt)
    # Ceil-chunk splitting can create empty trailing chunks; do not launch them.
    chunk = math.ceil(nt / splits)
    splits = math.ceil(nt / chunk)
    packed_result = pack_inputs(q, k, v, repair or (separate and pv_repair), selection, max(config.terms,pv_terms) if separate else config.terms, config.transpose_v, config.sparse_pack)
    packed, mp, np = packed_result[:3]
    keepalive = [packed_result]
    if selection:
        uniform = len(set(counts)) == 1 and not separate
        if uniform:
            fn, quota = select_tiles, counts[0]
        else:
            if counts_tensor is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("nonuniform head budgets in outer capture require FP4Attention or an explicit repair_mask")
                counts_tensor = torch.tensor(counts, device=q.device, dtype=torch.int32)
                if separate:counts_tensor=(counts_tensor,torch.tensor(pv_counts,device=q.device,dtype=torch.int32))
            fn, quota = select_tiles_head, counts_tensor[0] if separate else counts_tensor
            keepalive.append(counts_tensor)
        fn[(h, triton.cdiv(m, 128))](q, k, packed_result[3], flags, m, n, h, hk,
            q.stride(1), q.stride(2), k.stride(1), k.stride(2), quota, config.terms,
            triton.next_power_of_2(nt), **(dict(PVFLAGS=pv_flags,PVCOUNTS=counts_tensor[1],SEPARATE=True,PVTERMS=pv_terms) if separate else {}),num_warps=4)
    if config.sparse_pack and (repair or pv_repair):
        from ._attention_sparse_pack import pack_selected
        unions=pack_selected(q,k,v,packed,mp,np,flags,pv_flags,qrepair=repair,qfull=full,
             prepair=pv_repair if separate else repair,pfull=pv_full if separate else full,
             qterms=config.terms,pterms=pv_terms,transpose=config.transpose_v)
        keepalive.append(unions)
    out = torch.empty(tuple(q.shape), device=q.device, dtype=q.dtype)
    parts = out if splits == 1 else torch.empty((splits, h, m, 128), device=q.device, dtype=torch.float32)
    stats = out if splits == 1 else torch.empty((splits, h, m, 2), device=q.device, dtype=torch.float32)
    streaming_attention[(triton.cdiv(m, bm), h, splits)](*packed, flags, parts, stats,
        m, n, h, hk, mp, np, bm, 128, repair, full, config.terms, splits,
        config.lazy_p, config.fast_p, config.transpose_v, config.reload_q or config.sparse_pack, PVFLAGS=pv_flags,SEPARATE=separate,PV_REPAIR=pv_repair,PV_FULL=pv_full,PV_TERMS=pv_terms,num_warps=config.num_warps, num_stages=1)
    if splits > 1:
        combine[(triton.cdiv(m, 4), h)](parts, stats, out, m, h, splits,
            triton.next_power_of_2(splits), num_warps=4, enable_fp_fusion=False)
    selected = None
    if return_mask:
        selected = (flags > 1) if selection or repair_mask is not None else torch.full(
            mask_shape, full, device=q.device, dtype=torch.bool)
    if return_mask and separate:
        pv_selected=(pv_flags>1) if selection or repair_mask is not None else torch.full(mask_shape,pv_full,device=q.device,dtype=torch.bool)
        selected=torch.stack((selected,pv_selected))
    return out, selected, keepalive


def fp4_attention(q, k, v, config=None, *, repair_mask=None, return_mask=False):
    """Noncausal, unmasked GQA joint attention; no edge deletion or old-value reuse.

    Input/output layout is [1, sequence, heads, 128]. An explicit boolean
    repair_mask overrides coverage/head quotas, but never masks attention edges.
    A returned repair mask may be reused deliberately by the caller; Q/K/V/P
    and the output are always recomputed. No causal/padding-mask fallback.
    """
    with torch.cuda.device(q.device) if isinstance(q, torch.Tensor) and q.is_cuda else _null_context():
        out, selected, _ = _forward(q, k, v, config or RepairConfig(), repair_mask, return_mask=return_mask)
    return (out, selected) if return_mask else out


from contextlib import nullcontext as _null_context


class FP4Attention:
    """Optional CUDA Graph cache, isolated by stream, shape, and resolved policy.

    Cache entries retain read-only quotas and owned input buffers. Every call
    copies current inputs/mask and returns owned output storage. Layer/step/CFG
    branch context is explicit; there is no global diffusers monkeypatch.
    """

    def __init__(self, config=None, *, policy=None, graph_cache_size=2):
        if config is not None and policy is not None:
            raise ValueError("choose config or policy")
        if type(graph_cache_size) is not int or graph_cache_size < 0:
            raise ValueError("graph_cache_size must be a nonnegative integer")
        self.policy = policy or RepairPolicy(default=config or RepairConfig())
        if not isinstance(self.policy, RepairPolicy):
            raise TypeError("policy must be RepairPolicy")
        self.graph_cache_size = graph_cache_size
        self._cache = OrderedDict()
        self._lock = RLock()

    def clear(self):
        with self._lock:
            # Synchronize before retiring captured storage (possibly other streams).
            for entry in self._cache.values():
                entry[5].synchronize()
            self._cache.clear()

    def __call__(self, q, k, v, *, layer=None, step=None, branch=None,
                 repair_mask=None, return_mask=False):
        config = self.policy.resolve(layer=layer, step=step, branch=branch)
        _validate(q, k, v)
        with torch.cuda.device(q.device):
            if not self.graph_cache_size or torch.cuda.is_current_stream_capturing():
                return fp4_attention(q, k, v, config, repair_mask=repair_mask, return_mask=return_mask)
            stream = torch.cuda.current_stream(q.device)
            key = (q.device, stream.cuda_stream, tuple(q.shape), tuple(k.shape), config,
                   repair_mask is not None, return_mask, torch.is_autocast_enabled("cuda"))
            with self._lock:
                # Validate masks on hits too, before copy_ can broadcast/coerce.
                if repair_mask is not None:
                    expected = (q.shape[2], triton.cdiv(q.shape[1], 128), triton.cdiv(k.shape[1], 128))
                    if repair_mask.dtype != torch.bool or repair_mask.device != q.device or tuple(repair_mask.shape) != expected or not repair_mask.is_contiguous():
                        raise ValueError(f"repair_mask must be contiguous CUDA bool {expected}")
                if key not in self._cache:
                    if len(self._cache) >= self.graph_cache_size:
                        _, previous = self._cache.popitem(last=False)
                        previous[5].synchronize()
                    buffers = tuple(t.contiguous().clone() for t in (q, k, v))
                    mask_buffer = repair_mask.clone() if repair_mask is not None else None
                    quotas = torch.tensor(config.head_counts(q.shape[2], k.shape[1]), device=q.device, dtype=torch.int32)
                    if config.pv_head_quotas is not None or config.pv_terms is not None or config.pv_coverage is not None or config.pv_head_budgets:
                        quotas=(quotas,torch.tensor(config.pv_counts(q.shape[2],k.shape[1]),device=q.device,dtype=torch.int32))
                    for _ in range(3):
                        _forward(*buffers, config, mask_buffer, quotas, return_mask)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output, selected, held = _forward(*buffers, config, mask_buffer, quotas, return_mask)
                    self._cache[key] = (buffers, mask_buffer, graph, output, selected, stream, quotas, held)
                else:
                    self._cache.move_to_end(key)
                buffers, mask_buffer, graph, output, selected, _, _, _ = self._cache[key]
                for source, target in zip((q, k, v), buffers):
                    target.copy_(source)
                if mask_buffer is not None:
                    mask_buffer.copy_(repair_mask)
                graph.replay()
                result = output.clone()
                return (result, selected.clone()) if return_mask else result
