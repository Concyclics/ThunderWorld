"""CPU-only issued-MMA ledger. Counts FLOPs (2 per MAC), including padded tiles.

The bound concerns arithmetic, not latency. Non-MMA work and storage are separate.
Online A supports may change arbitrarily: cross terms use a provable intersection
upper bound unless an explicit support is supplied for a correctness audit.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

PHASES = (0, 7, 18, 28, 35)
BUDGETS = (1., 1.125, 1.25, 1.5, 2., 2.5)


def ceildiv(a, b):
    return (a + b - 1) // b


def integer(x, name, maximum=None):
    if type(x) is not int or x < 0 or (maximum is not None and x > maximum):
        raise ValueError(f'Invalid {name}: {x}')
    return x


def support(ids, size):
    result = tuple(ids)
    if len(set(result)) != len(result):
        raise ValueError('Repeated support block')
    for i in result:
        integer(i, 'block index', size - 1)
    return result


def linear_cost(m, n, k, *, bm=128, a_blocks=0, w_blocks=(), cross=False, a_support=None):
    for v in (m, n, k):
        if type(v) is not int or v < 1:
            raise ValueError('Positive integer dimensions required')
    if n % 128 or k % 128 or bm not in (32, 64, 128):
        raise ValueError('Unsupported execution tile/shape')
    nb, kb, mt = n // 128, k // 128, ceildiv(m, 128)
    a_blocks = integer(a_blocks, 'A quota', kb)
    w_blocks = support(w_blocks, nb * kb)
    if type(cross) is not bool:
        raise ValueError('cross must be boolean')
    counts = Counter(w % kb for w in w_blocks)
    # One online A set is shared over all N tiles. Sum the largest K frequencies.
    cross_per_m = sum(sorted(counts.values(), reverse=True)[:a_blocks]) if cross else 0
    if a_support is not None:
        if len(a_support) != mt:
            raise ValueError('A support must cover every M128 block')
        for ids in a_support:
            support(ids, kb)
            if len(ids) > a_blocks:
                raise ValueError('A support violates quota')
            if a_blocks == kb and len(ids) != kb:
                raise ValueError('Full A quota executes unconditional full repair')
    ctas = ceildiv(m, bm)
    cross_tiles = 0
    a_tiles = 0
    for row in range(ctas):
        a_tiles += len(a_support[row * bm // 128]) if a_support is not None else a_blocks
        cross_tiles += (sum(counts[i] for i in a_support[row * bm // 128]) if cross else 0) if a_support is not None else cross_per_m
    tile = 2 * bm * 128 * 128
    terms = dict(base=ctas * nb * kb * tile, a=a_tiles * nb * tile,
                 w=ctas * len(w_blocks) * tile, cross=cross_tiles * tile)
    mp = ceildiv(m, 16) * 16
    packed_base = mp * k * 9 // 16  # E2M1 codes + E4M3 scales
    # Sparse pack implementation writes residuals only for selected 128x128 A tiles.
    residual_rows = sum(min(128, mp - i * 128) for i in range(mt))
    selected_rows = (sum(min(128,mp-i*128)*len(ids) for i,ids in enumerate(a_support))
                     if a_support is not None else residual_rows*a_blocks)
    selected_bytes = selected_rows * 128 * 9 // 16
    return dict(kind='linear', shape=[m, n, k], launch=dict(bm=bm, bn=128, bk=128),
                mma_by_dtype={'fp4': terms, 'bf16': {}}, base_mma=terms['base'],
                total_mma=sum(terms.values()), a_blocks=a_blocks, w_blocks=list(w_blocks),
                cross_bound=a_support is None, padded_base_mma=terms['base'] - 2*m*n*k,
                packing=dict(base_write_bytes=packed_base, selected_residual_write_bytes=selected_bytes,
                             residual_allocation_bytes=packed_base if a_blocks else 0,
                             input_read_bytes_minimum=2*m*k),
                non_mma=dict(amax_elements=m*k, quantized_base_elements=mp*k,
                             selected_residual_elements=selected_rows*128,
                             selector_rank_pairs=mt*(1<<(kb-1).bit_length())**2 if 0<a_blocks<kb else 0),
                w1_union_bytes=len(w_blocks)*(128*128*9//16),
                cost_scope='issued MMA including tail padding; packing model must match recorded implementation')


def attention_cost(m, n, h, hk, *, bm=32, qk_counts=None, pv_counts=None,
                   qk_terms=3, pv_terms=3, selector=True, explicit_masks=None):
    if any(type(x) is not int or x < 1 for x in (m,n,h,hk)) or h % hk or bm not in (16,32,64):
        raise ValueError('Invalid attention shape/tile/GQA')
    nt, mt = ceildiv(n,128), ceildiv(m,128)
    qk_counts = tuple((0,)*h if qk_counts is None else qk_counts)
    pv_counts = tuple((0,)*h if pv_counts is None else pv_counts)
    if len(qk_counts)!=h or len(pv_counts)!=h or qk_terms not in (2,3,4) or pv_terms not in (2,3,4):
        raise ValueError('Invalid head quotas/terms')
    for x in qk_counts+pv_counts:integer(x,'attention quota',nt)
    ctas=ceildiv(m,bm);tile=2*bm*128*128
    base_one=ctas*h*nt*tile
    terms=dict(qk_base=base_one,pv_base=base_one)
    for prefix,counts,t in (('qk',qk_counts,qk_terms),('pv',pv_counts,pv_terms)):
        selected=ctas*sum(counts)
        if explicit_masks is not None:
            mask=explicit_masks[prefix]
            if len(mask)!=h or any(len(rows)!=mt for rows in mask):raise ValueError('Mask shape mismatch')
            for head in mask:
                for row in head:
                    if len(row)!=nt or any(type(v) is not bool for v in row):raise ValueError('Mask shape/dtype mismatch')
            selected=sum(sum(mask[head][r*bm//128]) for head in range(h) for r in range(ctas))
        terms[prefix+'_a']=selected*tile
        terms[prefix+'_w']=selected*tile if t>=3 else 0
        terms[prefix+'_cross']=selected*tile if t==4 else 0
        if prefix=='pv':pv_selected=selected
    # Existing attention selector issues 16 x (next_power_of_two(N128)*8) x 128 BF16.
    selection=selector and explicit_masks is None and ((any(qk_counts) and not all(c==nt for c in qk_counts)) or (any(pv_counts) and not all(c==nt for c in pv_counts)))
    bt=1<<(nt-1).bit_length()
    selector_mma=h*mt*2*16*(bt*8)*128 if selection else 0
    # Arbitrary independent head/M-block choices may cover all K/V blocks under GQA.
    union=[]
    for group in range(hk):
        heads=range(group*(h//hk),(group+1)*(h//hk))
        union.append(dict(k_blocks_upper=min(nt,mt*sum(qk_counts[i] for i in heads)) if qk_terms>=3 else 0,
                          v_blocks_upper=min(nt,mt*sum(pv_counts[i] for i in heads)) if pv_terms>=3 else 0))
    mp=ceildiv(m,16)*16;npad=nt*128
    q_rows=mp*sum(c>0 for c in qk_counts)
    if explicit_masks is not None:
        q_rows=sum(min(128,mp-row*128) for head in explicit_masks['qk'] for row,mask in enumerate(head) if any(mask))
        for group in range(hk):
            heads=range(group*(h//hk),(group+1)*(h//hk))
            for side,term_count,key in (('qk',qk_terms,'k_blocks_upper'),('pv',pv_terms,'v_blocks_upper')):
                union[group][key]=sum(any(explicit_masks[side][head][row][ki] for head in heads for row in range(mt)) for ki in range(nt)) if term_count>=3 else 0
    k_tiles=sum(u['k_blocks_upper'] for u in union);v_tiles=sum(u['v_blocks_upper'] for u in union)
    packed_base=(h*mp+2*hk*npad)*72
    return dict(kind='attention',shape=[m,n,h,hk,128],launch=dict(bm=bm,bn=128),
        mma_by_dtype={'fp4':terms,'bf16':{'selector':selector_mma}},base_mma=2*base_one,
        total_mma=sum(terms.values())+selector_mma,qk_counts=list(qk_counts),pv_counts=list(pv_counts),
        packing=dict(base_code_scale_write_bytes=packed_base,base_factor_write_bytes=(h*mp+2*hk*npad)*4,
                     selected_residual_write_bytes_upper=(q_rows+(k_tiles+v_tiles)*128)*72,
                     input_read_bytes_minimum=(h*m+2*hk*n)*128*2,
                     residual_allocation_bytes_upper=(h*mp+2*hk*npad)*72 if explicit_masks is not None or any(qk_counts+pv_counts) else 0),
        gqa_support_union_upper=union,gqa_union_scope='Exact provided mask union' if explicit_masks is not None else 'Upper bound, not measured packing savings',
        padded_base_mma=2*base_one-4*m*n*h*128,
        non_mma=dict(softmax_elements=ctas*bm*nt*128*h,
                     probability_base_quantized_elements=ctas*bm*nt*128*h,
                     probability_residual_quantized_elements=pv_selected*bm*128,
                     qkv_base_quantized_elements=(h*mp+2*hk*npad)*128,
                     qkv_residual_quantized_elements_upper=(q_rows+(k_tiles+v_tiles)*128)*128,
                     combine_output_elements=m*h*128,
                     selector_rank_pairs=h*mt*bt*bt if selection else 0))


class RepairPlan:
    """Exact scope assignments; no implicit fallback or wildcard budget holes."""
    def __init__(self, value):
        self.value=value
        if value.get('schema_version')!=1 or value.get('steps')!=35 or value.get('phases')!=list(PHASES):
            raise ValueError('Invalid compute plan schema/steps/phases')
        if type(value.get('budget')) not in (int,float) or value.get('budget') not in BUDGETS:raise ValueError('Unsupported budget')
        if not value.get('task'):raise ValueError('Plan task required')
        self.linears=value['linears'];self.attention=value['attention']

    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text()))

    @property
    def fingerprint(self):return hashlib.sha256(json.dumps(self.value,sort_keys=True).encode()).hexdigest()

    def linear(self, task, branch, phase, name):
        from .settings import LinearSpec
        if task!=self.value['task']:raise ValueError('Budget task mismatch')
        row=self.linears[f'{branch}|{phase}|{name}']
        return LinearSpec(0,0,row.get('cross',False),activation_blocks=row['a_blocks'],
                          weight_support=tuple(row['w_blocks']))

    def attention_config(self, task, branch, phase, layer, default):
        from dataclasses import replace
        if task!=self.value['task']:raise ValueError('Budget task mismatch')
        row=self.attention[f'{branch}|{phase}|{layer}']
        if 'qk_fractions' in row or 'pv_fractions' in row:
            if set(row) & {'qk_counts','pv_counts','full_attention'}:
                raise ValueError('Fractional and fixed/full attention quotas cannot be mixed')
            qk, pv = row['qk_fractions'], row['pv_fractions']
            if not qk or len(qk)!=len(pv):raise ValueError('Both head fraction lists are required')
            return replace(default,coverage=0.,head_budgets=tuple(enumerate(qk)),
                pv_coverage=0.,pv_head_budgets=tuple(enumerate(pv)),head_quotas=None,pv_head_quotas=None,
                sparse_pack=True,terms=row.get('qk_terms',3),pv_terms=row.get('pv_terms',3))
        # Full repair must include the current prompt's tail tile as well. The
        # ordinary warmup ledger checks its actual shape before Graph capture.
        if row.get('full_attention',False):
            if row.get('qk_counts') or row.get('pv_counts'):
                raise ValueError('Full attention cannot also specify fixed quotas')
            return replace(default,coverage=1.,head_budgets=(),sparse_pack=True,
                head_quotas=None,pv_head_quotas=None,pv_coverage=None,pv_head_budgets=(),terms=row.get('qk_terms',3),pv_terms=row.get('pv_terms',3))
        return replace(default,coverage=0.,head_budgets=(),sparse_pack=True,head_quotas=tuple(row['qk_counts']),
                       pv_head_quotas=tuple(row['pv_counts']),pv_coverage=None,pv_head_budgets=(),terms=row.get('qk_terms',3),
                       pv_terms=row.get('pv_terms',3))


def compile_request(rows,budget,*,steps=35,branches=2,expected_modules=None,expected_layers=None):
    """Rows are observed once per actual operator/phase/branch, no sampled extrapolation."""
    if type(budget) not in (int,float) or budget not in BUDGETS or steps!=35 or branches not in (1,2):raise ValueError('Invalid request protocol')
    seen=set();base=total=0;by_dtype=Counter();expanded=[]
    groups={}
    for row in rows:
        integer(row['base_mma'],'base MMA');integer(row['total_mma'],'total MMA')
        if row['base_mma']<=0 or row['total_mma']<row['base_mma']:raise ValueError('Invalid row MMA totals')
        for terms in row['mma_by_dtype'].values():
            for value in terms.values():integer(value,'MMA term')
        if sum(sum(t.values()) for t in row['mma_by_dtype'].values())!=row['total_mma']:raise ValueError('MMA subtotal mismatch')
        branch,ph=row['branch'],row['phase'];integer(branch,'branch',branches-1);integer(ph,'phase',3)
        key=(row['kind'],row['name'],branch,ph)
        if key in seen:raise ValueError('Duplicate operator execution row')
        seen.add(key)
        groups.setdefault((branch,ph,row['kind']),set()).add(row['name'])
        count=PHASES[ph+1]-PHASES[ph]
        base+=row['base_mma']*count;total+=row['total_mma']*count
        for dtype,terms in row['mma_by_dtype'].items():by_dtype[dtype]+=sum(terms.values())*count
        expanded.append({**row,'calls':count})
    for branch in range(branches):
        for ph in range(4):
            for kind,expected in (('linear',expected_modules),('attention',expected_layers)):
                actual=groups.get((branch,ph,kind),set())
                if expected is not None and actual!=set(expected):raise ValueError(f'Incomplete {kind} inventory at {branch}/{ph}')
                if not actual:raise ValueError('Empty execution inventory')
    if base<=0:raise ValueError('Empty base computation')
    cap=Fraction(str(budget))
    feasible=total*cap.denominator<=base*cap.numerator
    non_mma=Counter();packing=Counter();w_unions={}
    for row in expanded:
        for key,value in row.get('non_mma',{}).items():
            if type(value) is int:non_mma[key]+=value*row['calls']
        for key,value in row.get('packing',{}).items():
            if type(value) is int and 'allocation' not in key:packing[key]+=value*row['calls']
        if row['kind']=='linear':w_unions.setdefault(row['name'],set()).update(row['w_blocks'])
    result=dict(non_mma_work=dict(non_mma),packing_traffic=dict(packing),
                w1_support_union_bytes=sum(len(ids)*128*128*9//16 for ids in w_unions.values()),
                w1_support_union_blocks={name:len(ids) for name,ids in w_unions.items()},
                schema_version=1,budget=budget,steps=steps,branches=branches,base_mma=base,total_mma=total,
                ratio=total/base,feasible=feasible,mma_by_dtype=dict(by_dtype),rows=expanded,
                scope='Whole request generated projections + attention, all issued MMA FLOPs; not a latency bound')
    if not feasible:raise ValueError(f'Compute budget exceeded: {total}/{base} = {total/base:.9f} > {budget}')
    return result
