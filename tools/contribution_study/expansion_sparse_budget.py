"""Exact executed padded MMA counts for the constant-quota sparse ablation."""


def linear_cost(m,n,k,numerator,denominator=4):
    mt=(m+127)//128;nt=n//128;kt=k//128
    if min(m,n,k)<=0 or n%128 or k%128:raise ValueError('Invalid linear shape')
    quota=kt*numerator//denominator
    unit=2*128**3
    base=mt*nt*kt*unit
    repair=mt*nt*quota*unit
    mp=(m+15)//16*16
    return dict(kind='linear',shape=[m,n,k],tile=[128,128,128],base_mma=base,
        a_mma=repair,w_mma=repair,cross_mma=0,selector_mma=0,total_mma=base+2*repair,
        quota_per_side=quota,available_blocks=kt,ratio=(base+2*repair)/base,
        activation_pack_write_bytes=2*mp*k*9//16,
        selector='represented A1 energy times fixed W0 K-block energy; offline W1 block energy',
        full_residual_packing=True,compressed_weight_storage=False)


def attention_cost(m,n,h,hk,numerator,denominator=4):
    if min(m,n,h,hk)<=0 or h%hk:raise ValueError('Invalid attention shape')
    mt=(m+31)//32;nt=(n+127)//128;quota=nt*numerator//denominator
    # QK and PV each execute 32x128x128 per query/key tile, including tails.
    unit=2*32*128*128
    base=h*mt*nt*unit
    repair=h*mt*quota*unit
    mp=(m+15)//16*16;npad=nt*128
    return dict(kind='attention',shape=[m,n,h,hk],tile=[32,128,128],base_mma=2*base,
        qk_base_mma=base,pv_base_mma=base,qk_a_mma=repair,qk_w_mma=repair,
        pv_a_mma=repair,pv_w_mma=repair,cross_mma=0,selector_mma=0,
        a_mma=2*repair,w_mma=2*repair,total_mma=2*base+4*repair,
        quota_per_side_per_qk_pv=quota,available_blocks=nt,ratio=(base+2*repair)/base,
        qkv_pack_write_bytes=2*(h*mp+2*hk*npad)*72,
        full_residual_packing=True,gqa_packed_union_density=1.,
        selector='Current represented norm products for QK; V0/V1 norm proxies for PV')


def compile_request(rows,budget,steps=35,branches=2):
    if not rows or budget not in (1.5,2.):raise ValueError('Invalid sparse request')
    if {r['branch'] for r in rows}!=set(range(branches)):raise ValueError('Missing CFG branch')
    base=sum(r['base_mma'] for r in rows)*steps
    total=sum(r['total_mma'] for r in rows)*steps
    if total>base*budget:raise ValueError('Sparse execution exceeds compute budget')
    return dict(schema_version=1,budget=budget,steps=steps,branches=branches,
        base_mma=base,total_mma=total,ratio=total/base,feasible=True,
        mma_by_dtype={'native_nvfp4':total},selector_mma=0,cross_mma=0,
        counting='Scalar FLOP-equivalent of actually executed padded MMA tiles; not latency',
        scope='All generation projections/FFN and noncausal QK/PV, both CFG branches; '
              'common BF16 AR and non-MMA operators outside C0',
        support_bound='Exactly floor(available_blocks * fraction) per operand; '
                      'valid for any current-input support',
        non_mma='Full two-plane packing, residual-energy reductions, stable sorting, softmax, '
                'P quantization, split combine and unchanged operators remain',
        phases=[0,7,18,28,35],phase_policy='Same quota and selector in every phase',
        performance_eligible=False,rows=rows)
