"""Frozen same-bank repair choices and shape-safe request accounting."""
import copy
import math
from dataclasses import replace

from thunderworld_fp4 import RepairConfig
from thunderworld_fp4.cosmos.budget import RepairPlan, PHASES, linear_cost, attention_cost, compile_request

CONTROLS = {1.5: 'b1.5_protected_n', 2.: 'b2_joint', 2.5: 'b2.5_ffn_guarded'}
KEY_TILES = tuple(range(8, 17))


def variants(row, k, ranked):
    a = row['a_blocks']; jump = max(1, k//128//8)
    result = {'base': copy.deepcopy(row)}
    for label, count in [('a_low', max(k//128//8, a-jump)), ('a_high', min(k//128, a+jump))]:
        result[label] = dict(row, a_blocks=count)
    result['w_low'] = dict(row, w_blocks=sorted(ranked[:math.ceil(len(ranked)*.75)]))
    result['a_high_w_low'] = dict(result['w_low'], a_blocks=min(k//128, a+jump))
    result['cross'] = dict(row, cross=bool(a and row['w_blocks']))
    return result


FFN_CHOICES = {'base': ('base','base'), 'up_a_low': ('a_low','base'), 'down_a_low': ('base','a_low'),
               'up_a_high': ('a_high','base'), 'down_a_high': ('base','a_high'),
               'both_a_low': ('a_low','a_low'), 'both_a_high': ('a_high','a_high'),
               'both_w_low': ('w_low','w_low'), 'both_a_high_w_low': ('a_high_w_low','a_high_w_low'),
               'cross': ('cross','cross')}


def attention_setting(qk, pv, qt=3, pt=3):
    return dict(qk_fractions=list(qk), pv_fractions=list(pv), qk_terms=qt, pv_terms=pt)


def resolve_attention(row):
    plan = dict(schema_version=1, steps=35, phases=list(PHASES), budget=2., task='i2v', linears={}, attention={'0|0|0':row})
    return RepairPlan(plan).attention_config('i2v',0,0,0,RepairConfig(query_tile=32,transpose_v=True))


def attention_variants(row, h=16):
    # Partial choices use fractions of the CURRENT prompt, never a stale N.
    full=[1.]*h
    return dict(base=copy.deepcopy(row), qk75=attention_setting([.75]*h,full),
                qk50=attention_setting([.5]*h,full), pv75=attention_setting(full,[.75]*h),
                pv50=attention_setting(full,[.5]*h), both75=attention_setting([.75]*h,[.75]*h),
                full3=attention_setting(full,full), qk4=attention_setting(full,full,4,3),
                pv4=attention_setting(full,full,3,4))


def cost_choice(kind, settings, shapes, nt):
    if kind == 'attention':
        m, _, h, hk = shapes['attention']
        cfg=resolve_attention(settings['attention']);n=max(961,(nt-1)*128+1)
        return attention_cost(m,n,h,hk,qk_counts=cfg.head_counts(h,n),
                              pv_counts=cfg.pv_counts(h,n),qk_terms=cfg.terms,pv_terms=cfg.pv_terms or cfg.terms)
    rows=[linear_cost(*shapes[name], **row) for name,row in settings.items()]
    return dict(base_mma=sum(row['base_mma'] for row in rows), total_mma=sum(row['total_mma'] for row in rows))


def compile_plan(value, shapes, nt):
    rows=[]; plan=RepairPlan(value)
    for key,setting in value['linears'].items():
        branch,ph,name=key.split('|',2)
        rows.append(dict(linear_cost(*shapes['linears'][name],**setting),branch=int(branch),phase=int(ph),name=name))
    for key in value['attention']:
        branch,ph,layer=key.split('|',2); m,_,h,hk=shapes['attention'][layer]
        n=max(961,(nt-1)*128+1);cfg=plan.attention_config('i2v',int(branch),int(ph),int(layer),RepairConfig())
        rows.append(dict(attention_cost(m,n,h,hk,qk_counts=cfg.head_counts(h,n),pv_counts=cfg.pv_counts(h,n),
                                       qk_terms=cfg.terms,pv_terms=cfg.pv_terms or cfg.terms),branch=int(branch),phase=int(ph),name=layer))
    return compile_request(rows,value['budget'],expected_modules=list(shapes['linears']),expected_layers=list(shapes['attention']))
