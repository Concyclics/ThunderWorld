"""Human-readable stage schedules; all percentages are issued-work weighted."""
import argparse
import collections
import json
from pathlib import Path
from baselines import write
from schedule_envelope import compile_envelope
from thunderworld_fp4.cosmos.budget import PHASES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    catalog=json.loads((a.study/'candidates/candidates.json').read_text());output=[]
    lines=['# 降噪阶段修复计划','','以下统一展示两个 CFG 分支都为 8 个 key tiles 的计划；实际请求按各自长度计算，完整预算检查覆盖两分支独立的 81 种长度组合。百分比按 padded MMA 加权，不是层数平均。']
    for candidate in catalog['candidates']:
        plan=json.loads(Path(candidate['config']['budget_plan']).read_text())
        shape=json.loads((a.study/'calibration'/f'b{candidate["budget"]:g}'/'definitions.json').read_text())['shapes']
        ledger=compile_envelope(plan,shape,(8,8));stages=[]
        for ph in range(4):
            values=collections.defaultdict(collections.Counter)
            for row in ledger['rows']:
                if row['phase']!=ph:continue
                part=('ffn' if '.mlp_moe_gen.' in row['name'] else 'projection') if row['kind']=='linear' else 'attention'
                terms=row['mma_by_dtype']['fp4']
                values[part].update(terms)
                values[part]['base_total']+=row['base_mma']
            item=dict(steps=[PHASES[ph],PHASES[ph+1]])
            for part in ('ffn','projection'):
                counts=values[part];base=counts['base_total']
                item[part]={key:counts[key]/base for key in ('a','w','cross')}
            attention=values['attention']
            item['attention']={side:sum(attention[side+'_'+term] for term in ('a','w','cross'))/attention[side+'_base'] for side in ('qk','pv')}
            stages.append(item)
        signatures=[]
        for branch in range(2):
            signatures.append(len({json.dumps(dict(linears={k.split('|',2)[-1]:v for k,v in plan['linears'].items() if k.startswith(f'{branch}|{ph}|')},
                                                   attention={k.split('|',2)[-1]:v for k,v in plan['attention'].items() if k.startswith(f'{branch}|{ph}|')}),sort_keys=True) for ph in range(4)}))
        output.append(dict(name=candidate['name'],stages=stages,structural_graph_variants_by_cfg=signatures))
        lines+=['',f'## {candidate["name"]}','','| 降噪 steps | FFN A | FFN W | 投影 A | 投影 W | QK 补项/base | PV 补项/base |','|---|---:|---:|---:|---:|---:|---:|']
        for row in stages:
            lines.append(f'| [{row["steps"][0]}, {row["steps"][1]}) | {row["ffn"]["a"]:.1%} | {row["ffn"]["w"]:.1%} | {row["projection"]["a"]:.1%} | {row["projection"]["w"]:.1%} | {row["attention"]["qk"]:.4f} | {row["attention"]["pv"]:.4f} |')
        lines+=['',f'每个 CFG 分支的结构配置数：{signatures}。交叉项单列于 JSON；没有把 W1 并集密度当作每阶段执行密度。']
    a.archive.mkdir(parents=True,exist_ok=True);write(a.archive/'schedule_profiles.json',dict(complete=True,profiles=output))
    (a.archive/'SCHEDULES.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
