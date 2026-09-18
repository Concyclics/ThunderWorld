"""Describe calibrated local tradeoffs without treating them as trajectory gains."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from baselines import sha, write
from thunderworld_fp4.cosmos.budget import PHASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads((args.study / 'candidates/candidates.json').read_text())
    results = []
    for cap in (1.5, 2., 2.5):
        root = args.study / 'calibration' / f'b{cap:g}'
        manifest = json.loads((root / 'manifest.json').read_text())
        definitions = json.loads((root / 'definitions.json').read_text())
        if not manifest['complete'] or sha(root / 'definitions.json') != manifest['definitions_sha256']:
            raise ValueError('Complete, unchanged calibration required')
        samples = []
        for name, digest in manifest['cases'].items():
            file = root / (name + '.json')
            if sha(file) != digest:
                raise ValueError('Calibration panel changed')
            samples.append(json.loads(file.read_text()))
        for candidate in catalog['candidates']:
            if candidate['budget'] != cap or '_schedule_' not in candidate['name']:
                continue
            folder = Path(candidate['config']['budget_plan']).parent
            allocation = json.loads((folder / 'allocation.json').read_text())
            labels = {key: row['label'] for row in allocation['assignments'].values() for key in row['members']}
            if labels.keys() != definitions['groups'].keys():
                raise ValueError('Schedule does not cover every calibrated group')
            values = collections.defaultdict(list)
            for sample in samples:
                local = collections.defaultdict(lambda: [0., 0., 0])
                for key, label in labels.items():
                    definition = definitions['groups'][key]
                    phase = definition['phase']
                    calls = PHASES[phase + 1] - PHASES[phase]
                    group = definition['kind'], definition['branch'], phase
                    losses = sample['groups'][key]['losses']
                    local[group][0] += calls * losses['base']
                    local[group][1] += calls * losses[label]
                    local[group][2] += calls
                for group, (base, chosen, count) in local.items():
                    values[sample['entry']['benchmark'], *group].append((base / count, chosen / count))
            records = []
            for (benchmark, part, branch, phase), paired in sorted(values.items()):
                paired = np.asarray(paired)
                base, chosen = paired.mean(axis=0)
                records.append(dict(benchmark=benchmark, part=part, branch=branch, phase=phase,
                                    baseline_nmse=float(base), selected_nmse=float(chosen),
                                    relative_reduction=float(1 - chosen / base),
                                    degraded_scenes=int((paired[:, 1] > paired[:, 0]).sum()),
                                    scenes=len(paired)))
            # Aggregate on a common mean-NMSE definition for all three profiles.
            # The allocator's worst-panel objective has different grouping granularity.
            totals = {}
            for part in ('projection', 'ffn', 'attention'):
                rows = [row for row in records if row['part'] == part]
                weights = [PHASES[row['phase'] + 1] - PHASES[row['phase']] for row in rows]
                base = float(np.average([row['baseline_nmse'] for row in rows], weights=weights))
                chosen = float(np.average([row['selected_nmse'] for row in rows], weights=weights))
                totals[part] = dict(baseline_nmse=base, selected_nmse=chosen,
                                    relative_reduction=1 - chosen / base)
            results.append(dict(name=candidate['name'], budget=cap, components=totals, rows=records,
                                allocator_proxy_reduction=1 - allocation['proxy_loss'] / allocation['control_proxy_loss'],
                                calibration_sha256=sha(root / 'manifest.json')))
    args.archive.mkdir(parents=True, exist_ok=True)
    write(args.archive / 'local_tradeoffs.json', dict(complete=True, results=results,
        scope='Calibrated current inputs from each fixed control, not the newly scheduled trajectory. '
              'Equal dataset means; phase-call weighted; each component averaged over its own layers. '
              'Component NMSEs have different denominators and must not be compared as downstream error.'))
    lines = ['# 局部修复收益与预算交换', '',
             '下面是固定对照自身轨迹上、同输入 BF16 参考的 NMSE 变化。正数表示局部误差降低，负数表示增加。',
             '按数据集等权、阶段调用数加权；每个模块在自身层内平均。各模块的归一化分母不同，不能把模块之间的 NMSE 差值当作最终视频误差收益。', '',
             '| 候选 | 投影局部误差降低 | FFN 局部误差降低 | attention 局部误差降低 |',
             '|---|---:|---:|---:|']
    for row in results:
        lines.append('| ' + row['name'] + ' | ' + ' | '.join(
            f'{row["components"][part]["relative_reduction"]:+.2%}'
            for part in ('projection', 'ffn', 'attention')) + ' |')
    lines += ['', '逐阶段、CFG 分支、数据集的原始对比与退化场景数见 `local_tradeoffs.json`。',
              '这些数据用于解释调度选择，不参与验证集选参，也不证明某一模块的变化单独导致最终质量升降。']
    (args.archive / 'LOCAL_TRADEOFFS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
