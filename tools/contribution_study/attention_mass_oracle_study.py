"""Freeze, execute and report a paired current-attention-mass oracle extension."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from attention_mass_oracle import paired_gap
from structure_repair import COVERAGES, digest, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def presentation(root, output):
    """Export the same selected rows to Chinese prose and compact paper tables."""
    text = ['# 同预算 attention 修复对比', '',
        '展示预算：' + '、'.join(f'{x:g}×' for x in output['reported_attention_mma_budgets']) + '。', '',
        '沿用同批 Edge I2V / Policy 的 16 条轨迹、320 个 attention 面板。下表为相对当前输入 BF16 attention 输出的 NMSE ×10⁻³，越低越好。', '',
        '| 任务 | Attention MMA | Random | Window | Ours | Mass oracle |',
        '|---|---:|---:|---:|---:|---:|']
    delta = ['| 任务 | MMA | 相对 Random | 相对 Window | 相对 Mass oracle |',
             '|---|---:|---:|---:|---:|']
    intervals = ['| 任务 | MMA | 相对 oracle 的误差差距 95% CI |', '|---|---:|---:|']
    tex = [r'\begin{tabular}{llrrrr}', r'\toprule',
           r'Task & Attention MMA & Random & Window & Ours & Mass oracle \\', r'\midrule']
    tex_delta = [r'\begin{tabular}{llrrr}', r'\toprule',
                 r'Task & Attention MMA & vs. Random & vs. Window & vs. Mass oracle \\', r'\midrule']
    for name, cohort in output['cohorts'].items():
        task = {'edge_i2v': 'I2V', 'edge_policy': 'Policy'}.get(name, name)
        for budget in cohort['budgets'].values():
            beta = budget['mma_ratios'][0]
            means, gaps = budget['mean_nmse'], budget['ours_relative_nmse_change_pct']
            vals = [1000 * means[m] for m in ('random', 'window', 'structure', 'mass_oracle')]
            changes = [gaps[m] for m in ('random', 'window', 'mass_oracle')]
            text.append(f'| {task} | {beta:g}× | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
            delta.append(f'| {task} | {beta:g}× | ' + ' | '.join(f'{v:+.2f}%' for v in changes) + ' |')
            lo, hi = budget['comparison']['nmse_gap_ci95_pct']
            intervals.append(f'| {task} | {beta:g}× | [{lo:+.2f}%, {hi:+.2f}%] |')
            tex.append(f'{task} & {beta:g}' + r'$\times$ & ' + ' & '.join(f'{v:.3f}' for v in vals) + r' \\')
            tex_delta.append(f'{task} & {beta:g}' + r'$\times$ & ' + ' & '.join(f'{v:+.2f}' + r'\%' for v in changes) + r' \\')
    text += ['', '相对差距 = 100 × (E_ours / E_comparator − 1)。负值表示我们的误差更低，正值表示更高。', '', *delta, '', *intervals, '',
        '同预算仅指 attention 的 FP4 修复 MMA：全部 base blocks 保留，QK/PV 使用相同的三项修复 mask，逐 M128 行 tile 数配平，计入 M32/N128 padding。Oracle 使用当前完整高精度 attention mass；额外 FP32 QK、softmax 和选择开销单独列账。它最大化 probability mass，不保证最小修复误差。', '',
        '统计先在请求内平均面板，再对固定场景和种子等权平均；8 个 random mask 不当成独立样本。区间来自固定场景内的配对 seed bootstrap，10,000 次。原冻结实验的多重比较校正保持不变；仅缩小展示范围，不重新选参或生成。', '',
        '这些结果衡量 attention 算子输出误差，不是完整视频质量、动作成功率或速度。完整输出哈希及已有对照复测均已核对。']
    policy = output['cohorts'].get('edge_policy', {}).get('budgets', {}).get('0.5')
    if policy:
        gap = policy['exploratory_role_comparisons']['action']['nmse_gap_pct']
        text += ['', f'Policy 的 2× 整体均值差距较小，但 action query 单独计算的 NMSE 仍比 mass oracle 高 {gap:.2f}%；不能将整体接近外推到各个模态，CI 包含零也不等于已证明等价。']
    text += ['', '数据：`results.json`、`comparisons.csv`。论文表格：`paper_table.tex`（误差）、`paper_gaps.tex`（相对差距）。完整原始面板保留在 `*_raw.json`。']
    tex += [r'\bottomrule', r'\end{tabular}', '% NMSE values are multiplied by 1000; lower is better.']
    tex_delta += [r'\bottomrule', r'\end{tabular}', '% Relative NMSE change = 100*(Ours/comparator - 1); negative is better.']
    for name, lines in [('REPORT.zh.md', text), ('paper_table.tex', tex), ('paper_gaps.tex', tex_delta)]:
        (root / 'report' / name).write_text('\n'.join(lines) + '\n')


def check_frozen(p):
    for path, sha in p['frozen_files'].items():
        if digest(path) != sha:
            raise ValueError('Frozen source or data changed: ' + path)


def prepare(a):
    old = json.loads((a.reference / 'protocol.json').read_text())
    if a.root.exists():
        raise FileExistsError(a.root)
    frozen = {str(ROOT / p): sha for p, sha in old['source_sha256'].items()}
    files = [a.reference / 'protocol.json', HERE / 'attention_mass_oracle.py', HERE / 'attention_mass_oracle_run.py']
    jobs = {}
    for name in ('edge_i2v', 'edge_policy'):
        job = dict(old['jobs'][name])
        result_path = a.reference / name / 'result.json'
        result = json.loads(result_path.read_text())
        if not result['complete'] or len(result['rows']) != len(job['cases']) * len(old['evaluation_seeds']):
            raise ValueError('Incomplete reference cohort: ' + name)
        job['reference_dir'] = str((a.reference / name).resolve())
        files += [result_path, Path(job['prior']), Path(job['case_manifest'])]
        files += sorted((a.reference / name).glob('*_heads.repair.json'))
        # Includes the generator and installed Cosmos transformer/pipeline source.
        frozen.update({str(ROOT / p): sha for p, sha in result['source_sha256'].items()})
        frozen.update({v['path']: v['sha256'] for v in result['dependency_sources'].values()})
        jobs[name] = job
    frozen.update({str(f.resolve()): digest(f) for f in files})
    protocol = dict(schema='attention-mass-oracle-v1', frozen=True, jobs=jobs,
        reference_root=str(a.reference.resolve()), physical_gpu=a.gpu, performance_eligible=False,
        evaluation_seeds=old['evaluation_seeds'], steps=35, probe_steps=old['probe_steps'], branch=0,
        coverages=list(COVERAGES), terms=3, frozen_files=frozen,
        expected_requests=16, expected_panels=320,
        score='Current BF16 Q/K -> FP32 QK with TF32 off -> full-key softmax; sum real K128 keys and average real M128 rows per head; no sampling',
        optimum='Maximum captured probability mass under the same H*K per-M128-row quota. Not an oracle for minimum repair error.',
        budget='Exactly matched issued FP4 repair MMA including padding, same mask for QK/PV, all base blocks retained. Oracle FP32 score work is separately accounted, outside that budget.',
        selection='Frozen existing structure prior; no refit or test-error tuning. Oracle uses current Q/K only, no V/output/errors.',
        statistics='Equal panel weights within request, equal requests/scenes; paired seed bootstrap within fixed scenes; 10000 replicates; 3-budget familywise intervals per cohort',
        claims='Local attention output consistency on unchanged BF16 trajectories, not full-generation quality, deployment latency, or repair-error optimality')
    check_frozen(protocol)
    write(a.root / 'protocol.json', protocol)
    print(json.dumps(dict(frozen=True, requests=16, panels=320, output=str(a.root))), flush=True)


def run(a):
    p = json.loads((a.root / 'protocol.json').read_text())
    check_frozen(p)
    state = dict(complete=False, completed=[], performance_eligible=False)
    for name, job in p['jobs'].items():
        out = a.root / name
        previous = json.loads((out / 'result.json').read_text()) if (out / 'result.json').exists() else None
        if previous and previous.get('attention_mass_oracle', {}).get('exact_replay'):
            state['completed'].append(name)
            continue
        command = [sys.executable, str(HERE / 'attention_mass_oracle_run.py'), '--model', job['model'],
            '--model-path', job['model_path'], '--case-manifest', job['case_manifest'], '--cases', *job['cases'],
            '--output', str(out), '--seeds', *map(str, p['evaluation_seeds']), '--steps', '35', '--sample',
            '--modes', 'bf16', '--structure-prior', job['prior'], '--reference-dir', job['reference_dir'],
            '--probe-layers', *map(str, job['layers'])]
        if previous:
            command.append('--resume')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(p['physical_gpu']), TW_SHARED_CALIBRATION_GPUS=str(p['physical_gpu']),
                   TW_SHARED_CALIBRATION_LOCK_SCOPE='device', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        log = a.root / (name + '.log')
        state.update(active=name, command=command, log=str(log))
        write(a.root / 'progress.json', state)
        print(json.dumps(dict(starting=name, log=str(log))), flush=True)
        with log.open('a') as handle:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if process.returncode:
            state.update(paused=True, exit_code=process.returncode)
            write(a.root / 'progress.json', state)
            raise RuntimeError('Oracle experiment paused: ' + str(log))
        state['completed'].append(name)
        write(a.root / 'progress.json', state)
    state.update(complete=True, active=None)
    write(a.root / 'progress.json', state)


def report(a):
    p = json.loads((a.root / 'protocol.json').read_text())
    check_frozen(p)
    report_budgets = getattr(a, 'report_budgets', [1.5, 2.0])
    coverages = [(budget - 1) / 2 for budget in sorted(set(report_budgets))]
    if not set(coverages).issubset(p['coverages']):
        raise ValueError('Requested report budget was not measured')
    cohorts, csv_rows = {}, []
    text = ['# Current-attention-mass oracle', '', p['optimum'], '', p['budget'], '', p['claims'], '']
    text += ['Reported attention MMA budgets: ' + ', '.join(f'{x:.1f}x' for x in sorted(set(report_budgets))),
             'Reporting filter only; original raw observations and frozen multiplicity corrections are retained.', '']
    for name, job in p['jobs'].items():
        run_data = json.loads((a.root / name / 'result.json').read_text())
        if not run_data.get('attention_mass_oracle', {}).get('exact_replay'):
            raise ValueError('Full output/trajectory replay audit has not passed')
        values, mass_values, overlaps, role_values = [defaultdict(list) for _ in range(4)]
        raw, diagnostics, checked = [], [], 0
        for req in run_data['rows']:
            case, seed = req['case'], req['seed']
            stem = f'bf16_{case}_seed{seed}_r0_heads'
            current = json.loads((a.root / name / (stem + '.oracle.json')).read_text())
            old = json.loads((Path(job['reference_dir']) / (stem + '.repair.json')).read_text())
            if not current['complete'] or current['replay_audit']['mismatches']:
                raise ValueError('Local attention replay audit failed')
            checked += current['replay_audit']['checked']
            rows = [dict(r, case=case, seed=seed) for r in current['rows']]
            # Reuse old random/window only after exact per-panel and trajectory audits.
            rows += [dict(r, case=case, seed=seed) for r in old['rows']
                     if r['method'] == 'window' or r['method'].startswith('random_')]
            raw.extend(rows)
            for r in rows:
                method = 'random' if r['method'].startswith('random_') else r['method']
                values[case, seed, r['coverage'], method].append(r['nmse'])
                for role, error in r['role_nmse'].items():
                    role_values[case, seed, r['coverage'], method, role].append(error)
            for r in current['mass_diagnostics']:
                diagnostics.append(dict(r, case=case, seed=seed))
                for method, capture in r['captured_mass'].items():
                    method = 'random' if method.startswith('random_') else method
                    mass_values[case, seed, r['coverage'], method].append(capture)
                overlaps[r['coverage']].append(r['mask_overlap'])
        def table(store, cov, method, role=None):
            return [[float(np.mean(store[(c, s, cov, method) + (() if role is None else (role,))]))
                     for s in p['evaluation_seeds']] for c in job['cases']]
        base = table(values, 0., 'base')
        result = dict(requests=len(run_data['rows']), panels=len(run_data['rows']) * len(job['layers']) * len(p['probe_steps']),
                      exact_replayed_panel_methods=checked, base_nmse=float(np.mean(base)),
                      full_three_nmse=float(np.mean(table(values, 1., 'full_three'))), budgets={})
        text += [f'## {name}', '', '| Attention FP4 MMA | Random NMSE | Window | Ours | Mass oracle | vs random | vs window | vs oracle (95% CI) |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
        for cov in coverages:
            means = {m: float(np.mean(table(values, cov, m))) for m in ('random', 'window', 'structure', 'mass_oracle')}
            gaps = {m: 100 * (means['structure'] / means[m] - 1) for m in ('random', 'window', 'mass_oracle')}
            captures = {m: float(np.mean(table(mass_values, cov, m))) for m in ('random', 'window', 'structure', 'mass_oracle')}
            comparison = paired_gap(table(values, cov, 'structure'), table(values, cov, 'mass_oracle'), base)
            roles = sorted({key[-1] for key in role_values})
            role_errors = {r: {m: float(np.mean(table(role_values, cov, m, r))) for m in means} for r in roles}
            role_comparisons = {r: paired_gap(table(role_values, cov, 'structure', r),
                table(role_values, cov, 'mass_oracle', r), table(role_values, 0., 'base', r)) for r in roles}
            # Exploratory diagnostics, not extra independent statistical samples.
            group_errors = {}
            for group_field in ('phase', 'layer'):
                group_errors[group_field] = {str(group): {m: float(np.mean([
                    r['nmse'] for r in raw if r['coverage'] == cov and r['method'] == m and r[group_field] == group
                ])) for m in ('structure', 'mass_oracle')}
                    for group in sorted({r[group_field] for r in raw})}
            by_scene = {c: {m: float(np.mean([np.mean(values[c, s, cov, m]) for s in p['evaluation_seeds']])) for m in means} for c in job['cases']}
            ratios = sorted({r['mma_ratio'] for r in raw if r['coverage'] == cov})
            unions = {m: float(np.mean([r['gqa_kv_union_density'] for r in raw if r['coverage'] == cov and r['method'] == m])) for m in ('structure', 'mass_oracle')}
            result['budgets'][str(cov)] = dict(mma_ratios=ratios, mean_nmse=means, captured_mass=captures,
                ours_relative_nmse_change_pct=gaps,
                structure_mass_retained_pct=100 * captures['structure'] / captures['mass_oracle'],
                mean_mask_overlap=float(np.mean(overlaps[cov])), comparison=comparison,
                role_nmse=role_errors, exploratory_role_comparisons=role_comparisons,
                exploratory_group_nmse=group_errors, per_scene_nmse=by_scene, gqa_kv_union_density=unions)
            ci = comparison['nmse_gap_ci95_pct']
            text.append(f'| {ratios[0]:.1f}x | {means["random"]:.7f} | {means["window"]:.7f} | {means["structure"]:.7f} | {means["mass_oracle"]:.7f} | {gaps["random"]:+.2f}% | {gaps["window"]:+.2f}% | {comparison["nmse_gap_pct"]:+.2f}% [{ci[0]:+.2f}, {ci[1]:+.2f}] |')
            csv_rows.append(dict(cohort=name, coverage=cov, attention_mma=ratios[0], **means,
                vs_random_nmse_change_pct=gaps['random'], vs_window_nmse_change_pct=gaps['window'],
                structure_mass=captures['structure'], oracle_mass=captures['mass_oracle'],
                nmse_gap_pct=comparison['nmse_gap_pct'], gap_ci95_low=ci[0], gap_ci95_high=ci[1],
                oracle_benefit_recovered_pct=comparison['oracle_benefit_recovered_pct']))
        cohorts[name] = result
        write(a.root / 'report' / (name + '_raw.json'), dict(rows=raw, mass_diagnostics=diagnostics, performance_eligible=False))
        text += ['', 'Relative change = 100 * (Ours / comparator - 1). Negative means lower NMSE; positive means higher NMSE. The mass oracle optimizes probability mass, not repair benefit.', '']
    output = dict(complete=True, performance_eligible=False, protocol_sha256=digest(a.root / 'protocol.json'),
                  reporter_sha256=digest(__file__), reported_attention_mma_budgets=sorted(set(report_budgets)),
                  reporting_filter_only=True, cohorts=cohorts, scope=p['claims'])
    write(a.root / 'report/results.json', output)
    (a.root / 'report/REPORT.md').write_text('\n'.join(text) + '\n')
    with (a.root / 'report/comparisons.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    presentation(a.root, output)
    print(json.dumps(output, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'run', 'report'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--report-budgets', type=float, nargs='+', choices=[1.5, 2.0, 2.5], default=[1.5, 2.0],
                        help='Presentation budgets only; does not change the frozen experiment or statistical correction')
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.operation == 'prepare' and args.reference is None:
        parser.error('prepare requires --reference')
    globals()[args.operation](args)


if __name__ == '__main__':
    main()
