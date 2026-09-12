"""Regenerate the fixed September 2026 experiment report from archived observations."""
from pathlib import Path
import argparse
import json,statistics,csv
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[2]/'results/joint_inference_20260912')
root=parser.parse_args().results
qualities={m:json.loads((root/f'quality_{m}.json').read_text()) for m in ('edge','nano')}
expected=['performance_edge.json','performance_nano_i2v.json','performance_nano_policy_av.json']
performance=[]
for name in expected:
 d=json.loads((root/name).read_text())
 assert d['complete']
 model='edge' if 'edge' in name else 'nano'
 for row in d['rows']:performance.append(dict(model=model,**row))
primary=[('edge','robot_table','dual'),('edge','policy_table','dual'),('nano','robot_table','four'),('nano','policy_table','four_dual_attention'),('nano','av_pottery','four')]
labels={('edge','robot_table'):'Edge I2V',('edge','policy_table'):'Edge policy',('nano','robot_table'):'Nano I2V',('nano','policy_table'):'Nano policy',('nano','av_pottery'):'Nano AV'}
def row(model,case,config,audio='waveform'):
 return next(r for r in performance if (r['model'],r['case'],r['config'],r['audio_output'])==(model,case,config,audio))
def decision(model,case,config):return next(d for d in qualities[model]['decisions'] if d['case']==case and d['config']==config)
def ci(x):return f"{x['mean_paired_difference']:+.5f} [{x['ci95'][0]:+.5f}, {x['ci95'][1]:+.5f}]"
def speed(base,cand):return f'{base/cand:.3f}×'
lines=['# 联合 FP4 完整推理：实现与复测', '',
'实现分支 `feat/joint-inference-runtime`，主体提交 `ac12603`。正式性能均在物理 GPU3：RTX PRO 6000 Blackwell Server Edition（96GB、600W）测量，不是 RTX5090。GPU0 只保留实现验证和被中止的非正式性能尝试。', '',
'框架已实现默认无诊断探针的完整请求入口；最终种子测试中，Edge/Nano I2V 通过了严格 FP8 一致性门槛，两个 policy 与 Nano AV 未通过。高预算 FP4 的延迟需与下表一起判断，不能把一致性改善等同于保质加速。', '',
'## 质量结果', '',
'同一模型修订、相同生成投影范围、输入、35 steps、CFG 和 Graph 条件。FP8 是原生 W8A8 投影（W per-tensor、A 在线 per-token）加 BF16 attention。这里评价相对 BF16 的输出一致性，不是动作成功率或跨场景泛化。', '',
'最终8个种子为91001–91008；验证集为92001–92008。候选在最终测试前确定，测试结果没有参与校准或修改配置。下表区间为候选误差减 FP8 误差的配对95% bootstrap CI，负值更好。', '',
'| 任务 | 预定候选 | LPIPS差值及95% CI | 动作/音频主指标差值及95% CI | 全部条件通过 |',
'|---|---|---|---|---|']
for model,case,config in primary:
 d=decision(model,case,config);extra='—'
 metric='action.relative_l2' if case=='policy_table' else 'waveform.log_spectral_l1'
 if metric in d['metrics']:extra=ci(d['metrics'][metric])
 lines.append(f"| {labels[model,case]} | {config} | {ci(d['metrics']['video.lpips'])} | {extra} | {'是' if d['passed'] else '否'} |")
lines+=['', '视频还要求 SSIM 不降、相对 L2 不升；policy 还要求 action RMSE 不升；AV 还要求 waveform/sound latent L2 不升。完整守门条件、普通 W4A4、ACT2、完整交叉项等结果见 [Edge质量原始JSON](quality_edge.json) 和 [Nano质量原始JSON](quality_nano.json)。', '',
'![配对置信区间](quality_intervals.png)', '',
'## 完整请求性能', '',
'每配置独立进程加载和预热，先运行一次完整35步暖机请求，再按固定随机顺序交错5轮，每轮3个固定种子（15次请求）。以下中位数包含 CPU 请求状态/参数构造、图像读取、编码、Transformer、最终输出转换与解码；排除磁盘结果写入和离线质量指标。debug 全部关闭、调优全部命中、逐请求独占检查通过。', '',
'| 任务 | BF16秒 | FP8秒 | 普通FP4秒 | 修复FP4秒 | 相对BF16速度 | 相对FP8速度 | 修复峰值 allocated / reserved GiB |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for model,case,config in primary:
 b,e,a,c=(row(model,case,k) for k in ('bf16','fp8','base',config))
 vals=[f"{r['median_seconds']:.4f}" for r in (b,e,a,c)]
 lines.append('| '+' | '.join([labels[model,case],*vals,speed(b['median_seconds'],c['median_seconds']),speed(e['median_seconds'],c['median_seconds']),f"{c['peak_allocated_bytes']/2**30:.3f} / {c['peak_reserved_bytes']/2**30:.3f}"])+' |')
lines+=['', '速度倍率小于1表示修复方案更慢。完整各配置中位数、P90、样本量和显存见性能 JSON 与 [汇总CSV](performance.csv)。AV 上表包含波形解码；latent-only 数据单列如下，不能混用口径。', '',
'| Nano AV配置 | latent-only秒 | 包含波形解码秒 |', '|---|---:|---:|']
for cfg in ('bf16','fp8','base','four','four_legacy','ffn_only'):
 lines.append(f"| {cfg} | {row('nano','av_pottery',cfg,'latent')['median_seconds']:.4f} | {row('nano','av_pottery',cfg)['median_seconds']:.4f} |")
lines+=['', '## 消融与实现边界', '',
'legacy 对照在相同新入口中使用原始 CSR FFN 内核；投影范围、当前输入、修复预算和 Graph 条件完全一致。它是匹配范围的 FFN 内核消融，不是旧历史脚本整条路径的原始计时。旧脚本仅FFN的投影范围与当前联合方案不同，未混入排名。', '',
'| 任务 | legacy FFN秒 | 优化FFN秒 | 请求延迟降低 |', '|---|---:|---:|---:|']
for model,case,config in primary:
 old=row(model,case,config+'_legacy')['median_seconds'];new=row(model,case,config)['median_seconds']
 lines.append(f'| {labels[model,case]} | {old:.4f} | {new:.4f} | {(1-new/old)*100:+.2f}% |')
a=json.loads((root/'ablation.json').read_text());assert a['complete']
med=a['medians']
lines+=['', 'Edge policy 的探针/Graph/pointwise融合消融保持相同输出哈希；各变体15次。带探针结果只用于诊断，不进入正式性能排名。', '', '| 变体 | 请求中位数秒 |', '|---|---:|']
for name,value in med.items():lines.append(f'| {name} | {value:.4f} |')
lines += ['', f"去除历史摘要与GPU timestep回读的联合探针开销，延迟降低 {(1-med['graph']/med['historical_probes'])*100:+.2f}%；eager 改联合 Graph 为 {med['eager']/med['graph']:.3f}×；pointwise/amax 融合降低 {(1-med['graph']/med['unfused'])*100:+.2f}%。这些是同口径独立消融，不能直接相乘。", '',
'若原生密集 FP4/BF16 的有效吞吐比为 R，全量三乘积/四乘积修复仅 GEMM 的理想加速上限分别为 R/3、R/4；还未扣除 packing、非线性、attention 归约及其他组件。attention 的 padded-tile 乘积数为 1+(terms−1)×实际选中比例，尾块向上取整后不一定等于设定比例。它们是计算量模型，不是本机峰值吞吐测量。', '',
'高预算方案仍要付出多份 FP4 乘积、在线 packing/selector、softmax/P量化、combine、BF16 非线性与未量化部分的成本。全量 W1 采用规则寻址，A修复复用W0和FP32累加器；attention的Q残差重载作为逐位等价候选参与完整算子调优，未宣称其必然更快。已从实际编译 PTX 确认 FFN 和 attention 使用原生 e2m1 FP4 MMA，见 [ISA证据](native_fp4_isa.json)。没有 Tensor Core 峰值占用率的计数器证据，不能声称“打满算力”。', '',
'## 显存、离线校准与未达成项', '',
'W0/W1 离线量化；A及attention Q/K/V/P使用当前输入在线量化。导出资产只保存策略需要的W1支持并集，增加超出资产支持的W预算会被拒绝。全量W1接近两份FP4，含尺度后比单份FP8权重略大；不能同时宣称高预算方案仍具有单份W4存储。', '',
'不同M/A/cross变体共享W0，相同W支持共享W1；多个不同非零W档位仍分别持有GPU residual bank。PyTorch allocated/reserved峰值与常驻模型状态分开统计，Graph池包含中间激活/scratch；reserved不是权重大小，也不包含所有驱动层分配。当前仍先载入原始BF16 pipeline再替换投影，低暖机占用不代表低启动峰值，不保证32GB 5090能加载全部配置。', '',
'已提供无训练的prescale/尺度校准，以及逐block实际修复收益驱动的head配额候选生成。development第一层试验和原始收益已保存；这些局部代理结果尚未通过完整轨迹选参，未混入最终推荐。最终通过的I2V证据来自高预算配置，尚未证明低预算head稀疏策略具备同等质量。', '',
'policy与AV质量目标未达成；普通FP4或ACT2即便更快，也不因速度而绕过门槛。未使用隐藏BF16 attention回退、删base block、旧step输出复用或减少步数来宣布成功。', '',
'## 复现与交付', '',
'统一入口、配置语义、离线资产、调优、质量门槛和自动命令见 [使用文档](../../tools/joint_inference/README.md)。本目录保存原始测量JSON、严格门槛、GPU专属profiles、测试日志、模型文件SHA256和图表。大型模型和输出视频不纳入代码包，模型由目标机器自行下载。', '',
'GPU0 曾出现其他用户的 CUDA 进程，独占保护自动中止相应请求。未完成/重试前的局部轮次保留为历史尝试，未冒充完整5轮数据。早期排除CPU setup的计时也未进入最终排名。']

lines += ['', '## 冷启动与准备时间', '',
'每配置另起一个进程单测一次；保留磁盘 kernel cache。冷启动是父进程发起到 Graph 准备完毕，包含解释器/库导入；不是清空缓存后的首次编译测试。Graph capture 是 warmup 的子项，不能再次相加。', '',
'| 任务/配置 | 冷启动到准备就绪秒 | 模型加载秒 | 权重准备秒 | 预热/建图秒 | 其中capture秒 |',
'|---|---:|---:|---:|---:|---:|']
for filename,model in [('cold_edge.json','edge'),('cold_nano_i2v.json','nano'),('cold_nano_policy_av.json','nano')]:
    data=json.loads((root/filename).read_text());assert data['complete']
    for observation in data['measurements']:
        case,config,audio=observation['receipt'].split('|')
        if config.endswith('_legacy') or config=='ffn_only':continue
        t=observation['timings']
        lines.append(f"| {labels[model,case]} / {config} | {observation['cold_ready_seconds']:.3f} | {t['load_seconds']:.3f} | {t['offline_prepare_seconds']:.3f} | {t['warmup_seconds']:.3f} | {t['graph_capture_seconds']:.3f} |")
lines += ['', '完整进程时间、所有配置与显存快照见 cold JSON。冷启动样本量为每配置1次，不将其当作暖机请求中位数或P90。', '',
'## 验证记录', '',
'CPU/SM120测试、完整模型 debug/eager/Graph 对照、冻结调优后的35步轨迹逐位对照，以及内核 memcheck/racecheck 日志随目录提供。调优表包括真实 M416 上下投影全部8个 FFN 修复档位，以及当前 I2V/AV 请求观察到的其他 M。', '',
'最终结果使用相同量化和内核源哈希。后续 CLI 独立策略解析、Python benchmark 独占保护与显式 mask 调优键修复不改变本轮无显式 mask 的推理数值/热路径；每进程源码哈希均在 raw JSON 中保存。']

(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
with (root/'performance.csv').open('w') as f:
 keys=['model','case','config','audio_output','n','median_seconds','p90_seconds','peak_allocated_bytes','peak_reserved_bytes']
 w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(performance)
