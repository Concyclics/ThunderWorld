"""CPU-only evidence audit; regenerate comparable ranges without mixing cohorts."""
from pathlib import Path
import argparse,json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--retime',type=Path,default=ROOT/'results/optimization_handoff_20260911/retime.json')
parser.add_argument('--output',type=Path,default=ROOT/'results/optimization_handoff_20260911')
args=parser.parse_args();OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
def readcsv(path):
 with (ROOT/path).open() as f:return list(csv.DictReader(f))
def writecsv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
r=json.loads(args.retime.read_text());assert r['complete'] and len(r['rows'])==86
flat=[]
for row in r['rows']:
 for layout,ratio in row['speedups'].items():
  assert len(row['samples_ms'][layout])==5
  assert abs(ratio-row['median_ms']['original']/row['median_ms'][layout])<1e-10
  if row['family']=='A' or layout=='plane':assert row['errors'][layout]['bitwise']
  flat.append(dict(family=row['family'],model=row['model'],shape=row['shape'],tier=row['tier'],layout=layout,speedup=ratio,original_us=row['median_ms']['original']*1000,optimized_us=row['median_ms'][layout]*1000,previous_speedup=row['previous_speedups'][layout],relative_l2=row['errors'][layout]['relative_l2'],bitwise=row['errors'][layout]['bitwise']))
writecsv('retime.csv',flat);ranges=[]
for family in ['A','W']:
 for model in ['edge','nano']:
  for tier in ['a50','act2','wgt2','a50w','dual']:
   for layout in (['gather','direct'] if family=='A' else ['plane','pair']):
    values=[x['speedup'] for x in flat if (x['family'],x['model'],x['tier'],x['layout'])==(family,model,tier,layout)]
    if values:ranges.append(dict(family=family,model=model,tier=tier,layout=layout,shapes=len(values),minimum=min(values),maximum=max(values),above_1_05=sum(v>1.05 for v in values)))
writecsv('ranges.csv',ranges)
fusion=readcsv('results/execution_20260911/fusion_ffn.csv');launch=readcsv('results/auto_budget_20260910/kernels.csv');graph=readcsv('results/execution_20260911/summary.csv')
summary=[]
for model in ['edge','nano']:
 for tier in ['a50','act2','a50w','dual']:
  f=[float(x['speedup']) for x in fusion if x['model']==model and x['tier']==tier];l=[float(x['tuning_speedup']) for x in launch if x['model']==model and x['tier']==tier]
  summary.append(dict(model=model,tier=tier,launch_min=min(l),launch_max=max(l),full_ffn_fusion_min=min(f),full_ffn_fusion_max=max(f)))
writecsv('archived_stage_gains.csv',summary)
lines=['# Verified speed ranges','', '> Ratios compare the same repair math inside each experiment. Ranges span shapes, not confidence intervals. Do not multiply ratios across experiments.','', '## Current frozen-choice A/W retiming','', '| Family | Model | Tier | Layout | Shapes | Speedup range | Shapes >1.05× |','|---|---|---|---|---:|---:|---:|']
for x in ranges:lines.append(f"| {x['family']} | {x['model']} | {x['tier']} | {x['layout']} | {x['shapes']} | {x['minimum']:.3f}–{x['maximum']:.3f}× | {x['above_1_05']} |")
lines+=['','A includes selector + packing + GEMM; W repeats the fixed-support packing + GEMM protocol. W-pair numerical/trajectory limitations still apply.','', '## Archived launch tuning and full-FFN fusion','', '| Model | Tier | Native launch tuning | Complete FFN fusion |','|---|---|---:|---:|']
for x in summary:lines.append(f"| {x['model']} | {x['tier']} | {x['launch_min']:.3f}–{x['launch_max']:.3f}× | {x['full_ffn_fusion_min']:.3f}–{x['full_ffn_fusion_max']:.3f}× |")
lines+=['','## Archived complete-request Graph comparison','', '| Model | Mode | Task | Graph / eager speedup | Graph + fusion / eager speedup |','|---|---|---|---:|---:|']
for x in graph:
 if x['mode']=='bf16':continue
 lines.append(f"| {x['model']} | {x['mode']} | {x['case']} | {float(x['graph_speedup']):.3f}× | {float(x['combined_speedup']):.3f}× |")
lines+=['','Graph results are archived two-seed experiments, not new release timings. Separate-process small differences are noisy; BF16 controls revealed environmental variation. No cross-stage compounded speedup is claimed.','']
(OUT/'SPEED_AUDIT.md').write_text('\n'.join(lines))
paths=[args.retime,ROOT/'results/execution_20260911/fusion_ffn.csv',ROOT/'results/execution_20260911/summary.csv',ROOT/'results/auto_budget_20260910/kernels.csv']
def label(path):
 path=path.resolve()
 return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
(OUT/'audit_sources.json').write_text(json.dumps({'passed':True,'fresh_cases':86,'fresh_A_bitwise':True,'fresh_W_plane_bitwise':True,'sources':{label(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2)+'\n');print(OUT/'SPEED_AUDIT.md')
