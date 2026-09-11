"""Export compact versioned deployment policies; archived full policies remain intact."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'profiles';OUT.mkdir(exist_ok=True)
manifest={}
for model in ['edge','nano']:
 src=ROOT/f'results/auto_budget_20260910/records/runs5090/auto_budget_{model}_calibration/policy_validated.json';p=json.loads(src.read_text());assert p['complete']
 compact={k:p[k] for k in ['schema_version','model','model_revision','complete','phases','policies','launches','validation_gate']}
 compact['source_policy']={'path':str(src.relative_to(ROOT)),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
 compact['scope']='Runtime-only export of the archived validation-selected FFN policy. Refit on a new device/model/distribution; original full policy retains calibration units and costs.'
 for policy in compact['policies'].values():
  policy['by_task']={task:{k:v for k,v in value.items() if k in ['extra_flops','extra_flop_cap','realized_fraction','predicted_loss','predicted_ms']} for task,value in policy['by_task'].items()}
 target=OUT/(model+'.json');target.write_text(json.dumps(compact,indent=2)+'\n');manifest[model]={'path':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
