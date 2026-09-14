"""Resolve portable policy paths after unpacking; does not alter repair budgets."""
import argparse
import json
from pathlib import Path
from thunderworld_fp4.cosmos.settings import RunConfig
from thunderworld_fp4.cutlass.artifact import verify_library

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--policy',type=Path,required=True)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--small-library',type=Path)
p.add_argument('--direct-pack',action='store_true',help='Opt in to fused activation-scale stores; repair budget is unchanged')
p.add_argument('--resource-mode',choices=['shared-calibration','exclusive'],default='shared-calibration')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
config=json.loads((a.policy/'config.json').read_text())
config.update(budget_plan=str((a.policy/'plan.json').resolve()),linear_backend='cutlass-fused',
              cutlass_library=str(a.library.resolve()),resource_mode=a.resource_mode,cutlass_direct_pack=a.direct_pack)
verify_library(a.library)
if a.small_library:
    verify_library(a.small_library);config['cutlass_small_library']=str(a.small_library.resolve())
config=RunConfig.from_dict(config)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(config.to_dict(),indent=2)+'\n')
