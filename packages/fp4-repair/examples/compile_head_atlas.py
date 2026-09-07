"""Export explicit per-layer budgets from an existing ThunderWorld head atlas.

This exports an experimental policy, not a new calibration result. The legacy
projection-sensitivity atlas must be validated for attention repair separately.
"""
import argparse
import json
from pathlib import Path
from thunderworld_fp4 import policy_from_head_atlas

parser=argparse.ArgumentParser()
parser.add_argument('atlas',type=Path)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--coverage',type=float,default=.5)
parser.add_argument('--protected-coverage',type=float,default=1.)
args=parser.parse_args()
policy=policy_from_head_atlas(json.loads(args.atlas.read_text()),coverage=args.coverage,
                             protected_coverage=args.protected_coverage)
args.output.write_text(json.dumps({'schema_version':1,'default':policy.default.to_dict(),
    'layers':{str(rule.layers[0]):rule.config.to_dict() for rule in policy.rules}},indent=2)+'\n')
print(f'Exported {len(policy.rules)} layer configurations to {args.output}')
