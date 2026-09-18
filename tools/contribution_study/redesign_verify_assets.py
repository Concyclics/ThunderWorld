"""Verify physical W1 support unions and unchanged RTN byte planes on CPU."""
import argparse,json
from pathlib import Path
import torch
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('calibration','candidates','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    am=json.loads((a.calibration/'rtn_assets/manifest.json').read_text());cm=json.loads((a.candidates/'candidates.json').read_text());checks=[]
    for candidate in cm['candidates']:
        bank=Path(candidate['assets']);manifest=json.loads((bank/'manifest.json').read_text());plan=json.loads(Path(candidate['config']['budget_plan']).read_text());kept=full=0
        if sha(bank/'manifest.json')!=candidate['assets_sha256']:raise ValueError('Changed asset manifest')
        for name,entry in manifest['modules'].items():
            path=bank/entry['file'];source_path=a.calibration/'rtn_assets'/am['modules'][name]['file']
            if sha(path)!=entry['sha256'] or sha(source_path)!=am['modules'][name]['sha256']:raise ValueError('Corrupt tensor asset')
            asset=torch.load(path,weights_only=True,map_location='cpu');source=torch.load(source_path,weights_only=True,map_location='cpu')
            if 'codes' in asset or 'scales' in asset:raise ValueError('Hidden complete W1 plane')
            supports=[r['w_blocks'] for key,r in plan['linears'].items() if key.split('|',2)[-1]==name];union=sorted({i for ids in supports for i in ids})
            if asset['residual_indices'].tolist()!=union:raise ValueError('Asset differs from exact phase union')
            design=plan.get('design',{})
            if design.get('w1_static_across_phase_cfg'):
                if any(ids!=supports[0] for ids in supports):raise ValueError('Static W1 support changes by phase/CFG')
                nb,kb=source['score'].shape
                if design['w1_axis']=='N128 with full K':
                    if any(sum(i//kb==n for i in union) not in (0,kb) for n in range(nb)):raise ValueError('Incomplete K reduction in N bundle')
                elif design['w1_axis']=='K128 with full N':
                    if any(sum(i%kb==k for i in union) not in (0,nb) for k in range(kb)):raise ValueError('Incomplete N coverage in K bundle')
                else:raise ValueError('Unknown static support axis')
            positions={int(v):i for i,v in enumerate(source['residual_indices'])};selected=[positions[i] for i in union]
            for key in ('codes_base','scales_base','factor','prescale'):
                if not torch.equal(asset[key].reshape(-1).view(torch.uint8),source[key].reshape(-1).view(torch.uint8)):raise ValueError('Changed RTN base representation')
            for key in ('codes_residual','scales_residual'):
                if not torch.equal(asset[key],source[key][selected]):raise ValueError('Changed retained W1 bytes')
            kept+=len(union);full+=source['score'].numel()
        checks.append(dict(name=candidate['name'],modules=len(manifest['modules']),w1_blocks=kept,all_w1_blocks=full,w1_density=kept/full,passed=True))
        print(json.dumps(checks[-1]),flush=True)
    write(a.output,dict(complete=True,candidates_sha256=sha(a.candidates/'candidates.json'),checks=checks,
        scope='CPU exact encoded-byte/base/support verification. Runtime current-input checks and quality are separate.'))


if __name__=='__main__':main()
