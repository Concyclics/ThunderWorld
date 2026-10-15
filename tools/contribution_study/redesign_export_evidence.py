"""Archive exact native intervention metrics; large RMS panels stay reproducible locally."""
import argparse,gzip,json
from pathlib import Path
from baselines import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cm=json.loads((a.calibration/'manifest.json').read_text())
    if not cm['complete']:raise ValueError('Incomplete calibration')
    records=[]
    for case,digest in cm['cases'].items():
        path=a.calibration/(case+'.json')
        if sha(path)!=digest:raise ValueError('Corrupt calibration')
        value=json.loads(path.read_text())
        value['linears']={name:[{k:v for k,v in row.items() if k!='rms'} for row in rows] for name,rows in value['linears'].items()}
        value.update(source_sha256=digest,omitted='Per-channel RMS only; all exact native errors/head features retained. Full source is reproducible by redesign_calibrate.py.')
        records.append(value)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_bytes(gzip.compress(json.dumps(dict(complete=True,calibration_sha256=sha(a.calibration/'manifest.json'),cases=records),separators=(',',':'),allow_nan=False).encode(),mtime=0))


if __name__=='__main__':main()
