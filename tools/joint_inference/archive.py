"""Archive raw measurement rows and deduplicated receipts without model/video blobs."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('matrix',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    matrix=json.loads(a.matrix.read_text())
    if not matrix['complete']:raise ValueError('Archive complete experiments only')
    result=dict(complete=True,schema_version=1,matrix=matrix,receipts={},measurements=[],sources={})
    result['sources'][str(a.matrix)]=hashlib.sha256(a.matrix.read_bytes()).hexdigest()
    for job in matrix['jobs']:
        path=Path(job['output'])/'result.json';run=json.loads(path.read_text())
        if not run['complete']:raise ValueError('Incomplete request run')
        key='|'.join([job['case'],job['config'],job['audio_output']])
        static={k:v for k,v in run.items() if k not in ('rows','timings','memory','cold_ready_seconds')}
        result['receipts'].setdefault(key,static)
        # Keep exact numeric observations, all preparation timings, memory
        # snapshots and source hashes from every independent process.
        result['measurements'].append(dict(receipt=key,round=job['round'],rows=run.get('rows',[]),
            timings=run['timings'],memory=run.get('memory'),cold_ready_seconds=run.get('cold_ready_seconds'),
            process_seconds=job.get('process_seconds'),source_sha256=run['source_sha256']))
        result['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
