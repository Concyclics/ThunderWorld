"""Audit a finished real-video benchmark and archive the exact measured sources."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source_dir = a.output.parent/'source_snapshots'
    source_dir.mkdir(parents=True, exist_ok=True)
    report = {'complete': False, 'results': []}
    for path in a.results:
        d = json.loads(path.read_text())
        args = d['args']
        assert d['complete'] and args['steps']==35 and args['rounds']>=3, path
        assert not any(d['transformer_loading_info'].values()), path
        assert len(d['rows']) == len(args['modes'])*(args['rounds']+1), path
        assert not args['allow_split_k'], 'Audit experimental Split-K separately'
        sources = {}
        for name, expected in d['source_sha256'].items():
            archived = source_dir/(expected+'.py')
            data = Path(name).read_bytes()
            if hashlib.sha256(data).hexdigest()==expected:
                archived.write_bytes(data)
            else:
                assert archived.exists() and hashlib.sha256(archived.read_bytes()).hexdigest()==expected, name
            sources[name] = str(archived)
        for mode in args['modes']:
            rows = [r for r in d['rows'] if r['mode']==mode]
            timed = [r for r in rows if r['timing_sample']]
            assert len(timed)==args['rounds']
            assert len({r['video_sha256'] for r in timed})==1, (path, mode, 'nondeterministic repeats')
            assert len({r['video_sha256'] for r in rows})==2, (path, mode, 'seed failed to change output')
            for r in rows:
                assert r['video_shape']==[args['frames'], args['height'], args['width'], 3]
                assert len(r['transformer_call_seconds'])==args['steps']*2
                expected_ffn = len(d['replaced_modules'])*args['steps']*2 if mode!='bf16' else 0
                expected_attn = d['model_config']['num_hidden_layers']*args['steps']*2 if mode.startswith('joint') else 0
                assert r['operator_calls']=={'ffn': expected_ffn, 'attention': expected_attn}
                if mode.endswith('optimized'):
                    assert r['vs_original_fp4']['identical'], (path, mode, 'optimizer changed decoded output')
        if args.get('cuda_graph'):
            assert d['graph_counts']['captures']==2*len(args['modes'])
            expected_replays=len(args['modes'])*(min(2,args['steps'])*2+(args['rounds']+1)*args['steps']*2)
            assert d['graph_counts']['replays']==expected_replays
        report['results'].append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'requests':len(d['rows']),'full_denoising_steps':args['steps'],
                                  'same_seed_repeat_outputs':'identical','optimized_vs_original_fp4_outputs':'identical',
                                  'operator_call_counts':'verified','checkpoint_loading':'no missing/unexpected/mismatched keys',
                                  'source_snapshots':sources})
    report['complete']=True
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Validated',len(report['results']),'full-model result files')


if __name__=='__main__':
    main()
