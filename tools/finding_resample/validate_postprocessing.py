"""Validate metric coverage and decoded-waveform identity independently."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile


def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();checks=[]
    def key(r):return r['mode'],r['case'],r['seed'],r['repeat']
    for path in a.inputs:
        report=json.loads((path/'result.json').read_text());assert report['complete']
        rows={key(r):r for r in report['rows']}
        perceptual=json.loads((path/'perceptual.json').read_text());assert perceptual['complete']
        assert {key(r) for r in perceptual['rows']}==set(rows) and len(perceptual['rows'])==len(rows)
        for row in perceptual['rows']:
            for name,values in row.items():
                if not name.startswith('vs_'):continue
                arr=np.asarray(values['per_frame']);assert len(arr)==rows[key(row)]['video_shape'][0]
                assert np.isfinite(arr).all() and arr.min()>=-1e-7
                assert np.isclose(arr.mean(),values['mean']) and np.isclose(arr.max(),values['max'])
        audio_path=path/'audio_decode.json';count=0
        if audio_path.exists():
            audio=json.loads(audio_path.read_text());assert audio['complete']
            expected={key(r) for r in report['rows'] if 'sound_latent' in r['sha256']}
            assert {key(r) for r in audio['rows']}==expected and len(audio['rows'])==len(expected)
            for row in audio['rows']:
                stem=f'{row["mode"]}_{row["case"]}_seed{row["seed"]}_r{row["repeat"]}'
                rate,wave=wavfile.read(path/(stem+'.wav'))
                wave=wave.T
                assert rate==row['sampling_rate'] and list(wave.shape)==row['shape']
                assert np.isfinite(wave).all() and wave.min()>=-1 and wave.max()<=1
                assert hashlib.sha256(wave.tobytes()).hexdigest()==row['sha256']
                assert row['decode_seconds']>0;count+=1
        checks.append(dict(input=str(path),lpips_trajectories=len(rows),verified_waveforms=count,passed=True))
    a.output.write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n');print(checks)


if __name__=='__main__':main()
