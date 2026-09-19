"""CPU export of immutable generation outputs to judge videos and sample indices."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from baselines import sha, write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def frames_uint8(video, case):
    expected=(case.get('frames',61),case.get('height',192),case.get('width',320),3)
    if video.shape!=expected or not np.isfinite(video).all():
        raise ValueError('Nonfinite or incorrectly shaped generated video')
    return np.rint(np.clip(video,0,1)*255).astype(np.uint8)


def probe(path, ffprobe, frames, height, width, fps):
    value=json.loads(subprocess.check_output([ffprobe,'-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=codec_name,pix_fmt,width,height,nb_read_frames,r_frame_rate',
        '-of','json',str(path)],text=True))['streams']
    if len(value)!=1:raise ValueError('Expected one video stream')
    info=value[0];a,b=map(int,info['r_frame_rate'].split('/'))
    if (int(info['nb_read_frames']),info['height'],info['width'])!=(frames,height,width) or a/b!=fps:
        raise ValueError('Encoded video changed dimensions, frame count or FPS')
    return info


def encode(frames, target, *, lossless, ffmpeg='/usr/bin/ffmpeg', ffprobe='/usr/bin/ffprobe', fps=24):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name(target.stem+'.partial'+target.suffix)
    n,h,w,c=frames.shape
    if c!=3 or frames.dtype!=np.uint8:raise ValueError('Expected uint8 RGB frames')
    command=[ffmpeg,'-hide_banner','-loglevel','info','-nostdin','-y','-f','rawvideo',
        '-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',str(fps),'-i','pipe:0','-an','-threads','2']
    command+=['-c:v','ffv1','-level','3','-g','1','-slicecrc','1','-pix_fmt','bgr0'] if lossless else [
        '-c:v','libx264','-preset','medium','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart']
    command+=[str(temp)]
    with target.with_suffix(target.suffix+'.encode.log').open('ab') as log:
        log.write((json.dumps(command)+'\n').encode());log.flush()
        subprocess.run(command,input=frames.tobytes(),stdout=subprocess.DEVNULL,stderr=log,check=True)
        info=probe(temp,ffprobe,n,h,w,fps)
        if lossless:
            decoded=subprocess.check_output([ffmpeg,'-v','error','-nostdin','-threads','2','-i',str(temp),
                '-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stderr=log)
            if decoded!=frames.tobytes():raise ValueError('Lossless video RGB roundtrip differs')
    temp.replace(target)
    return dict(file=str(target),sha256=sha(target),stream=info,bytes=target.stat().st_size,
                rgb_roundtrip_bitwise_equal=True if lossless else None)


def jsonlines(path, rows):
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows))
    temp.replace(path)


def export_available(root, verified):
    (root/'judge_blind').mkdir(parents=True,exist_ok=True)
    manifest=root/'final/manifest.json';study=json.loads(manifest.read_text())
    identity=dict(manifest_sha256=sha(manifest),exporter_sha256=sha(__file__),video=study['video_export'])
    reuse=study.get('reuse',{});reused_ids=set(reuse.get('case_ids',[]))
    rows=[]
    for method,spec in study['methods'].items():
        for entry in study['cases']:
            cid=entry['case']['id'];folder=root/'final/generation'/method/cid
            receipt=folder/'result.json'
            if not receipt.exists():continue
            result=json.loads(receipt.read_text())
            reused=cid in reused_ids
            expected_manifest=reuse['manifest_sha256'] if reused else identity['manifest_sha256']
            if reused and sha(receipt)!=reuse['receipt_sha256'][method+'/'+cid]:
                raise ValueError('Changed original reusable receipt')
            if (result['identity']['manifest_sha256']!=expected_manifest or
                result['identity'].get('method',method)!=method or
                result['identity']['case']!=entry['case'] or result['identity']['seeds']!=entry['seeds']):
                raise ValueError('Changed generated input or configuration')
            for row in result['rows']:
                seed=row['seed']
                if seed not in entry['seeds']:raise ValueError('Unexpected generated seed')
                key=fingerprint([study['model_name'],method,cid,seed])[:24]
                dest=root/'videos'/entry['benchmark']/method/cid
                metadata=dest/f'seed{seed}.json'
                source=folder/row['outputs']
                provenance=dict(identity=identity,row_sha256=fingerprint(row),npz_sha256=row['outputs_sha256'])
                origin=None
                if reused:
                    origin=Path(reuse['root'])/'videos'/entry['benchmark']/method/cid/f'seed{seed}.json'
                    provenance['reused_metadata']=dict(file=str(origin),sha256=reuse['metadata_sha256'][f'{method}/{cid}/{seed}'])
                if metadata.exists():
                    sample=json.loads(metadata.read_text())
                    if sample['provenance']!=provenance:raise ValueError('Changed exported request')
                    if key not in verified:
                        if sha(source)!=row['outputs_sha256']:raise ValueError('Changed raw generation')
                        for asset in sample['videos'].values():
                            if sha(asset['file'])!=asset['sha256']:raise ValueError('Changed exported video')
                else:
                    if sha(source)!=row['outputs_sha256']:raise ValueError('Corrupt raw generation')
                    if reused:
                        if sha(origin)!=provenance['reused_metadata']['sha256']:raise ValueError('Changed reusable video metadata')
                        old=json.loads(origin.read_text())
                        if (old['method_spec']!=spec or old['provenance']['row_sha256']!=fingerprint(row) or
                            old['provenance']['identity']['manifest_sha256']!=expected_manifest or
                            not old['videos']['lossless_rgb']['rgb_roundtrip_bitwise_equal'] or
                            (old['frames'],old['height'],old['width'],old['fps'])!=(entry['case']['frames'],entry['case']['height'],entry['case']['width'],24)):
                            raise ValueError('Changed reusable video configuration')
                        dest.mkdir(parents=True,exist_ok=True);videos=copy.deepcopy(old['videos'])
                        for kind,asset in videos.items():
                            src=Path(asset['file']);link=dest/src.name
                            if sha(src)!=asset['sha256']:raise ValueError('Changed reusable video bytes')
                            if not link.exists():link.symlink_to(src)
                            if link.resolve()!=src.resolve():raise ValueError('Changed reusable video link')
                            asset['file']=str(link)
                        rgb_sha=old['rgb_uint8_sha256'];frame_count=old['frames'];height=old['height'];width=old['width']
                    else:
                        with np.load(source) as data:video=data['video']
                        if hashlib.sha256(video.tobytes()).hexdigest()!=row['sha256']['video']:
                            raise ValueError('Raw video signal hash mismatch')
                        frames=frames_uint8(video,entry['case'])
                        videos={kind:encode(frames,dest/f'seed{seed}{suffix}',lossless=lossless,
                            ffmpeg=study['video_export']['ffmpeg'],ffprobe=study['video_export']['ffprobe'])
                            for kind,suffix,lossless in [('judge_mp4','.mp4',False),('lossless_rgb','.mkv',True)]}
                        rgb_sha=hashlib.sha256(frames.tobytes()).hexdigest();frame_count=len(frames);height=frames.shape[1];width=frames.shape[2]
                    sample=dict(complete=True,sample_id=key,model=study['model_name'],model_path=study['model'],
                        method=method,benchmark=entry['benchmark'],category=entry['category'],source_id=entry['source_id'],
                        case_id=cid,seed=seed,prompt=entry['case']['prompt'],negative_prompt=entry['case'].get('negative_prompt'),
                        input_image=study['input_copies'][cid],input_image_sha256=entry['image_sha256'],
                        raw_outputs=str(source),generation_receipt=str(receipt),generation_log=str(folder/'generation.log'),
                        exported_utc=datetime.now(timezone.utc).isoformat(),generated_npz_mtime_ns=source.stat().st_mtime_ns,
                        request=row,method_spec=spec,provenance=provenance,videos=videos,
                        rgb_uint8_sha256=rgb_sha,frames=frame_count,fps=24,
                        height=height,width=width,performance_eligible=False)
                    write(metadata,sample)
                    print(json.dumps(dict(exported=key,model=study['model_name'],method=method,case=cid,seed=seed)),flush=True)
                alias=root.parent/'judge_blind/videos'/f'{key}.mp4';alias.parent.mkdir(parents=True,exist_ok=True)
                target=Path(sample['videos']['judge_mp4']['file'])
                if not alias.exists():alias.symlink_to(target)
                if alias.resolve()!=target.resolve():raise ValueError('Changed blind video alias')
                verified.add(key)
                rows.append(dict(sample_id=key,model=study['model_name'],method=method,benchmark=entry['benchmark'],
                    category=entry['category'],source_id=entry['source_id'],case_id=cid,seed=seed,prompt=sample['prompt'],
                    input_image=sample['input_image'],video_path=str(target),
                    lossless_video_path=sample['videos']['lossless_rgb']['file'],raw_outputs=str(source),
                    metadata=str(metadata),blind_video_path=str(alias)))
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate sample IDs')
    jsonlines(root/'judge_index.jsonl',rows)
    blind=[dict(sample_id=r['sample_id'],benchmark=r['benchmark'],category=r['category'],source_id=r['source_id'],
        prompt=r['prompt'],input_image=r['input_image'],video_path=r['blind_video_path']) for r in rows]
    blind.sort(key=lambda r:r['sample_id'])
    jsonlines(root/'judge_blind/inputs.jsonl',blind)
    expected=sum(len(e['seeds']) for e in study['cases'])*len(study['methods'])
    state=dict(complete=len(rows)==expected,completed=len(rows),expected=expected,identity=identity,performance_eligible=False)
    write(root/'export_progress.json',state)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--follow',action='store_true');a=p.parse_args();verified=set()
    (a.root/'judge_blind').mkdir(parents=True,exist_ok=True)
    while True:
        state=export_available(a.root,verified)
        if state['complete'] or not a.follow:return
        flow=a.root/'workflow.json'
        if flow.exists():
            status=json.loads(flow.read_text())
            if status.get('paused'):return
            if status.get('generation_complete'):raise RuntimeError('Generation finished with missing video outputs')
        time.sleep(5)


if __name__=='__main__':main()
