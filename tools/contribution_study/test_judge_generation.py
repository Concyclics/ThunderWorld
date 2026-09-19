from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from baselines import sha,write
from judge_generation import claim,snapshot
from judge_video_export import frames_uint8,encode,export_available


def test_frame_conversion_preserves_count_and_rejects_bad_inputs():
    case=dict(frames=3,height=8,width=10)
    x=np.linspace(-.1,1.1,3*8*10*3,dtype=np.float32).reshape(3,8,10,3)
    assert frames_uint8(x,case).shape==x.shape
    assert frames_uint8(x,case).dtype==np.uint8
    with pytest.raises(ValueError):frames_uint8(x[:2],case)
    x.flat[1]=np.nan
    with pytest.raises(ValueError):frames_uint8(x,case)


def test_two_gpus_claim_distinct_cost_ordered_tasks(tmp_path):
    write(tmp_path/'manifest.json',dict(frozen=True))
    tasks=[dict(id='nano_long',estimated_seconds=90),dict(id='nano_medium',estimated_seconds=60),dict(id='edge',estimated_seconds=10)]
    write(tmp_path/'queue.json',dict(job_sha256=sha(tmp_path/'manifest.json'),tasks={t['id']:dict(status='pending',attempts=[]) for t in tasks}))
    with ThreadPoolExecutor(2) as pool:
        selected=list(pool.map(lambda gpu:claim(tmp_path,dict(tasks=tasks),gpu),[0,1]))
    assert {r['id'] for r in selected}=={'nano_long','nano_medium'}
    assert claim(tmp_path,dict(tasks=tasks),0)['id']=='edge'
    assert claim(tmp_path,dict(tasks=tasks),1) is None


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'),reason='System video codecs required')
def test_lossless_export_and_resume_reject_corrupt_video(tmp_path,monkeypatch):
    root=tmp_path/'edge';folder=root/'final/generation/arcquant/case';folder.mkdir(parents=True)
    case=dict(id='case',frames=3,height=8,width=10,prompt='Test motion',image=str(tmp_path/'input.png'))
    Path(case['image']).write_bytes(b'input fixture')
    entry=dict(case=case,seeds=[7],benchmark='rbench',category='test',source_id='case',image_sha256=sha(case['image']))
    encoding=dict(ffmpeg='/usr/bin/ffmpeg',ffprobe='/usr/bin/ffprobe')
    manifest=root/'final/manifest.json'
    write(manifest,dict(model_name='edge',model='/model',methods={'arcquant':dict(config={})},
        cases=[entry],input_copies={'case':case['image']},video_export=encoding))
    video=np.random.default_rng(19).random((3,8,10,3),dtype=np.float32)
    np.savez_compressed(folder/'seed7.npz',video=video)
    row=dict(seed=7,outputs='seed7.npz',outputs_sha256=sha(folder/'seed7.npz'),
        sha256={'video':hashlib.sha256(video.tobytes()).hexdigest()})
    write(folder/'result.json',dict(complete=True,identity=dict(manifest_sha256=sha(manifest),case=case,seeds=[7]),rows=[row]))
    assert export_available(root,set())['complete']
    sample=json.loads((root/'videos/rbench/arcquant/case/seed7.json').read_text())
    assert sample['videos']['lossless_rgb']['rgb_roundtrip_bitwise_equal']
    assert sample['videos']['judge_mp4']['stream']['nb_read_frames']=='3'
    before=Path(sample['videos']['judge_mp4']['file']).stat().st_mtime_ns
    assert export_available(root,set())['complete']
    assert Path(sample['videos']['judge_mp4']['file']).stat().st_mtime_ns==before
    blind=json.loads((root/'judge_blind/inputs.jsonl').read_text())
    assert 'method' not in blind and 'arcquant' not in blind['video_path']

    # A full-corpus study can reuse proven output bytes without fabricating a new
    # generation receipt, rerunning inference, or transcoding the old videos.
    full=tmp_path/'full/edge';alias=full/'final/generation/arcquant/case'
    alias.parent.mkdir(parents=True);alias.symlink_to(folder,target_is_directory=True)
    study=json.loads(manifest.read_text())
    original_metadata=root/'videos/rbench/arcquant/case/seed7.json'
    study['reuse']=dict(root=str(root),case_ids=['case'],manifest_sha256=sha(manifest),
        receipt_sha256={'arcquant/case':sha(folder/'result.json')},
        metadata_sha256={'arcquant/case/7':sha(original_metadata)})
    write(full/'final/manifest.json',study)
    with monkeypatch.context() as patch:
        def unexpected_encode(*args,**kwargs):raise AssertionError('Reused output must not be encoded again')
        patch.setattr('judge_video_export.encode',unexpected_encode)
        assert export_available(full,set())['complete']
    copied=json.loads((full/'videos/rbench/arcquant/case/seed7.json').read_text())
    assert copied['provenance']['reused_metadata']['sha256']==sha(original_metadata)
    assert Path(copied['videos']['judge_mp4']['file']).resolve()==Path(sample['videos']['judge_mp4']['file'])
    assert json.loads((folder/'result.json').read_text())['identity']['manifest_sha256']==sha(manifest)
    with Path(sample['videos']['judge_mp4']['file']).open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='Changed exported video'):export_available(root,set())
    with pytest.raises(ValueError,match='Changed exported video'):export_available(full,set())


def test_progress_uses_full_expected_count_and_counts_reused_rows(tmp_path):
    folder=tmp_path/'edge/final/generation/method/case';folder.mkdir(parents=True)
    write(folder/'result.json',dict(rows=[dict(seed=1),dict(seed=2)]))
    write(tmp_path/'edge/export_progress.json',dict(complete=False,completed=2))
    write(tmp_path/'queue.json',dict(tasks={'task':dict(status='pending')}))
    job=dict(models={'edge':dict(expected_videos=16940,reused_videos=720)},
             tasks=[dict(id='task',model='edge')],gpus=[0,1])
    result=snapshot(tmp_path,job)
    assert result['models']['edge']==dict(generation_complete=False,generated=2,expected=16940,
        reused=720,exported=2,complete=False,performance_eligible=False)
