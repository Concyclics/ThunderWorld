"""Freeze a ten-scene I2V adaptation with timestamp-aligned real-video targets."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from baselines import sha,write


def resize_crop(frame,height=192,width=320):
    """Match Cosmos3's float32 bilinear/antialias resize, center crop and round."""
    h,w=frame.shape[:2];scale=max(width/w,height/h)
    rh,rw=math.ceil(scale*h),math.ceil(scale*w)
    value=torch.from_numpy(frame.copy()).permute(2,0,1).unsqueeze(0).float()
    value=F.interpolate(value,size=(rh,rw),mode='bilinear',align_corners=False,antialias=True)
    top,left=round((rh-height)/2),round((rw-width)/2)
    return value[0,:,top:top+height,left:left+width].round().clamp(0,255).byte().permute(1,2,0).numpy()


def nearest_indices(timestamps,target):
    timestamps=np.asarray(timestamps,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    if len(timestamps)<2 or np.any(np.diff(timestamps)<=0):raise ValueError('Non-increasing video timestamps')
    if target[0]<timestamps[0]-1.e-8 or target[-1]>timestamps[-1]+1.e-8:
        raise ValueError('Reference does not cover the requested time span; no padding or time stretching')
    return np.abs(target[:,None]-timestamps[None,:]).argmin(1)


def decode(path):
    target=np.arange(61,dtype=np.float64)/24.
    timestamps=[];frames=[];first=None;origin=None
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_type='NONE';stream.codec_context.thread_count=1
        info=dict(width=stream.width,height=stream.height,average_rate=str(stream.average_rate),
                  time_base=str(stream.time_base),declared_frames=stream.frames)
        for frame in container.decode(stream):
            if frame.pts is None:raise ValueError('Missing presentation timestamp')
            seconds=float(frame.pts*stream.time_base)
            if origin is None:origin=seconds
            t=seconds-origin
            rgb=frame.to_ndarray(format='rgb24')
            if first is None:first=rgb.copy()
            frames.append(resize_crop(rgb));timestamps.append(t)
            if t>=target[-1]-1.e-8:break
    indices=nearest_indices(timestamps,target)
    aligned=np.stack(frames)[indices].astype(np.float32)/255.
    error=np.abs(np.asarray(timestamps)[indices]-target)
    if error.max()>1/24.+1.e-6:raise ValueError('Timestamp mismatch exceeds one generated frame')
    info.update(origin_pts_seconds=origin,decoded_relative_pts=timestamps,target_pts=target.tolist(),
        selected_frame_indices=indices.tolist(),selected_pts=np.asarray(timestamps)[indices].tolist(),
        maximum_timestamp_error_seconds=float(error.max()),unique_reference_frames=len(set(indices.tolist())))
    return first,aligned,info


def prepare(root,base):
    torch.set_num_threads(8)
    manifest=root/'edge/final/manifest.json'
    if manifest.exists():raise ValueError('Study already frozen; do not replace generated-input identities')
    selection=json.loads((root/'selection.json').read_text())
    if not selection['frozen'] or len(selection['cases'])!=10:raise ValueError('Expected 10 frozen scene IDs')
    teacher=json.loads((base/'edge/final/manifest.json').read_text())
    reference=dict(complete=False,selection_sha256=sha(root/'selection.json'),source_sha256=sha(__file__),
        source_repo=selection['source_repo'],source_revision=selection['source_revision'],cases={},
        temporal='PTS nearest-neighbor at t=0,1/24,...,60/24 seconds from first decoded frame; ties choose earlier',
        spatial='Aspect-preserving float32 bilinear antialias resize, centered 320x192 crop, round to uint8 / 255',
        primary='Future frames 1..60, excluding the provided conditioning frame',
        secondary='All 61 frames, including the conditioning frame',
        method_independent=True,alignment_search_against_generated_outputs=False)
    (root/'references').mkdir(parents=True,exist_ok=True);(root/'inputs').mkdir(exist_ok=True)
    cases=[];copies={}
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _preprocess_conditioning_image
    for i,entry in enumerate(selection['cases']):
        sid=Path(entry['file_name']).stem;cid='paibench_c__'+sid
        path=root/'source'/entry['video'];first,video,info=decode(path)
        image=root/'inputs'/(cid+'.png');Image.fromarray(first).save(image)
        actual=_preprocess_conditioning_image(Image.fromarray(first),height=192,width=320)
        actual=((actual[0].permute(1,2,0)+1)*127.5).round().byte().numpy()
        if not np.array_equal(actual,np.rint(video[0]*255).astype(np.uint8)):
            raise ValueError('Reference spatial transform differs from the generator conditioning transform')
        output=root/'references'/(cid+'.npz');np.savez_compressed(output,video=video)
        case=dict(id=cid,task='i2v',prompt=entry['caption_text'],image=str(image),height=192,width=320,frames=61,
                  negative_prompt='distorted objects, flickering, blurry, low quality')
        item=dict(benchmark='paibench_c_i2v',category=entry['category'],source_id=sid,case=case,
            seeds=[92222001+i],image_sha256=sha(image),pixel_sha256=hashlib.sha256(first.tobytes()).hexdigest())
        cases.append(item);copies[cid]=str(image)
        reference['cases'][cid]=dict(source_file=str(path),source_sha256=sha(path),aligned_file=str(output),
            aligned_sha256=sha(output),signal_sha256=hashlib.sha256(video.tobytes()).hexdigest(),
            input_image=str(image),input_image_sha256=sha(image),first_frame_transform_matches=True,**info)
        print(json.dumps(dict(prepared_reference=cid,max_time_error=info['maximum_timestamp_error_seconds'])),flush=True)
    reference['complete']=True;write(root/'references/manifest.json',reference)
    study=copy.deepcopy(teacher)
    study.update(cases=cases,input_copies=copies,reference_manifest=str(root/'references/manifest.json'),
        reference_manifest_sha256=sha(root/'references/manifest.json'),
        selection=selection,role='exploratory_real_video_i2v',
        sources=dict(generation_recipe=dict(path=str(base/'edge/final/manifest.json'),sha256=sha(base/'edge/final/manifest.json')),
                     real_videos=dict(repo=selection['source_repo'],revision=selection['source_revision'])),
        statistics=dict(unit='scene, one paired seed per scene',bootstrap_seed=220921,replicates=10000,
            inferential=False,interpretation='Exploratory ten-scene real-video reference experiment; no official benchmark score'))
    study['protocol'].update(metric_reference='Aligned real video, future frames 1..60 (primary); all frames (secondary)',
        dataset_scope='PAI-Bench-C source videos adapted to image-to-video using first frame plus original caption. '
                      'No depth, segmentation, edge or blur conditioning; not the official PAI-Bench-C task.',
        temporal_alignment=reference['temporal'],spatial_alignment=reference['spatial'])
    write(manifest,study)
    write(root/'cohort.json',dict(frozen=True,manifest=str(manifest),manifest_sha256=sha(manifest),cases=cases))
    print(json.dumps(dict(complete=True,manifest=str(manifest),cases=10,methods=len(study['methods']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=Path('/home/yufeng/thunderworld/fp4_expansion_demo_20260921'))
    args=parser.parse_args();prepare(args.root.resolve(),args.base.resolve())
