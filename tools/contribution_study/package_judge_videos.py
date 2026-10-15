"""Package completed judge MP4s with portable indices and original provenance.

No video is re-encoded. Referenced symlinks are read as bytes, so the ZIPs do not
depend on source-machine paths. Large raw NPZ/MKV copies stay in the source job.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()


def jsonlines(rows):
    return ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows).encode()


def event(name, **data):
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=name, **data)), flush=True)


class Bundle:
    def __init__(self, path, prefix):
        self.prefix = prefix
        self.zip = zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6)
        self.members = {}

    def info(self, name, compressed):
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError('Unsafe archive member: '+name)
        info = zipfile.ZipInfo(self.prefix+'/'+name)
        info.compress_type = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        return info

    def file(self, name, source, expected=None, compressed=True):
        source = Path(source)
        if name in self.members:
            previous = self.members[name]
            if previous['source'] != str(source.resolve()) or (expected and previous['sha256'] != expected):
                raise ValueError('Conflicting archive member: '+name)
            return previous['sha256']
        before = source.stat()
        digest = hashlib.sha256()
        size = 0
        with source.open('rb') as src, self.zip.open(self.info(name, compressed), 'w', force_zip64=True) as dst:
            while block := src.read(1024*1024):
                digest.update(block)
                dst.write(block)
                size += len(block)
        after = source.stat()
        actual = digest.hexdigest()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != before.st_size:
            raise ValueError('Source changed while packaging: '+str(source))
        if expected and actual != expected:
            raise ValueError('Source hash mismatch: '+str(source))
        self.members[name] = dict(sha256=actual, bytes=size, source=str(source.resolve()))
        return actual

    def data(self, name, value):
        if name in self.members:
            raise ValueError('Duplicate archive member: '+name)
        self.zip.writestr(self.info(name, True), value)
        self.members[name] = dict(sha256=hashlib.sha256(value).hexdigest(), bytes=len(value), source=None)


def verify(path, bundle, samples, ffprobe):
    event('verify_start', bundle=bundle.prefix, files=len(bundle.members))
    with zipfile.ZipFile(path) as archive:
        expected = {bundle.prefix+'/'+name: value for name, value in bundle.members.items()}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError('Archive membership differs')
        for name, value in expected.items():
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != value['sha256'] or archive.getinfo(name).file_size != value['bytes']:
                raise ValueError('Archive content mismatch: '+name)
        for sample in samples:
            for field in ('video_path', 'input_image', 'metadata', 'generation_receipt', 'generation_log'):
                if bundle.prefix+'/'+sample[field] not in expected:
                    raise ValueError('Nonportable sample reference: '+sample[field])
        probes = []
        with tempfile.TemporaryDirectory(prefix='judge-zip-probe-') as temporary:
            for method in sorted({sample['method'] for sample in samples}):
                sample = next(row for row in samples if row['method']==method)
                video = Path(temporary)/(sample['sample_id']+'.mp4')
                with archive.open(bundle.prefix+'/'+sample['video_path']) as source, video.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
                streams = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-count_frames',
                    '-select_streams', 'v:0', '-show_entries',
                    'stream=codec_name,width,height,nb_read_frames,r_frame_rate', '-of', 'json', str(video)],
                    text=True))['streams']
                if len(streams)!=1:
                    raise ValueError('Expected one video stream')
                stream = streams[0]
                if (stream['codec_name'], stream['width'], stream['height'],
                    int(stream['nb_read_frames']), stream['r_frame_rate']) != ('h264', 320, 192, 61, '24/1'):
                    raise ValueError('Unexpected video dimensions or codec')
                probes.append(dict(sample_id=sample['sample_id'], method=method, stream=stream))
    return dict(all_members_sha256_verified=True, all_zip_crc_verified=True,
                all_index_paths_resolve=True, decoded_representatives=probes)


def pack(root, output, key, rows, ffprobe):
    model, benchmark = key
    name = model+'_'+benchmark
    target = output/(name+'.zip')
    receipt_path = output/(name+'.package.json')
    if target.exists():
        previous = read(receipt_path)
        if previous['source_index_sha256'] != sha(root/'judge_index.jsonl') or sha(target) != previous['sha256']:
            raise ValueError('Existing package differs from this frozen job')
        return previous
    partial = target.with_suffix('.zip.partial')
    if partial.exists():
        raise ValueError('Incomplete prior package exists: '+str(partial))
    event('pack_start', bundle=name, videos=len(rows))
    bundle = Bundle(partial, name)
    samples = []
    try:
        for index, row in enumerate(sorted(rows, key=lambda row:row['sample_id'])):
            metadata = read(row['metadata'])
            for field in ('sample_id', 'model', 'method', 'benchmark', 'case_id', 'seed', 'prompt'):
                if metadata[field] != row[field]:
                    raise ValueError('Export metadata identity mismatch: '+field)
            if not metadata['complete'] or (metadata['frames'], metadata['height'], metadata['width'], metadata['fps']) != (61,192,320,24):
                raise ValueError('Unexpected export configuration')
            video = metadata['videos']['judge_mp4']
            if Path(video['file']).resolve() != Path(row['video_path']).resolve():
                raise ValueError('Video reference mismatch')
            sid = row['sample_id']
            portable = {field:row[field] for field in ('sample_id','model','method','benchmark',
                'category','source_id','case_id','seed','prompt')}
            log_source = Path(metadata['generation_log'])
            log_id = hashlib.sha256(str(log_source.resolve()).encode()).hexdigest()[:24]
            portable.update(video_path='videos/'+sid+'.mp4', input_image='inputs/'+Path(row['input_image']).name,
                metadata='metadata/'+sid+'.json', generation_log='logs/'+log_id+'.log',
                generation_receipt='receipts/'+row['method']+'/'+row['case_id']+'.json',
                video_sha256=video['sha256'], input_image_sha256=metadata['input_image_sha256'],
                frames=61, width=320, height=192, fps=24, steps=35)
            bundle.file(portable['video_path'], row['video_path'], video['sha256'], compressed=False)
            bundle.file(portable['input_image'], row['input_image'], metadata['input_image_sha256'], compressed=False)
            bundle.file(portable['metadata'], row['metadata'])
            bundle.file(portable['generation_log'], log_source)
            bundle.file(portable['generation_receipt'], metadata['generation_receipt'])
            samples.append(portable)
            if (index+1)%500==0:
                event('pack_progress', bundle=name, videos=index+1, expected=len(rows))
        manifest = dict(schema='thunderworld-judge-video-package-v1', model=model, benchmark=benchmark,
            videos=len(rows), scenes=len({row['case_id'] for row in rows}),
            methods=dict(Counter(row['method'] for row in rows)), frames=61, width=320, height=192,
            fps=24, steps=35, paths_relative_to='this extracted package directory',
            source_job=str(root), source_job_sha256=sha(root/'manifest.json'),
            source_index_sha256=sha(root/'judge_index.jsonl'),
            omitted_payloads=['lossless MKV copies', 'raw floating-point NPZ outputs', 'model weights'],
            metadata_note='Original metadata and generation receipts retain source paths as provenance; '
                          'use judge_index.jsonl or judge_inputs.jsonl for portable paths.',
            no_reencoding=True, performance_eligible=False)
        bundle.data('manifest.json', encoded(manifest))
        bundle.data('judge_index.jsonl', jsonlines(samples))
        blind = [{field:row[field] for field in ('sample_id','benchmark','category','source_id',
                  'prompt','input_image','video_path')} for row in samples]
        bundle.data('judge_inputs.jsonl', jsonlines(blind))
        for archive_name, source in [('job_manifest.json',root/'manifest.json'),
                ('model_manifest.json',root/model/'final/manifest.json'),
                ('pool_status.json',root/'pool_status.json')]:
            bundle.file('provenance/'+archive_name, source)
        documentation = f'''# {name}: {len(rows):,} generated videos

All five methods, both seeds per scene; 320x192, 61 frames, 24 FPS, 35 steps.
MP4 files are the original H.264 CRF15 exports, copied without re-encoding.

Use judge_index.jsonl to map anonymous video IDs to model/method/scene/seed.
Use judge_inputs.jsonl for judge inputs without method labels. Both indices use
paths relative to this directory, so resolve them against this directory on another host.
Inputs, original per-video metadata, generation receipts and complete referenced
generation logs are included. Original metadata/receipts retain source-machine
paths for provenance; use the portable indices for files in this package.

From this extracted directory, verify all packaged files with:
    sha256sum -c SHA256SUMS

Every MP4 and input image was checked against its export SHA256. Every archive
member was read back and checked (SHA256 and ZIP CRC). One video per method was
decoded for a frame-count/shape/FPS check. Referenced source symlinks became regular files.

Lossless MKV copies, original float32 NPZ outputs and model weights remain in the
source job and are not duplicated in this MP4 delivery package. MP4 pixel values
include lossy codec changes; use lossless/original outputs for strict numerical comparisons.
'''
        bundle.data('README.md', documentation.encode())
        sums = ''.join(value['sha256']+'  '+name+'\n' for name,value in sorted(bundle.members.items()))
        bundle.data('SHA256SUMS', sums.encode())
    finally:
        bundle.zip.close()
    verification = verify(partial, bundle, samples, ffprobe)
    receipt = dict(manifest, archive=target.name, bytes=partial.stat().st_size, sha256=sha(partial),
                   members=len(bundle.members), verification=verification,
                   completed_utc=datetime.now(timezone.utc).isoformat())
    receipt_path.write_bytes(encoded(receipt))
    partial.replace(target)
    event('pack_complete', bundle=name, bytes=receipt['bytes'], videos=len(rows))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffprobe', default='/usr/bin/ffprobe')
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    pool = read(root/'pool_status.json')
    if not pool['complete'] or not all(value['complete'] for value in pool['models'].values()):
        raise ValueError('Full generation and export must finish before packaging')
    rows = [json.loads(line) for line in (root/'judge_index.jsonl').read_text().splitlines()]
    ids, keys, expected, groups = set(), set(), set(), defaultdict(list)
    for model in pool['models']:
        study = read(root/model/'final/manifest.json')
        if study['steps'] != 35:
            raise ValueError('Unexpected generation steps')
        for entry in study['cases']:
            for method in study['methods']:
                for seed in entry['seeds']:
                    expected.add((model, method, entry['case']['id'], seed))
    for row in rows:
        key = (row['model'], row['method'], row['case_id'], row['seed'])
        if key in keys or row['sample_id'] in ids:
            raise ValueError('Duplicate sample')
        ids.add(row['sample_id'])
        keys.add(key)
        groups[(row['model'],row['benchmark'])].append(row)
    if keys != expected or len(rows) != sum(value['expected'] for value in pool['models'].values()):
        raise ValueError('Export index differs from frozen cohort')
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 20*1024**3:
        raise ValueError('Need at least 20 GiB free for this video delivery')
    archives = [pack(root, output, key, group, args.ffprobe) for key,group in sorted(groups.items())]
    (output/'SHA256SUMS').write_text(''.join(row['sha256']+'  '+row['archive']+'\n' for row in archives))
    report = dict(complete=True, videos=len(rows), total_bytes=sum(row['bytes'] for row in archives),
        source_job=str(root), source_job_sha256=sha(root/'manifest.json'), archives=archives,
        completed_utc=datetime.now(timezone.utc).isoformat())
    (output/'packages.json').write_bytes(encoded(report))
    table = '\n'.join(f"| {row['archive']} | {row['videos']:,} | {row['bytes']/10**9:.3f} GB |" for row in archives)
    (output/'README.md').write_text('''# Cosmos Edge/Nano 完整视频交付

全部 33,880 个生成样本的 MP4；每个 ZIP 独立可用，包含输入图、prompt/seed/方法索引、
原始元数据、生成 receipt 与完整引用日志。解压后以包目录为相对路径基准读取索引。

| 压缩包 | 视频数 | 大小 |
|---|---:|---:|
'''+table+'''

在当前目录运行 `sha256sum -c SHA256SUMS` 可检查四个 ZIP。
解压后每包还提供逐文件 SHA256SUMS。packages.json 包含完整计数及验证结果。
未重新编码视频；源码中的历史视频符号链接已经打成真实文件，迁移后不依赖原目录。
无损 MKV、原始浮点 NPZ 和模型权重保留在源目录，不包含在本次 MP4 交付中。
''')
    event('all_complete', videos=len(rows), bytes=report['total_bytes'], output=str(output))


if __name__ == '__main__':
    main()
