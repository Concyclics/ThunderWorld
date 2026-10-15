import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/fp4-repair/src'))
sys.path.insert(0, str(ROOT / 'tools/public'))
from thunderworld_fp4 import hub
from common import immutable
from data import prompt_catalog, image_entry, COSMOS_REVISION


def test_import_is_cpu_only():
    subprocess.run([sys.executable, '-c',
        'import thunderworld_fp4.hub; import sys; assert "torch" not in sys.modules'],
        check=True, env={'PATH': str(Path(sys.executable).parent),
                         'PYTHONPATH': str(ROOT / 'packages/fp4-repair/src')})


def test_argument_validation_precedes_download(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('Unnecessary model download')
    monkeypatch.setattr(hub, 'resolve_release', forbidden)
    with pytest.raises(ValueError, match='Inference needs'):
        hub.main(['--model', 'edge'])


def test_launch_keeps_paths_and_prompt_literal(tmp_path):
    cmd = hub.inference_command(tmp_path / 'release folder', '2x', image=tmp_path / 'my image.png',
        prompt='a robot; $(echo unsafe) `literal`', output=tmp_path / 'output folder')
    assert cmd[cmd.index('--prompt')+1] == 'a robot; $(echo unsafe) `literal`'
    assert cmd[cmd.index('--image')+1] == str(tmp_path / 'my image.png')
    assert cmd[-4:] == ['--debug', 'off', '--execution', 'graph']
    with pytest.raises(ValueError, match='not both'):
        hub.inference_command(tmp_path, '2x', image='x', prompt='y', case_json='z', output='o')


def test_release_is_pinned_and_profile_scoped(tmp_path, monkeypatch):
    entry = hub.registry()['models']['edge']
    (tmp_path / 'release.json').write_text(json.dumps(dict(repo_id=entry['repo_id'],
        base_model={'revision': entry['base_required']['revision']}, profiles={'1.5x': {}})))
    (tmp_path / 'infer.py').write_text('')
    calls = []
    def download(*args, **kwargs): calls.append((args, kwargs)); return str(tmp_path)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=download))
    assert hub.resolve_release('edge', '1.5x', local_files_only=True) == tmp_path
    assert calls[0][1]['revision'] == entry['revision']
    assert calls[0][1]['ignore_patterns'] == ['profiles/2x/**']
    assert calls[0][1]['local_files_only']
    with pytest.raises(ValueError, match='identity mismatch'):
        hub.resolve_release('nano', '1.5x', tmp_path)


def test_frozen_manifest_cannot_be_repurposed(tmp_path):
    p = tmp_path / 'manifest.json'
    immutable(p, {'seed': 1}); immutable(p, {'seed': 1})
    with pytest.raises(ValueError, match='Changed frozen inputs'):
        immutable(p, {'seed': 2})
    assert json.loads(p.read_text()) == {'seed': 1}


def test_registry_and_figure_receipts():
    import hashlib
    for model, entry in hub.registry()['models'].items():
        assert len(entry['revision']) == 40
        assert set(entry['profiles']) == {'1.5x', '2x'}
        assert all(entry['generation_bitwise_checks'].values())
        data = json.loads((ROOT / f'configs/public/{model}-calibration.json').read_text())
        identities, seeds = set(), set()
        for role, rows in data['splits'].items():
            assert len(rows) == data['counts'][role]
            for row in rows:
                assert row['case_id'] not in identities
                identities.add(row['case_id'])
                assert not seeds.intersection(row['seeds'])
                seeds.update(row['seeds'])
    for row in json.loads((ROOT / 'docs/assets/provenance.json').read_text()):
        assert hashlib.sha256((ROOT / 'docs/assets' / row['file']).read_bytes()).hexdigest() == row['sha256']


def test_image_and_prompt_identity(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    import hashlib
    p = tmp_path / 'rbench/imgs/a.png'; p.parent.mkdir(parents=True)
    img = Image.new('RGB', (5, 3), color=(23, 71, 18)); img.save(p)
    meta = dict(benchmark='rbench', category='robot', source_id='a', case_id='rbench__a', seeds=[11],
        pixel_sha256=hashlib.sha256(str(img.size).encode()+img.tobytes()).hexdigest(),
        prompt_sha256=hashlib.sha256(b'robot').hexdigest())
    value = image_entry(meta, dict(image='imgs/a.png', prompt='robot'), {}, dataset_root=tmp_path)
    assert value['case']['height'] == 192 and value['case']['frames'] == 61
    with pytest.raises(ValueError, match='prompt_sha256'):
        image_entry(meta, dict(image='imgs/a.png', prompt='changed'), {}, dataset_root=tmp_path)


def test_reference_capture_no_panel_copy(tmp_path):
    spec = importlib.util.spec_from_file_location('public_workflow', ROOT / 'tools/public/run.py')
    workflow = importlib.util.module_from_spec(spec); spec.loader.exec_module(workflow)
    reference_capture = workflow.reference_capture
    from common import sha
    capture = tmp_path / 'capture'; capture.mkdir()
    (capture / 'weight.pt').write_bytes(b'weight fixture')
    (capture / 'panel.pt').write_bytes(b'panel fixture')
    rows = [dict(name='layer', file='panel.pt', sha256=sha(capture / 'panel.pt'), scene_index=i) for i in range(16)]
    immutable(capture / 'manifest.json', dict(complete=True, model_identity={'test': 'identity'},
        weights={'layer': dict(file='weight.pt', sha256=sha(capture / 'weight.pt'))}, rows=rows))
    output = reference_capture(capture, tmp_path / 'view')
    view = json.loads((output / 'manifest.json').read_text())
    assert (output / view['weights']['layer']).resolve() == capture / 'weight.pt'
    assert len(list(output.iterdir())) == 1
    (capture / 'panel.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Changed capture'):
        reference_capture(capture, tmp_path / 'view2')


def test_calibration_dry_run_has_no_historical_paths(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'tools/public/run.py'), 'joint',
        '--model', 'nano', '--work', str(tmp_path / 'with spaces'), '--dry-run'],
        capture_output=True, text=True, check=True)
    assert '--controls' in result.stdout and '--budgets 1.25 1.5 2.0 2.5' in result.stdout
    assert 'ffn_followup' not in result.stdout


def test_prepare_native_manifest_cpu(tmp_path, monkeypatch):
    from common import sha
    spec = importlib.util.spec_from_file_location('public_workflow_native', ROOT / 'tools/public/run.py')
    workflow = importlib.util.module_from_spec(spec); spec.loader.exec_module(workflow)
    base = tmp_path / 'base'; base.mkdir(); (base / 'config.json').write_text('{}')
    monkeypatch.setattr(hub, 'registry', lambda: {'models': {'edge': {'base_required': {
        'identity': {'config.json': sha(base / 'config.json')}}}}})
    immutable(tmp_path / 'cohort.json', dict(model_name='edge', frozen=True, model=str(base),
        cases={'final': [dict(case={'id': 'fixture'}, seeds=[1, 2])]}))
    args = types.SimpleNamespace(work=tmp_path, model='edge', split='final',
        methods=['bf16', 'fp8', 'fp4_base'], candidates=None, manifest=tmp_path / 'evaluation.json')
    workflow.prepare_manifest(args)
    study = json.loads(args.manifest.read_text())
    assert study['methods']['fp8']['config']['attention_backend'] == 'bf16'
    base_config = study['methods']['fp4_base']['config']
    assert base_config['attention']['default']['coverage'] == 0
    assert base_config['ffn']['activation'] == base_config['ffn']['weight'] == 0
    assert study['steps'] == 35 and study['performance_eligible'] is False
    assert 'tools/public/run.py' in study['source_files']


def test_idle_guard_accepts_explicit_device_and_rejects_other_jobs(monkeypatch):
    from thunderworld_fp4.cosmos.cli import idle_guard
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: '')
    idle_guard()
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: '999999\n')
    with pytest.raises(RuntimeError, match='competing PIDs'):
        idle_guard()
