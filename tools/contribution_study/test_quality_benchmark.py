import json
from pathlib import Path
from unittest.mock import patch

import pytest

from baselines import sha, write
from dataset_infer import run, selected_cases, source_hashes, validate_rows
from quality_benchmark import copy_checked, persist_bank


def test_fresh_seeds_do_not_need_or_invent_historical_match(tmp_path):
    cases=[dict(case=dict(id='a'),seeds=[101,102])]
    assert source_hashes({},'method',cases)=={}
    with pytest.raises(ValueError,match='historical hash absent'):
        source_hashes({},'method',cases,True)
    hashes=tmp_path/'hashes.json';write(hashes,{'method':{'a':{'101':{'video':'abc'}}}})
    assert source_hashes({'source_output_hashes':str(hashes)},'method',cases)['a']['101']=={'video':'abc'}
    with pytest.raises(ValueError,match='historical hash absent'):
        source_hashes({'source_output_hashes':str(hashes)},'method',cases,True)


def test_case_validation_rejects_changed_inputs_duplicate_seeds_and_unknown_selection(tmp_path):
    image=tmp_path/'input.png';image.write_bytes(b'fixture')
    case=dict(case=dict(id='a',image=str(image)),image_sha256=sha(image),seeds=[1])
    assert selected_cases({'cases':[case]})==[case]
    with pytest.raises(ValueError,match='Unknown case'):
        selected_cases({'cases':[case]},['absent'])
    case['seeds']=[1,1]
    with pytest.raises(ValueError,match='duplicate seed'):
        selected_cases({'cases':[case]})
    case['seeds']=[1];image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed input image'):
        selected_cases({'cases':[case]})


def test_complete_receipt_cannot_hide_missing_or_corrupt_outputs(tmp_path):
    output=tmp_path/'seed1.npz';output.write_bytes(b'fixture')
    identity={'seeds':[1,2]}
    result=dict(identity=identity,complete=True,rows=[dict(seed=1,outputs=output.name,outputs_sha256=sha(output))])
    with pytest.raises(ValueError,match='missing seeds'):
        validate_rows(result,tmp_path,identity)
    result['complete']=False
    validate_rows(result,tmp_path,identity)
    output.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt resumed output'):
        validate_rows(result,tmp_path,identity)


def test_persisted_bank_copies_exact_bytes_and_checks_every_tensor(tmp_path):
    src=tmp_path/'src';src.mkdir();(src/'weight.pt').write_bytes(b'weightbytes')
    write(src/'manifest.json',dict(complete=True,modules={'linear':dict(file='weight.pt',sha256=sha(src/'weight.pt'))}))
    dst=tmp_path/'dst';persist_bank(src,dst,sha(src/'manifest.json'))
    assert sha(src/'manifest.json')==sha(dst/'manifest.json')
    assert (dst/'weight.pt').read_bytes()==b'weightbytes'
    (dst/'weight.pt').write_bytes(b'bad')
    with pytest.raises(ValueError,match='Changed persisted asset'):
        persist_bank(src,dst,sha(src/'manifest.json'))


def test_worker_complete_resume_is_cpu_only_and_still_checks_bytes(tmp_path):
    image=tmp_path/'img';image.write_bytes(b'image')
    cfg=dict(debug='off',resource_mode='shared-calibration',precision='bf16',attention_backend='bf16')
    manifest=tmp_path/'manifest.json'
    entry=dict(case=dict(id='a',image=str(image)),seeds=[1],image_sha256=sha(image))
    write(manifest,dict(frozen=True,steps=35,methods={'bf16':dict(config=cfg)},cases=[entry]))
    import dataset_infer
    identity=dict(manifest_sha256=sha(manifest),method='bf16',source_sha256=sha(dataset_infer.__file__),
        runtime_sha256=sha(Path(dataset_infer.__file__).with_name('baseline_runtime.py')),smoke=False,
        cases=['a'],require_source_hash=False,source_output_hashes_sha256=None)
    output=tmp_path/'outputs';folder=output/'a';folder.mkdir(parents=True)
    (folder/'seed1.npz').write_bytes(b'generated')
    ci=dict(**identity,case=entry['case'],image_sha256=entry['image_sha256'],seeds=[1])
    write(folder/'result.json',dict(identity=ci,complete=True,rows=[dict(seed=1,outputs='seed1.npz',outputs_sha256=sha(folder/'seed1.npz'))]))
    with patch('baseline_runtime.make_runner',side_effect=AssertionError('Must not load a model')):
        assert run(manifest,'bf16',output)['complete']
    (folder/'seed1.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt resumed output'):
        run(manifest,'bf16',output)
