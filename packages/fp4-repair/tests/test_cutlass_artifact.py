"""CPU checks for stale artifacts and accidental backend/profile substitution."""
from dataclasses import replace
import hashlib
import json
import pytest
from thunderworld_fp4.cosmos.settings import RunConfig
from thunderworld_fp4.cutlass.artifact import verify_library
from thunderworld_fp4.cutlass.build import source_hashes


def test_backend_requires_explicit_compatible_library():
    baseline=RunConfig()
    with pytest.raises(ValueError):replace(baseline,linear_backend='cutlass-fused')
    with pytest.raises(ValueError):replace(baseline,cutlass_library='test.so')
    with pytest.raises(ValueError):replace(baseline,cutlass_small_library='small.so')
    config=replace(baseline,linear_backend='cutlass-fused',cutlass_library='test.so')
    assert config.fingerprint!=baseline.fingerprint
    assert RunConfig.from_dict(config.to_dict())==config
    direct=replace(config,cutlass_direct_pack=True)
    assert direct.fingerprint!=config.fingerprint
    assert RunConfig.from_dict(direct.to_dict())==direct
    with pytest.raises(ValueError):replace(baseline,cutlass_direct_pack=True)
    with pytest.raises(ValueError):replace(direct,share_pack=True)
    with pytest.raises(ValueError):replace(config,kernel_variant='legacy')
    with pytest.raises(ValueError):replace(config,precision='fp8',attention_backend='bf16')


def test_artifact_rejects_modified_binary_and_header(tmp_path):
    library=tmp_path/'kernel.so';library.write_bytes(b'unit test artifact')
    metadata=dict(schema_version=1,backend='cutlass-fused-sm120',sources=source_hashes(),tile=[128,128,128],
                  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    manifest=library.with_suffix('.build.json');manifest.write_text(json.dumps(metadata))
    assert verify_library(library)['library_sha256']==metadata['library_sha256']
    library.write_bytes(b'changed')
    with pytest.raises(ValueError,match='digest'):verify_library(library)
    metadata['library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
    metadata['sources']['repair_collective.cuh']='changed';manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='Stale'):verify_library(library)
