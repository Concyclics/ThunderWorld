"""Consume the frozen final split only after independent validation passes."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import time

from baselines import sha, write
from evidence_validation import audit, report
from importance_workflow import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'workspace', 'cohort', 'archive'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    validation = a.root/'validation'
    while True:
        path = validation/'decision.json'
        if path.exists() and json.loads(path.read_text()).get('complete'):
            break
        status = validation/'workflow.json'
        if status.exists() and json.loads(status.read_text()).get('paused'):
            raise RuntimeError('Validation paused; final split must remain unused')
        time.sleep(5)
    decision = json.loads(path.read_text())
    quality = validation/'report/quality.json'
    if decision['quality_sha256'] != sha(quality):
        raise ValueError('Changed validation decision')
    selected = decision['final_eligible']
    root = a.root/'final'
    root.mkdir(parents=True, exist_ok=True)
    status = root/'orchestrator.json'
    identity = dict(validation_decision_sha256=sha(path), cohort_sha256=sha(a.cohort), source_sha256=sha(__file__),
                    report_implementation_sha256=sha(Path(__file__).with_name('evidence_validation.py')))
    state = dict(complete=False, identity=identity, performance_eligible=False)
    if status.exists():
        old = json.loads(status.read_text())
        if old['identity'] != identity:
            raise ValueError('Changed final study identity')
        if old['complete']:
            return
    if not selected:
        state.update(complete=True, skipped=True, final_consumed=False,
            reason='No representation candidate passed independent validation; stop rather than search on final outputs.')
        write(status, state)
        a.archive.mkdir(parents=True, exist_ok=True)
        write(a.archive/'final_decision.json', state)
        return
    cohort = json.loads(a.cohort.read_text())
    previous = json.loads((validation/'manifest.json').read_text())
    pairs = [p for p in previous['primary_comparisons'] if p['candidate'] in selected]
    names = list(dict.fromkeys(['bf16', 'fp8', *[p['reference'] for p in pairs], *selected]))
    manifest = copy.deepcopy(previous)
    manifest.update(role='final', cases=cohort['cases']['final'], selection_eligible=False,
        methods={name: previous['methods'][name] for name in names}, primary_comparisons=pairs,
        final_identity=identity,
        scope='72 reserved scenes, 144 paired videos/method. Assets/support frozen after validation. '
              'Four/Six remains a nearest-neighbor representation control, not evidence for an outlier contribution.')
    file = root/'manifest.json'
    if file.exists():
        if json.loads(file.read_text()) != manifest:
            raise ValueError('Changed frozen final manifest')
    else:
        write(root/'audit.json', audit(cohort, a.workspace, 'final'))
        write(file, manifest)
    write(status, state)
    try:
        run(file, root)
        measured = report(root)
        measured['matched_control_passed'] = measured.pop('final_eligible')
        measured['exceeds_fp8'] = [name for name in selected if all(r['passed'] for r in measured['contrasts']
                                    if r['candidate'] == name and r['reference'] == 'fp8')]
        measured['scope'] = manifest['scope']+' Scene-clustered/category-stratified paired intervals; no final-set fitting. No speed claim.'
        write(root/'report/quality.json', measured)
        text = (root/'report/REPORT.md').read_text()
        text = text.replace('进入独立最终测试的候选：', '通过独立最终集同预算对照门槛的候选：')
        text = text.replace('Only independent final data can confirm a selected validation result.', 'This is the independent final comparison.')
        (root/'report/REPORT.md').write_text(text)
        archive = a.archive/'final'
        archive.mkdir(parents=True, exist_ok=True)
        for name, source in (('manifest', file), ('audit', root/'audit.json'), ('quality', root/'report/quality.json')):
            (archive/(name+'.json.gz')).write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        (archive/'REPORT.md').write_bytes((root/'report/REPORT.md').read_bytes())
        state.update(complete=True, final_consumed=True, matched_control_passed=measured['matched_control_passed'],
            exceeds_fp8=measured['exceeds_fp8'], quality_sha256=sha(root/'report/quality.json'))
        write(status, state)
        write(a.archive/'final_decision.json', state)
    except Exception as error:
        state.update(paused=True, error=repr(error))
        write(status, state)
        raise


if __name__ == '__main__':
    main()
