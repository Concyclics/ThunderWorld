"""Report mechanism evidence without promoting local errors to quality claims."""
import json
import statistics
from baselines import sha,write


def report(study):
    source=study/'mechanism.json'
    data=json.loads(source.read_text())
    groups={}
    for name,row in data['modules'].items():
        for method,values in row['objective'].items():
            for term,value in values.items():
                if term in row['invalid_policies'].get(method,[]):continue
                groups.setdefault((method,term),[]).append(value)
    rows=[dict(method=m,terms=t,module_count=len(values),mean_module_objective=statistics.mean(values))
          for (m,t),values in sorted(groups.items())]
    result=dict(complete=data['complete'],source_sha256=sha(source),rows=rows,
        performance_eligible=False,generation_quality_qualified=False,
        scope='Unweighted module summary of full-panel normalized errors; no final-output quality claim.')
    write(study/'report.json',result)
    lines=['# Contribution mechanism report','',result['scope'],'',
           '| Method | Terms | Modules | Mean local objective |','|---|---|---:|---:|']
    lines += [f"| {r['method']} | {r['terms']} | {r['module_count']} | {r['mean_module_objective']:.8g} |" for r in rows]
    (study/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))
