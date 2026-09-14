"""Require every Dense cell; retain actual OOM outcomes without imputation."""
from pathlib import Path
import collections,json,statistics
HERE=Path(__file__).resolve().parent
groups=collections.defaultdict(list); workers=[]; attempts=[]
for proc in (1,2,3):
    folder=HERE/'results'/f'process_{proc:02d}'
    meta=json.loads((folder/'metadata.json').read_text())
    done=json.loads((folder/'complete.json').read_text())
    check=json.loads((folder/'correctness.json').read_text())
    assert done['complete'] and done['cells']==18 and not meta['smoke']
    assert meta['allocator_budget_bytes']==28_000_000_000 and check['cached_recompute_equal']
    assert not any(meta[k] for k in ('offload','quantization','truncation','rope_extension'))
    rows=[json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines()]
    assert len(rows)==done['records']
    seen=set()
    for length in (32768,131072):
        workloads=json.loads((HERE/'workloads'/f'{length}.json').read_text())
        for wi in range(3):
            for phase in ('prefill','ttft','e2e'):
                cell=json.loads((folder/f'cell_{length}_{wi}_{phase}.json').read_text())
                rr=[r for r in rows if (r['source_tokens'],r['workload'],r['phase'])==(length,wi,phase)]
                assert rr and rr[0]['warmup'] and rr[0]['rep']==-1
                for r in rr:
                    ident=(proc,length,wi,phase,r['rep']); assert ident not in seen; seen.add(ident)
                    assert r['process']==proc and r['model_tokens']==length+513
                    assert r['selected_chunks'] is None and r['book_index']==workloads[wi]['book_index']
                    if r['status']=='ok':
                        assert 0<r['peak_allocated_bytes']<=r['peak_reserved_bytes']<=28_000_000_000
                        assert r['latency_ms']>0
                        assert r['output_tokens']==len(r['generated_ids'])==(128 if phase=='e2e' else 0)
                    else:
                        assert r['status']=='OOM' and r['latency_ms'] is None and r['error']
                if cell['status']=='ok':
                    assert [r['rep'] for r in rr]==[-1,0,1,2] and all(r['status']=='ok' for r in rr)
                else:
                    assert cell['status']=='OOM' and rr[-1]['status']=='OOM'
                groups[(length,phase)].append({'process':proc,'workload':wi,'status':cell['status'],
                    'records':[r for r in rr if not r['warmup'] and r['status']=='ok'],
                    'oom_attempts':sum(r['status']=='OOM' for r in rr)})
    assert len(seen)==len(rows)
    workers.append({'metadata':meta,'complete':done,'correctness':check}); attempts.extend(rows)
summary={}
for (length,phase),cells in groups.items():
    assert len(cells)==9
    failed=sum(c['status']=='OOM' for c in cells)
    item={'status':'OOM' if failed else 'ok','oom_cells':failed,'cells':9,
          'formal_successes':sum(len(c['records']) for c in cells),
          'oom_attempts':sum(c['oom_attempts'] for c in cells)}
    if failed==0:
        rr=[r for c in cells for r in c['records']]
        medians=[statistics.median(r['latency_ms'] for r in rr if r['process']==p) for p in (1,2,3)]
        item.update(latency_ms=statistics.median(medians),process_medians_ms=medians,
                    peak_GB=max(r['peak_allocated_bytes'] for r in rr)/1e9,
                    reserved_peak_GB=max(r['peak_reserved_bytes'] for r in rr)/1e9)
    summary.setdefault(str(length),{})[phase]=item
result={'complete':True,'expected_cells':54,'recorded_attempts':len(attempts),
        'formal_successes':sum(not r['warmup'] and r['status']=='ok' for r in attempts),
        'oom_attempts':sum(r['status']=='OOM' for r in attempts),'budget_GB':28,
        'budget_scope':'whole process CUDA allocator including weights, not incremental',
        'summary':summary,'workers':workers}
(HERE/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='workers'},indent=2))
