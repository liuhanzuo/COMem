"""Validate completed B300 observations and summarize independent processes."""
from pathlib import Path
import json,statistics
HERE=Path(__file__).resolve().parent
ARMS=('dense_stock','dense_shared','replay_read','comem_read','replay_ttft','comem_ttft')
records=[]; metadata=[]
for process in (1,2,3):
    folder=HERE/'results'/f'process_{process:02d}'
    complete=json.loads((folder/'complete.json').read_text())
    rows=[json.loads(s) for s in (folder/'records.jsonl').read_text().splitlines()]
    assert complete['complete'] and complete['records']==len(rows)==54
    expected={(w,a,r) for w in range(3) for a in ARMS for r in range(3)}
    assert {(r['workload'],r['arm'],r['rep']) for r in rows}==expected
    assert all(r['source_tokens']==131072 and r['process']==process and r['latency_ms']>0 and r['peak_allocated_bytes']>=r['baseline_allocated_bytes'] for r in rows)
    assert all(r['pack_tokens']==(131585 if r['arm'].startswith('dense') else 6657) for r in rows)
    check=json.loads((folder/'correctness.json').read_text())
    assert check['same_top1'] and check['max_abs_logit_difference']<=.125
    m=json.loads((folder/'metadata.json').read_text())
    assert m['compute_capability']==[10,3] and not m['args']['smoke']
    metadata.append(m); records.extend(rows)
for m in metadata[1:]:
    for k in ('gpu_display_name','torch','transformers','cuda','adapter_config','sdpa','adapter_parameter_count'):
        assert m[k]==metadata[0][k],k
workloads=json.loads((HERE/'workloads.json').read_text())
assert all(r['selected_chunks']==workloads[r['workload']]['selected']['12'] for r in records if not r['arm'].startswith('dense'))
summary={}
for arm in ARMS:
    rows=[r for r in records if r['arm']==arm]
    medians=[statistics.median(r['latency_ms'] for r in rows if r['process']==p) for p in (1,2,3)]
    summary[arm]={'latency_ms':statistics.median(medians),'process_medians_ms':medians,'peak_GB':max(r['peak_allocated_bytes'] for r in rows)/1e9,'n':len(rows)}
speedups={'same_adapter_full_vs_comem_read':summary['dense_shared']['latency_ms']/summary['comem_read']['latency_ms'],'stock_full_vs_comem_read':summary['dense_stock']['latency_ms']/summary['comem_read']['latency_ms'],'same_pack_depth_read':summary['replay_read']['latency_ms']/summary['comem_read']['latency_ms'],'same_pack_ttft':summary['replay_ttft']['latency_ms']/summary['comem_ttft']['latency_ms']}
(HERE/'summary.json').write_text(json.dumps({'record_count':len(records),'summary':summary,'speedups':speedups,'metadata':metadata},indent=2),encoding='utf-8')
print(json.dumps({'summary':summary,'speedups':speedups},indent=2))
