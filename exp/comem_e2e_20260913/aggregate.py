"""Aggregate the complete directly timed E2E requests, excluding smoke."""
from pathlib import Path
import collections, json, statistics
HERE=Path(__file__).resolve().parent
groups=collections.defaultdict(list)
metadata=[]
for proc in (1,2,3):
    folder=HERE/'results'/f'process_{proc:02d}'
    done=json.loads((folder/'complete.json').read_text())
    meta=json.loads((folder/'metadata.json').read_text())
    check=json.loads((folder/'correctness.json').read_text())
    assert done['complete'] and done['records']==36
    assert meta['process']==proc and not meta['smoke']
    assert meta['generation_tokens']==128 and meta['reps']==3 and meta['warmups']==1
    assert meta['document_write_included'] and meta['store_rebuilt_each_request']
    assert check['j0_top1_equal'] and all(c['equal'] for c in check['decode'].values())
    rows=[json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines()]
    expected={(l,w,arm,rep) for l in (32768,131072) for w in range(3)
              for arm in ('replay_k12','comem_k12') for rep in range(3)}
    assert len(rows)==len(expected)==36
    assert {(r['source_tokens'],r['workload'],r['arm'],r['rep']) for r in rows}==expected
    for row in rows:
        w=json.loads((HERE/'workloads'/f"{row['source_tokens']}.json").read_text())[row['workload']]
        assert row['process']==proc and row['output_tokens']==len(row['generated_ids'])==128
        assert row['decode_steps']==127 and row['pack_tokens']==6657
        assert row['selected_chunks']==w['selected']['12'] and row['book_index']==w['book_index']
        assert row['latency_ms']>0
        groups[(row['source_tokens'],row['arm'])].append(row)
    metadata.append(meta)
summary={}
for (length,arm),rows in groups.items():
    assert len(rows)==27
    medians=[statistics.median(r['latency_ms'] for r in rows if r['process']==p) for p in (1,2,3)]
    summary.setdefault(str(length),{})[arm]={'latency_ms':statistics.median(medians),
        'process_medians_ms':medians,'peak_GB':max(r['peak_allocated_bytes'] for r in rows)/1e9,
        'n':len(rows),'min_ms':min(r['latency_ms'] for r in rows),'max_ms':max(r['latency_ms'] for r in rows)}
result={'complete':True,'formal_requests':108,'output_tokens':128,'document_write_included':True,
    'aggregation':'median of three process medians; nine requests per process/cell',
    'summary':summary,'metadata':metadata}
(HERE/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
