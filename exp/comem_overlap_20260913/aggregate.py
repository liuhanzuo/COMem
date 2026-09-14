import os
"""Validate complete paired outputs and report quality/cost without pseudoreplication."""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,json,re,string,statistics,zlib
import numpy as np
HERE=Path(__file__).resolve().parent
ARMS=('replay','w0','w32')
CELLS=('longeval_8k','longeval_16k','longeval_32k','qasper')
with gzip.open(HERE/'samples.jsonl.gz','rt',encoding='utf-8') as f:
    SAMPLES={r['id']:r for r in map(json.loads,f)}

def save(name,data):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def f1(pred,answer):
    def norm(s):
        s=s.lower().translate(str.maketrans('','',string.punctuation))
        return re.sub(r'\b(a|an|the)\b',' ',s).split()
    p,a=norm(pred),norm(answer)
    if not p or not a:return float(p==a)
    common=sum((Counter(p)&Counter(a)).values())
    return 2*common/(len(p)+len(a))

def rescore(sample,prediction):
    if sample['benchmark']=='longeval':
        m=re.search(r'\d{4,}',prediction)
        return float((m.group(0) if m else '')==sample['answers'][0])
    return max([f1(prediction,a) for a in sample['answers']] or [0.])

def diff_ci(records,ids,a,b,key):
    rng=np.random.default_rng(20260913+zlib.crc32(key.encode()))
    # Stratify LongEval macro by length; paired item resampling within each cell.
    groups=defaultdict(list)
    for i in ids:groups[SAMPLES[i]['cell']].append(100*(records[i,a]['score']-records[i,b]['score']))
    boots=np.zeros(20000)
    for vals in groups.values():
        v=np.array(vals);boots+=v[rng.integers(len(v),size=(len(boots),len(v)))].mean(1)/len(groups)
    return {'difference_pp':statistics.mean(statistics.mean(v) for v in groups.values()),
            'ci95_pp':np.quantile(boots,[.025,.975]).tolist(),'bootstrap_replicates':len(boots),
            'unit':'paired example; length-stratified for LongEval macro'}

def quality():
    dirs=[HERE/'quality'/f'shard_{i:02d}' for i in range(4)]
    if not all((d/'complete.json').exists() for d in dirs):return None
    records={};checks=[]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(os.environ['COMEM_MODEL'],local_files_only=True)
    for d in dirs:
        complete=json.loads((d/'complete.json').read_text())
        assert complete['complete'] and complete['samples']==125 and complete['records']==375
        checks.append(json.loads((d/'correctness.json').read_text()))
        for r in map(json.loads,(d/'predictions.jsonl').read_text(encoding='utf-8').splitlines()):
            key=r['id'],r['arm'];assert key not in records and r['status']=='ok'
            s=SAMPLES[r['id']];assert r['selected']==s['selected'] and r['cached_positions']==s['cached_positions']
            assert r['prediction']==tok.decode(r['generated_ids'],skip_special_tokens=True)
            assert abs(rescore(s,r['prediction'])-r['score'])<1e-12
            assert 1<=len(r['generated_ids'])<=s['budget']
            records[key]=r
    assert set(records)=={(i,a) for i in SAMPLES for a in ARMS}
    result={'complete':True,'samples':len(SAMPLES),'outputs':len(records),'cpu_rescore_equal':True,'cells':{},'exploratory_boundary':{},'correctness':checks}
    for cell in (*CELLS,'longeval_macro'):
        ids=[i for i,s in SAMPLES.items() if s['cell']==cell or cell=='longeval_macro' and s['benchmark']=='longeval']
        result['cells'][cell]={'n':len(ids),'metric':'F1' if cell=='qasper' else 'accuracy',
            'scores_percent':{a:statistics.mean(records[i,a]['score'] for i in ids)*100 for a in ARMS},
            'w32_minus_w0':diff_ci(records,ids,'w32','w0',cell+'w32w0'),
            'w32_minus_replay':diff_ci(records,ids,'w32','replay',cell+'w32replay')}
        if cell!='qasper':
            result['cells'][cell]['paired_wins_losses']={f'{a}_vs_{b}':{
                'wins':sum(records[i,a]['score']>records[i,b]['score'] for i in ids),
                'losses':sum(records[i,a]['score']<records[i,b]['score'] for i in ids)} for a,b in [('w32','w0'),('w32','replay')]}
        result['cells'][cell]['at_generation_limit']={a:sum(len(records[i,a]['generated_ids'])==SAMPLES[i]['budget'] for i in ids) for a in ARMS}
    for key,predicate in {
        'record_crosses_boundary':lambda s:s['target_crosses_chunk_boundary'],
        'record_does_not_cross_boundary':lambda s:not s['target_crosses_chunk_boundary'],
        'record_starts_in_first_32_positions':lambda s:s['target_boundary_distance']<32,
        'full_record_in_selected_or_query':lambda s:s['target_in_selected_or_query'],
        'full_record_added_only_by_overlap':lambda s:not s['target_in_selected_or_query'] and s['target_in_overlap_or_query'],
        'full_record_absent_even_after_overlap':lambda s:not s['target_in_overlap_or_query'],
    }.items():
        ids=[i for i,s in SAMPLES.items() if s['benchmark']=='longeval' and predicate(s)]
        result['exploratory_boundary'][key]={'n':len(ids),'scores_percent':{a:100*statistics.mean(records[i,a]['score'] for i in ids) if ids else None for a in ARMS}}
    save('quality_summary.json',result)
    return result,records

def cost(quality_records=None):
    dirs=[HERE/'cost'/f'process_{i:02d}' for i in (1,2,3)]
    if not all((d/'complete.json').exists() for d in dirs):return None
    rows=[];seen=set();timing_ids={i for i,s in SAMPLES.items() if s['timing_subset']}
    for d in dirs:
        complete=json.loads((d/'complete.json').read_text())
        assert complete['complete'] and complete['samples']==20 and complete['records_including_warmup']==240
        for r in map(json.loads,(d/'records.jsonl').read_text(encoding='utf-8').splitlines()):
            key=r['id'],r['arm'],r['process'],r['rep'];assert key not in seen;seen.add(key)
            assert r['id'] in timing_ids
            if r['status']=='ok':
                assert abs(rescore(SAMPLES[r['id']],r['prediction'])-r['score'])<1e-12
                assert len(r['generated_ids'])==128
                assert abs(r['e2e_ms']-r['prepare_ms']-r['online_ms'])<1e-6
                assert abs(r['online_ms']-r['ttft_ms']-r['decode_ms'])<1e-6
                assert abs(r['ttft_ms']-r['selection_ms']-r['fetch_ms']-r['query_write_prefill_ms'])<1e-6
                assert max(r['peak_allocated_bytes'],r['peak_reserved_bytes'])<=28_000_000_000
                assert r['selected']==SAMPLES[r['id']]['selected']
            if not r['warmup']:rows.append(r)
    assert seen=={(i,a,p,r) for i in timing_ids for a in ARMS for p in (1,2,3) for r in (-1,0,1,2)}
    out={'complete':True,'unique_samples':20,'formal_observations':len(rows),'processes':3,'cells':{},'nominal_maximum_bytes':28_000_000_000,'effective_allocator_cap_bytes':28e9/1024**3*1e9,'cap_note':'run_cost passed 28e9/1024**3 to gpu_gate, whose cap_gb is decimal GB. All processes therefore use a stricter approximately 26.077 GB cap, below the nominal 28 GB maximum. The runtime expression is unchanged across processes.','aggregation':'Latency: median of process medians; process median covers five documents and three repeats. Quality: per-item scores, with repeat disagreements disclosed. GPU: maximum over formal attempts. Persistent bytes: residual tensors or token IDs, excluding index/token duplicates.'}
    for cell in CELLS:
        result={}
        for a in ARMS:
            group=[r for r in rows if r['cell']==cell and r['arm']==a]
            assert len(group)==45
            if any(r['status']!='ok' for r in group):
                result[a]={'status':'OOM','failed_attempts':sum(r['status']!='ok' for r in group)};continue
            result[a]={'status':'ok','n_documents':5,'n_observations':45}
            for metric in ('prepare_ms','document_write_ms','selection_ms','fetch_ms','query_write_prefill_ms','ttft_ms','decode_ms','online_ms','e2e_ms'):
                process_values=[statistics.median(r[metric] for r in group if r['process']==p) for p in (1,2,3)]
                result[a][metric]={'median':statistics.median(process_values),'process_medians':process_values,'process_range':[min(process_values),max(process_values)]}
            result[a]['peak_allocated_GB']=max(r['peak_allocated_bytes'] for r in group)/1e9
            result[a]['peak_reserved_GB']=max(r['peak_reserved_bytes'] for r in group)/1e9
            result[a]['persistent_bytes_range']=[min(r['persistent_bytes'] for r in group),max(r['persistent_bytes'] for r in group)]
            peritem=defaultdict(list)
            for r in group:peritem[r['id']].append(r)
            result[a]['local_quality_percent']=100*statistics.mean(statistics.mean(r['score'] for r in rs) for rs in peritem.values())
            result[a]['score_disagreement_items']=sum(len({r['score'] for r in rs})>1 for rs in peritem.values())
            result[a]['generated_ids_disagreement_items']=sum(len({tuple(r['generated_ids']) for r in rs})>1 for rs in peritem.values())
            if quality_records:
                result[a]['b300_subset_quality_percent']=100*statistics.mean(quality_records[i,a]['score'] for i in peritem)
                result[a]['cross_device_prediction_disagreement_items']=sum(any(r['prediction']!=quality_records[i,a]['prediction'] for r in rs) for i,rs in peritem.items())
        if all(result[a]['status']=='ok' for a in ARMS):
            result['w32_over_w0']={m:result['w32'][m]['median']/result['w0'][m]['median'] for m in ('document_write_ms','ttft_ms','online_ms','e2e_ms')}
            for i in timing_ids:
                matching=[r for r in rows if r['id']==i and r['cell']==cell and r['arm'] in ('w0','w32')]
                if matching:assert len({r['persistent_bytes'] for r in matching})==1
        out['cells'][cell]=result
    save('cost_summary.json',out)
    return out

if __name__=='__main__':
    q=quality();c=cost(q[1] if q else None)
    if q: print(json.dumps({'quality':q[0]['cells'],'boundary':q[0]['exploratory_boundary']},indent=2))
    else:print('Quality incomplete; no final estimates emitted.')
    if c:print(json.dumps({'cost':c['cells']},indent=2))
    else:print('Cost incomplete; no final estimates emitted.')
