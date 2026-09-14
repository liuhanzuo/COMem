import os
from pathlib import Path
import gzip,json,random,zlib,collections
from common import HERE,dump,parts,selection
from comem.selectors import iter_bm25_indices
from eval import longeval,longbench
from transformers import AutoTokenizer

def main():
    tok=AutoTokenizer.from_pretrained(os.environ['COMEM_MODEL'],local_files_only=True)
    tok.model_max_length=10**9
    out=HERE/'samples.jsonl.gz'
    assert not out.exists(),'Do not replace already selected confirmation samples'
    rows=[]
    for length in ('8k','16k','32k'):
        seed=20260913+zlib.crc32(length.encode())%100000
        for i in range(100):
            prompt,answer,label,nlines=longeval.build_lines_prompt(longeval._LENGTH_TOKENS[length],tok,random.Random(seed*1000+i))
            record=f'line {label}: REGISTER_CONTENT is <{answer}>'
            start=prompt.index(record)
            encoded=tok(prompt,add_special_tokens=True,return_offsets_mapping=True)
            positions=[j for j,(a,b) in enumerate(encoded['offset_mapping']) if b>start and a<start+len(record)]
            rows.append({'id':f'longeval_{length}_{i:03d}','benchmark':'longeval','cell':f'longeval_{length}','index':i,'length':length,'input_ids':encoded['input_ids'],'question_ids':tok.encode(f'line {label}',add_special_tokens=False),'answers':[answer],'budget':16,'seed':seed*1000+i,'target_label':label,'target_span':[min(positions),max(positions)+1],'target_boundary_distance':min(positions)%512,'n_lines':nlines,'timing_subset':i<5})
        print(length,'prepared',flush=True)
    path=HERE.parent/'comem_frozen_j12_20260912/data/longbench/qasper.jsonl'
    data=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    assert len(data)==200
    for i,r in enumerate(data):
        prompt=longbench.format_prompt(r,'qasper')
        rows.append({'id':f'qasper_{i:03d}','benchmark':'qasper','cell':'qasper','index':i,'input_ids':tok.encode(prompt,add_special_tokens=True),'question_ids':tok.encode(r['input'].strip(),add_special_tokens=False),'answers':r['answers'],'budget':128,'timing_subset':i<5})
    for row in rows:
        chunks,query=parts(row)
        row['selected']=iter_bm25_indices(chunks,row['question_ids'],12,iter_hop_topk=4,iter_rounds=0)
        row['source_positions']=sum(len(c) for c in chunks)
        row['query_positions']=len(query)
        row['cached_positions']=sum(len(chunks[i]) for i in row['selected'])
        if 'target_span' in row:
            a,b=row['target_span'];source_end=row['source_positions']
            coverage=set()
            for i in row['selected']:coverage.update(range(i*512,(i+1)*512))
            row['target_in_selected_or_query']=all(p in coverage or p>=source_end for p in range(a,b))
            expanded=set(coverage)
            for i in row['selected']:expanded.update(range(max(0,i*512-32),i*512))
            row['target_in_overlap_or_query']=all(p in expanded or p>=source_end for p in range(a,b))
            row['target_crosses_chunk_boundary']=(a//512)!=(b-1)//512
    with gzip.open(out,'wt',encoding='utf-8',compresslevel=3) as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    dump(HERE/'sample_summary.json',{'n':len(rows),'cells':dict(collections.Counter(r['cell'] for r in rows)),'timing_ids':[r['id'] for r in rows if r['timing_subset']],'long_eval_seed':20260913,'quality_budget':{'longeval':16,'qasper':128},'cost_generation_tokens':128})
    print('Prepared',len(rows),'fixed samples')
if __name__=='__main__':main()
