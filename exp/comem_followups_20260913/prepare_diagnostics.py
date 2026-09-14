import os
"""Unseen PG19 validation books, two deterministic non-overlapping windows/book."""
from pathlib import Path
import concurrent.futures,gzip,json,random,requests
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
def main():
    raw=HERE/'data/pg19_validation';raw.mkdir(parents=True,exist_ok=True)
    r=requests.get('https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o',params={'prefix':'validation/','maxResults':1000},timeout=60);r.raise_for_status()
    objects=[x for x in r.json()['items'] if x['name'].endswith('.txt')]
    assert 'nextPageToken' not in r.json()
    train=json.loads(Path(os.environ['COMEM_TRAIN_MANIFEST']).read_text())
    assert not ({Path(x['name']).name for x in train['objects']} & {Path(x['name']).name for x in objects})
    def fetch(obj):
        p=raw/Path(obj['name']).name
        if not p.exists():
            res=requests.get('https://storage.googleapis.com/deepmind-gutenberg/'+obj['name'],timeout=60);res.raise_for_status();p.write_bytes(res.content)
        assert p.stat().st_size==int(obj['size'])
        return obj,p.read_text(encoding='utf-8')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: books=list(pool.map(fetch,objects))
    tok=AutoTokenizer.from_pretrained(os.environ['COMEM_MODEL'],local_files_only=True);tok.model_max_length=10**9
    rng=random.Random(20260913);rows=[]
    for obj,txt in books:
        ids=tok.encode(txt,add_special_tokens=False)
        # Whole 4097-token blocks supply 4096 inputs plus the last next-token label.
        blocks=list(range(len(ids)//4097));rng.shuffle(blocks)
        for k,b in enumerate(blocks[:2]):rows.append({'id':obj['name']+f':{b}','book':obj['name'],'token_start':b*4097,'input_and_next_ids':ids[b*4097:(b+1)*4097]})
    with gzip.open(HERE/'data/distillation_windows.jsonl.gz','wt',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    meta={'source':'gs://deepmind-gutenberg/validation/','books':len(books),'windows':len(rows),'window_inputs':4096,'query_positions':512,'seed':20260913,'selection':'two random non-overlapping 4097-token blocks per official validation book, no selection on model outputs','training_book_ids_disjoint':True,'objects':[{'name':x['name'],'size':x['size']} for x in objects]}
    (HERE/'data/distillation_samples.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in meta.items() if k!='objects'}),flush=True)
if __name__=='__main__':main()
