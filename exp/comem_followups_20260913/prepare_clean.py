import os
"""Full official six-task support; exact released last-chunk prompt convention."""
from pathlib import Path
import gzip,json,sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from transformers import AutoTokenizer
from comem.selectors import iter_bm25_indices
from eval import longbench
import torch
TASKS=('narrativeqa','qasper','hotpotqa','2wikimqa','multifieldqa_en','musique')
def main():
    tok=AutoTokenizer.from_pretrained(os.environ['COMEM_MODEL'],local_files_only=True);tok.model_max_length=10**9
    (HERE/'data').mkdir(parents=True,exist_ok=True)
    keep=json.loads((HERE/'reference/clean_subset_ids.json').read_text())['longbench']['by_task'];counts={}
    with gzip.open(HERE/'data/clean_samples.jsonl.gz','wt',encoding='utf-8') as out:
        for task in TASKS:
            data=HERE.parent/'comem_frozen_j12_20260912/data/longbench'/f'{task}.jsonl'
            rows=[json.loads(l) for l in data.read_text(encoding='utf-8').splitlines()]
            inds=set(keep[task]['indices']);ids=set(keep[task]['ids']);n=0
            for index,row in enumerate(rows):
                uid=row['_id'];clean=index in inds;assert clean==(uid in ids),(task,index,uid)
                prompt=longbench.format_prompt(row,task);tokens=tok.encode(prompt,add_special_tokens=True)
                chunks=list(torch.tensor(tokens).split(512));question=tok.encode(row['input'].strip(),add_special_tokens=False)
                selected=iter_bm25_indices(chunks[:-1],question,12,iter_hop_topk=4,iter_rounds=0)
                record={'id':f'{task}:{index}','task':task,'index':index,'dataset_id':uid,'clean':clean,'input_ids':tokens,'question_ids':question,'selected':selected,'answers':row['answers'],'budget':longbench.DATASET2MAXGEN[task]}
                out.write(json.dumps(record,ensure_ascii=False)+'\n');n+=clean
            counts[task]={'full':len(rows),'clean':n,'removed':len(rows)-n};print(task,counts[task],flush=True)
    (HERE/'data/clean_samples.json').write_text(json.dumps({'tasks':counts,'filter':'same published normalized 13gram sketch; strict containment<.10 against full PG19 train; independent of method outputs','paired_method_ids':True,'row_index_and_dataset_id_membership_equal':True,'protocol':'released last-chunk query, default BOS fallback first token, no chat, no YaRN'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
