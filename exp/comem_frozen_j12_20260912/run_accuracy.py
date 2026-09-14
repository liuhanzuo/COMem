"""Frozen j=12 accuracy remeasurement using the released CoMem prompt path.

Synthetic examples are sampled independently from the adapted overview row.
No adapter, quantization, positional extension, or chat template is applied.
"""
from pathlib import Path
import argparse, collections, gzip, json, os, platform, random, sys, time, zlib
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from comem import CoMem
from comem import selectors
from eval import ruler, longeval, longbench
import prepare_babilong as babi

LENGTHS = ('8k','16k','32k','64k','128k')
RULER_TASKS = ('niah_single_2','niah_multikey_1','variable_tracking')
LB_TASKS = ('narrativeqa','qasper','hotpotqa','2wikimqa','multifieldqa_en','musique')
BABI_TASKS = ('qa1','qa2','qa5')
BABI_LENGTHS = ('0k','1k','2k','4k','8k','16k','32k')

def dump(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)

def cells():
    # Shorter natural-task cells yield useful complete results early.
    for task in LB_TASKS: yield 'longbench',task,''
    for length in BABI_LENGTHS:
        for task in BABI_TASKS: yield 'babilong',task,length
    for length in LENGTHS: yield 'longeval','lines',length
    for length in LENGTHS:
        for task in RULER_TASKS: yield 'ruler',task,length

def samples(benchmark,task,length,tok,shard,nshards):
    if benchmark=='longbench':
        rows=[json.loads(line) for line in (ROOT/'data/longbench'/f'{task}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        assert len(rows)==(150 if task=='multifieldqa_en' else 200)
        for i in range(shard,len(rows),nshards):
            row=rows[i]
            yield i,longbench.format_prompt(row,task),row['input'].strip(),row['answers'],longbench.DATASET2MAXGEN[task],{'dataset_index':i}
    elif benchmark=='babilong':
        rows=babi.load_task(task,length)
        assert len(rows)==100
        for i in range(shard,100,nshards):
            row=rows[i]
            assert babi.score_prediction(row['target'],row,task)==1
            yield i,babi.format_prompt(row,task),row['question'].strip(),[row['target']],20,{'question':row['question'],'target':row['target']}
    elif benchmark=='longeval':
        seed=1234+(zlib.crc32(length.encode())%100000)
        for i in range(shard,100,nshards):
            prompt,expected,label,nlines=longeval.build_lines_prompt(longeval._LENGTH_TOKENS[length],tok,random.Random(seed*1000+i))
            # No-adapter LongEval generation budget: 48 tokens.
            yield i,prompt,f'line {label}',[expected],48,{'seed':seed*1000+i,'label':label,'n_lines':nlines}
    else:
        assert os.environ.get('PYTHONHASHSEED')=='0'
        seed=42+(hash((task,length))%100000)
        icl=ruler._make_vt_icl(random.Random(seed+777),4) if task=='variable_tracking' else None
        for i in range(shard,100,nshards):
            prompt,answers,_=ruler._build_sample(task,ruler._LENGTH_TOKENS[length],tok,random.Random(seed*1000+i),icl)
            yield i,prompt,ruler._bare_question(prompt),answers,60 if task=='variable_tracking' else 48,{'seed':seed*1000+i,'haystack':'pg19_essay.txt' if task!='variable_tracking' else 'RULER repeated noise'}

def score(benchmark,task,pred,answers,extra):
    if benchmark=='longbench': return longbench.compute_f1_multi(pred,answers)
    if benchmark=='babilong': return babi.score_prediction(pred,extra,task)
    if benchmark=='longeval': return float(longeval.extract_prediction(pred)==answers[0])
    return ruler._string_match_all_one(pred,answers)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--nshards',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2); torch.set_num_interop_threads(4); torch.manual_seed(42)
    out=ROOT/('smoke/job_'+os.environ['SLURM_JOB_ID'] if a.smoke else 'results')/f'shard_{a.shard:02d}'
    out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    tok.pad_token=tok.eos_token
    tok.model_max_length=10**9
    ruler._ESSAY_PATH=str(ROOT/'data/pg19_essay.txt')
    assert Path(ruler._ESSAY_PATH).stat().st_size>1_000_000
    assert ruler._load_essay_words(), 'No silent noise fallback'
    dev=torch.cuda.get_device_properties(0)
    meta={'job':os.environ['SLURM_JOB_ID'],'array_task':os.environ.get('SLURM_ARRAY_TASK_ID'),'host':platform.node(),'gpu_name':dev.name,'compute_capability':[dev.major,dev.minor],'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'model':a.model,'adapter':None,'dtype':'bfloat16','split':12,'chunk_size':512,'topk':12,'selector':'iter_bm25','hop_topk':4,'iter_rounds':0,'chat_template':False,'rope_extension':False,'query_boundary':'last chunk of the full formatted prompt, released CoMem path','shard':a.shard,'nshards':a.nshards,'fresh_evaluation':True,'synthetic_cohort':'independent from the adapted overview row','ruler_seed':42,'PYTHONHASHSEED':os.environ.get('PYTHONHASHSEED'),'longeval_seed':1234,'smoke':a.smoke}
    dump(out/'metadata.json',meta)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    assert not any('lora_' in name for name,_ in model.named_parameters())
    assert all(param.device.type=='cuda' and param.dtype==torch.bfloat16 for param in model.parameters())
    cm=CoMem(model,12,tokenizer=tok); rp=CoMem(model,0,tokenizer=tok)
    check_ids=torch.tensor([tok.encode('A short passage contains a name and a number. The name is Alice and the number is 42.',add_special_tokens=True)],device='cuda')
    stock=model(input_ids=check_ids,use_cache=True,logits_to_keep=1).logits
    split=rp.read_prefill(None,[],rp.write_chunk(check_ids))[0]
    checks={'j0_max_abs_difference':float((stock-split).abs().max()),'j0_top1_equal':bool(stock.argmax(-1).eq(split.argmax(-1)).all())}
    assert checks['j0_top1_equal'] and checks['j0_max_abs_difference']<=.125
    prompt=('Alice stores the number 42 in a notebook. Bob stores the number 73. '*30)+' What number does Alice store? Answer:'
    ids=torch.tensor([tok.encode(prompt,add_special_tokens=True)],device='cuda')
    kw={'chunk_size':512,'selector':'iter_bm25','topk':12,'iter_hop_topk':4,'iter_rounds':0,'sink_tokens':'bos','bare_question_ids':tok.encode('Alice number',add_special_tokens=False),'max_new_tokens':8}
    cached=cm.generate_from_ids(ids,use_kv_cache=True,**kw)
    recompute=cm.generate_from_ids(ids,use_kv_cache=False,**kw)
    checks.update({'j12_cached_output':cached,'j12_recompute_output':recompute,'j12_outputs_equal':cached==recompute})
    dump(out/'correctness.json',checks)
    assert cached==recompute, checks
    del stock,split,ids,check_ids
    all_rows=[]; started=time.time()
    selected_cells=list(cells())
    if a.smoke: selected_cells=[('longbench','qasper',''),('babilong','qa1','0k'),('longeval','lines','8k'),('ruler','niah_single_2','8k'),('ruler','variable_tracking','8k')]
    for benchmark,task,length in selected_cells:
        name='_'.join(x for x in (benchmark,task,length) if x)
        cell_rows=[]
        with gzip.open(out/(name+'.inputs.jsonl.gz'),'wt',encoding='utf-8',compresslevel=3) as inputs, (out/(name+'.predictions.jsonl')).open('w',encoding='utf-8') as predictions:
            for index,prompt,question,answers,maxgen,extra in samples(benchmark,task,length,tok,a.shard,a.nshards):
                ids_cpu=tok.encode(prompt,add_special_tokens=True)
                ids=torch.tensor([ids_cpu],device='cuda',dtype=torch.long)
                question_ids=tok.encode(question,add_special_tokens=False)
                chunks=list(torch.tensor(ids_cpu,dtype=torch.long).split(512))
                selected=selectors.iter_bm25_indices(chunks[:-1],question_ids,12,iter_hop_topk=4,iter_rounds=0)
                record={'benchmark':benchmark,'task':task,'length':length,'index':index,'source_tokens':len(ids_cpu),'selected_chunks':selected,'read_pack_tokens':1+sum(len(chunks[i]) for i in selected)+len(chunks[-1]),'max_new_tokens':maxgen,'answers':answers,**extra}
                inputs.write(json.dumps({**record,'input_ids':ids_cpu,'question_ids':question_ids},ensure_ascii=False)+'\n')
                t0=time.perf_counter()
                pred=cm.generate_from_ids(ids,chunk_size=512,max_new_tokens=maxgen,selector='iter_bm25',topk=12,sink_tokens='bos',bare_question_ids=question_ids,iter_rounds=0,iter_hop_topk=4,use_kv_cache=True)
                row={**record,'prediction':pred,'score':score(benchmark,task,pred,answers,extra),'wall_seconds':time.perf_counter()-t0,'status':'ok'}
                assert 0<=row['score']<=1
                predictions.write(json.dumps(row,ensure_ascii=False)+'\n'); predictions.flush()
                cell_rows.append(row); all_rows.append(row)
                dump(out/'progress.json',{'elapsed_seconds':time.time()-started,'records':len(all_rows),'cell':name,'cell_records':len(cell_rows),'last_index':index})
                if len(cell_rows)%5==0 or a.smoke: print(json.dumps({'cell':name,'n':len(cell_rows),'last_seconds':row['wall_seconds'],'score':row['score']}),flush=True)
                del ids,chunks
                if a.smoke: break
        dump(out/(name+'.summary.json'),{'benchmark':benchmark,'task':task,'length':length,'n':len(cell_rows),'score':100*sum(r['score'] for r in cell_rows)/len(cell_rows)})
    dump(out/'complete.json',{'complete':True,'records':len(all_rows),'elapsed_seconds':time.time()-started,'cells':len(selected_cells)})

if __name__=='__main__': main()
