import argparse,gzip,json,os,platform,random,time,gc
from common import *
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=4);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.set_num_interop_threads(4);torch.manual_seed(42)
    out=HERE/('smoke/job_'+os.environ['SLURM_JOB_ID'] if a.smoke else 'quality')/f'shard_{a.shard:02d}'
    out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    wrapper,model=attach(base,a.adapter)
    dump(out/'metadata.json',{'job':os.environ['SLURM_JOB_ID'],'node':platform.node(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':transformers.__version__,'adapter':a.adapter,'shard':a.shard,'nshards':a.nshards,'timing_claim':False,'protocol':'PROTOCOL_zh.md'})
    dump(out/'correctness.json',check_paths(wrapper,model,tok))
    with gzip.open(HERE/'samples.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(s) for s in f]
    if a.smoke:rows=[next(r for r in rows if r['cell']=='longeval_8k'),next(r for r in rows if r['cell']=='qasper')]
    else:rows=rows[a.shard::a.nshards]
    cm,rp=CoMem(model,12,tokenizer=tok),CoMem(model,0,tokenizer=tok)
    rng=random.Random(20260913+a.shard);started=time.time();n=0
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            chunks,query=parts(row);ix=selection(chunks,row);source=torch.cat(chunks)
            arms=list(ARMS);rng.shuffle(arms)
            for arm in arms:
                reader=rp if arm=='replay' else cm
                states=[reader.write_chunk(chunks[i]) if arm=='replay' else write_one(cm,source,i,32 if arm=='w32' else 0) for i in ix]
                bos,_=reader._bos_eos(tok,fallback_first=row['input_ids'][0])
                sink=reader.write_chunk([bos])
                generated,_,_=decode(reader,sink,states,query,tok.eos_token_id,row['budget'])
                text=tok.decode(generated,skip_special_tokens=True)
                record={'id':row['id'],'cell':row['cell'],'arm':arm,'generated_ids':generated,'prediction':text,'score':score(row,text),'selected':ix,'cached_positions':sum(h.shape[1] for h in states),'status':'ok'}
                f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();n+=1
                del states,sink
            dump(out/'progress.json',{'records':n,'samples':n//3,'last_id':row['id'],'elapsed_s':time.time()-started})
            if n%15==0 or a.smoke:print(json.dumps({'records':n,'last_id':row['id'],'elapsed_s':time.time()-started}),flush=True)
    dump(out/'complete.json',{'complete':True,'samples':len(rows),'records':n,'elapsed_s':time.time()-started})
if __name__=='__main__':main()
