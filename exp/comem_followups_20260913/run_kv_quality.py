from pathlib import Path
import argparse,contextlib,gzip,json,os,platform,random,sys,time,gc
from common import *
from kv_control import KVControl,assemble,generate,checks
from transformers import AutoModelForCausalLM,AutoTokenizer
import transformers
METHODS=('replay','comem','chunkkv')
@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=4);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(42)
    out=HERE/('kv_smoke/job_'+os.environ['SLURM_JOB_ID'] if a.smoke else 'kv_quality')/f'shard_{a.shard:02d}';out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval();wrapper,model=attach(base,a.adapter)
    tok.bos_token_id=model.config.bos_token_id
    dump(out/'correctness.json',checks(model,tok))
    dump(out/'metadata.json',{'job':os.environ['SLURM_JOB_ID'],'node':platform.node(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':transformers.__version__,'methods':METHODS,'adapter_conditions':['on','off'],'adapter':'principal unchanged checkpoint, also applied to replay and KV for adaptation control','bos':tok.bos_token_id,'KV_control':'fixed set selected using layer1 K+V deviation, full layers0:2, r=.15 above, no async I/O','timing_claim':False})
    with gzip.open(HERE/'data/kv_samples.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(l) for l in f]
    rows=[next(r for r in rows if r['cell']==c) for c in ('longeval_8k','qasper')] if a.smoke else rows[a.shard::a.nshards]
    cm,rp=CoMem(model,12,tokenizer=tok),CoMem(model,0,tokenizer=tok);cb=KVControl(rp)
    rng=random.Random(20260913+a.shard);began=time.time();n=0
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            chunks,query=parts(row);ix=selection(chunks,row);arms=[(m,on) for m in METHODS for on in (False,True)];rng.shuffle(arms)
            for method,on in arms:
                with contextlib.nullcontext() if on else wrapper.disable_adapter():
                    if method=='chunkkv':
                        pack,merged=assemble(cb,[chunks[i] for i in ix],query,tok.bos_token_id);ids,_,_,stats=generate(cb,pack,merged,len(query),tok.eos_token_id,row['budget']);del pack,merged
                    else:
                        reader=rp if method=='replay' else cm;states=[reader.write_chunk(chunks[i]) for i in ix];sink=reader.write_chunk([tok.bos_token_id]);ids,_,_=decode(reader,sink,states,query,tok.eos_token_id,row['budget']);stats={};del states,sink
                prediction=tok.decode(ids,skip_special_tokens=True);rec={'id':row['id'],'cell':row['cell'],'method':method,'adapter_on':on,'generated_ids':ids,'prediction':prediction,'score':score(row,prediction),'selected':ix,'stats':stats,'status':'ok'}
                f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();n+=1
            dump(out/'progress.json',{'records':n,'samples':n//6,'total_samples':len(rows),'elapsed_s':time.time()-began});print(json.dumps({'records':n,'id':row['id']}),flush=True)
    dump(out/'complete.json',{'complete':True,'records':n,'elapsed_s':time.time()-began})
if __name__=='__main__':main()
