"""Re-evaluate full and clean support together; filtering occurs only in scoring."""
from common import *
from transformers import AutoModelForCausalLM,AutoTokenizer
import argparse,contextlib,gzip,os,platform,random,transformers
@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--nshards',type=int,default=4);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(42)
    out=HERE/('clean_smoke/job_'+os.environ['SLURM_JOB_ID'] if a.smoke else 'clean_quality')/f'shard_{a.shard:02d}';out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval();wrapper,model=attach(base,a.adapter)
    cm,rp=CoMem(model,12,tokenizer=tok),CoMem(model,0,tokenizer=tok)
    dump(out/'metadata.json',{'job':os.environ['SLURM_JOB_ID'],'node':platform.node(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':transformers.__version__,'methods':['comem','replay_no_adapter'],'frozen_method':'re-score existing full j12 1150 saved predictions','prompt':'released last-chunk query','bos':'tokenizer BOS or first input token (released default)','filter':'reference/clean_subset_ids.json; identical full support; filtering applied after generation','support':'all 1150 official examples; clean filtering only at scoring'})
    with gzip.open(HERE/'data/clean_samples.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(l) for l in f]
    rows=[rows[0],next(r for r in rows if r['task']=='qasper')] if a.smoke else rows[a.shard::a.nshards]
    rng=random.Random(20260913+a.shard);n=0;began=time.time()
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            chunks,query=parts(row);ix=selection(chunks,row);arms=['comem','replay_no_adapter'];rng.shuffle(arms)
            # Smoke additionally verifies frozen endpoint against the saved full-set scorer.
            if a.smoke:arms.append('without_lora')
            for arm in arms:
                reader=rp if arm=='replay_no_adapter' else cm
                with contextlib.nullcontext() if arm=='comem' else wrapper.disable_adapter():
                    states=[reader.write_chunk(chunks[i]) for i in ix];bos,_=reader._bos_eos(tok,fallback_first=row['input_ids'][0]);sink=reader.write_chunk([bos])
                    generated,_,_=decode(reader,sink,states,query,tok.eos_token_id,row['budget']);del states,sink
                text=tok.decode(generated,skip_special_tokens=True)
                rec={k:row[k] for k in ('id','task','index','dataset_id','answers','clean')};rec.update({'arm':arm,'selected':ix,'generated_ids':generated,'prediction':text,'score':longbench.compute_f1_multi(text,row['answers']),'status':'ok'})
                if a.smoke and arm=='without_lora':
                    reference=json.loads((HERE/'data/clean_frozen_smoke.json').read_text(encoding='utf-8'))[row['id']]
                    assert ix==reference['selected_chunks'],(row['id'],'selection mismatch')
                    assert text.strip()==reference['prediction'].strip(),(row['id'],'frozen prediction mismatch',text,reference['prediction'])
                    assert abs(rec['score']-reference['score'])<1e-12
                    rec['saved_frozen_prediction_equal']=True
                f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();n+=1
            dump(out/'progress.json',{'records':n,'total_samples':len(rows),'last_id':row['id'],'elapsed_s':time.time()-began});print(row['id'],flush=True)
    dump(out/'complete.json',{'complete':True,'records':n,'samples':len(rows),'elapsed_s':time.time()-began})
if __name__=='__main__':main()
