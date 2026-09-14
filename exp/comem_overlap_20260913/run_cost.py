"""Local paired quality/cost subset; fixed 128-token generation, full Write."""
import argparse,gzip,json,platform,random,statistics,time
from common import *
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True)
    p.add_argument('--process',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--reps',type=int,default=3)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(16);torch.manual_seed(42)
    sys.path.insert(0,str(HERE.parent))
    from gpu_gate import acquire_gpu
    admission=acquire_gpu(need_gb=23,cap_gb=28e9/1024**3,idle_slack_gb=5,poll=10,max_wait=6*3600,tag='comem_overlap_20260913')
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *args,**kwargs:False
    out=HERE/('cost_smoke' if a.smoke else 'cost')/f'process_{a.process:02d}'
    out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    wrapper,model=attach(base,a.adapter)
    dump(out/'metadata.json',{'process':a.process,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':transformers.__version__,'adapter':a.adapter,'adapter_identity':'Author-confirmed same as paper checkpoint','cap_bytes':28_000_000_000,'admission':admission,'fixed_generation_tokens':128,'formal_repetitions':a.reps,'warmups':1,'protocol':'PROTOCOL_zh.md','fetch_scope':'residual fetch for CoMem; token fetch and embedding for replay'})
    dump(out/'correctness.json',check_paths(wrapper,model,tok))
    with gzip.open(HERE/'samples.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(s) for s in f]
    rows=[r for r in rows if r['timing_subset']]
    if a.smoke:rows=[next(r for r in rows if r['cell']=='longeval_8k'),next(r for r in rows if r['cell']=='qasper')]
    cm,rp=CoMem(model,12,tokenizer=tok),CoMem(model,0,tokenizer=tok)
    rng=random.Random(20260913+a.process);started=time.time();records=0
    def operation(row,arm):
        reader=rp if arm=='replay' else cm
        torch.cuda.reset_peak_memory_stats();t0=timed_stamp()
        chunks,query=parts(row);raw=[c.pin_memory() for c in chunks]
        source=torch.cat(chunks)
        write_start=timed_stamp()
        residuals=[write_one(cm,source,i,32 if arm=='w32' else 0).to('cpu').pin_memory() for i in range(len(chunks))] if arm!='replay' else None
        prep=timed_stamp()
        ix=selection(chunks,row);selected_at=timed_stamp()
        states=[residuals[i].to('cuda',non_blocking=True) if residuals is not None else rp.write_chunk(raw[i]) for i in ix]
        fetched=timed_stamp()
        bos,_=reader._bos_eos(tok,fallback_first=row['input_ids'][0])
        sink=reader.write_chunk([bos])
        generated,first,ended=decode(reader,sink,states,query,tok.eos_token_id,128,fixed=True)
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        assert len(generated)==128 and peak<=28_000_000_000 and reserved<=28_000_000_000
        return {'id':row['id'],'cell':row['cell'],'arm':arm,'process':a.process,'prepare_ms':1000*(prep-t0),'document_write_ms':1000*(prep-write_start),'selection_ms':1000*(selected_at-prep),'fetch_ms':1000*(fetched-selected_at),'query_write_prefill_ms':1000*(first-fetched),'ttft_ms':1000*(first-prep),'decode_ms':1000*(ended-first),'online_ms':1000*(ended-prep),'e2e_ms':1000*(ended-t0),'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'persistent_bytes':sum(t.numel()*t.element_size() for t in (residuals if residuals is not None else raw)),'generated_ids':generated,'selected':ix,'cached_positions':sum(h.shape[1] for h in states),'status':'ok'}
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            for rep in range(-1,1 if a.smoke else a.reps):
                arms=list(ARMS);rng.shuffle(arms)
                for arm in arms:
                    gc.collect();torch.cuda.empty_cache()
                    try:
                        record=operation(row,arm)
                        record.update({'rep':rep,'warmup':rep==-1})
                        qids=quality_ids(record['generated_ids'],tok.eos_token_id,row['budget'])
                        record['prediction']=tok.decode(qids,skip_special_tokens=True)
                        record['score']=score(row,record['prediction'])
                    except torch.cuda.OutOfMemoryError as error:
                        record={'id':row['id'],'cell':row['cell'],'arm':arm,'process':a.process,'rep':rep,'warmup':rep==-1,'status':'OOM','error':str(error)}
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();records+=1
                dump(out/'progress.json',{'records_including_warmup':records,'id':row['id'],'rep':rep,'elapsed_s':time.time()-started})
                print(json.dumps({'id':row['id'],'rep':rep,'records':records,'elapsed_s':time.time()-started}),flush=True)
    dump(out/'complete.json',{'complete':True,'samples':len(rows),'records_including_warmup':records,'elapsed_s':time.time()-started})
if __name__=='__main__':main()
