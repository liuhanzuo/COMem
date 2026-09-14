"""Local matched costs. Whole-document CPU stores, selection, fetch, 128 outputs."""
from common import *
from kv_control import KVControl,generate,checks
from transformers import AutoTokenizer,AutoModelForCausalLM
import argparse,contextlib,gzip,platform,random,transformers
@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True);p.add_argument('--process',required=True,type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(16);torch.manual_seed(42)
    sys.path.insert(0,str(HERE.parent));from gpu_gate import acquire_gpu
    admission=acquire_gpu(need_gb=23,cap_gb=28,idle_slack_gb=5,poll=10,max_wait=6*3600,tag='comem_followup_kv')
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *args,**kwargs:False
    out=HERE/('kv_cost_smoke' if a.smoke else 'kv_cost')/f'process_{a.process:02d}';out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval();wrapper,model=attach(base,a.adapter);tok.bos_token_id=model.config.bos_token_id
    cm,rp=CoMem(model,12,tokenizer=tok),CoMem(model,0,tokenizer=tok);cb=KVControl(rp)
    dump(out/'correctness.json',checks(model,tok))
    dump(out/'metadata.json',{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':transformers.__version__,'process':a.process,'cap_bytes':28_000_000_000,'admission':admission,'generation_tokens':128,'formal_reps':3,'warmups':1,'adapter_conditions':['on','off'],'store':'whole document pinned CPU; token IDs for replay, BF16 residuals for CoMem, BF16 all-layer K/V for chunkKV','pipeline':'synchronous HF reference, no overlap of fetch and recompute','KV_query':'zeros for query K/V placeholders; every query position recomputed at all layers; identical to full query cached initialization'})
    with gzip.open(HERE/'data/kv_samples.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(l) for l in f]
    rows=[r for r in rows if r['timing_subset']]
    if a.smoke:rows=[next(r for r in rows if r['cell']==c) for c in ('longeval_8k','qasper')]
    rng=random.Random(20260913+a.process);began=time.time();n=0
    def operation(row,method,on):
        torch.cuda.reset_peak_memory_stats();t0=timed_stamp();chunks,query=parts(row);raw=[x.pin_memory() for x in chunks]
        t_write=timed_stamp()
        if method=='comem':store=[cm.write_chunk(c).cpu().pin_memory() for c in chunks]
        elif method=='chunkkv':
            store=[]
            for c in chunks:
                kv,_=cb.prefill_chunk_full(c)
                store.append(torch.stack([torch.stack([k,v]) for k,v in kv]).cpu().pin_memory());del kv
        else:store=raw
        prepared=timed_stamp();ix=selection(chunks,row);selected=timed_stamp()
        if method=='chunkkv':
            loaded=[store[i].to('cuda',non_blocking=True) for i in ix];fetched=timed_stamp()
            skv,_=cb.prefill_chunk_full([tok.bos_token_id]);qkv=[(torch.zeros_like(k[...,:1,:]).expand(-1,-1,len(query),-1).contiguous(),torch.zeros_like(v[...,:1,:]).expand(-1,-1,len(query),-1).contiguous()) for k,v in skv]
            kvs=[skv]+[[(tensor[l,0],tensor[l,1]) for l in range(cb.num_layers)] for tensor in loaded]+[qkv]
            offsets=[0];off=1
            for i in ix:offsets.append(off);off+=len(chunks[i])
            offsets.append(off);merged=cb.concat_kv_reindex(kvs,offsets)
            pack=torch.cat([rp._as_ids([tok.bos_token_id]).reshape(-1)]+[raw[i].to('cuda',non_blocking=True) for i in ix]+[rp._as_ids(query).reshape(-1)]).view(1,-1)
            del kvs,loaded,qkv,skv
            generated,first,ended,stats=generate(cb,pack,merged,len(query),tok.eos_token_id,128,fixed=True)
        else:
            reader=cm if method=='comem' else rp;states=[store[i].to('cuda',non_blocking=True) if method=='comem' else rp.write_chunk(raw[i]) for i in ix];fetched=timed_stamp();sink=reader.write_chunk([tok.bos_token_id]);generated,first,ended=decode(reader,sink,states,query,tok.eos_token_id,128,fixed=True);stats={}
        alloc=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();assert len(generated)==128 and max(alloc,reserved)<=28_000_000_000
        qids=quality_ids(generated,tok.eos_token_id,row['budget']);prediction=tok.decode(qids,skip_special_tokens=True)
        return {'id':row['id'],'cell':row['cell'],'method':method,'adapter_on':on,'process':a.process,'prepare_ms':1000*(prepared-t0),'write_ms':1000*(prepared-t_write),'selection_ms':1000*(selected-prepared),'fetch_ms':1000*(fetched-selected),'ttft_ms':1000*(first-prepared),'online_ms':1000*(ended-prepared),'e2e_ms':1000*(ended-t0),'persistent_bytes':sum(t.numel()*t.element_size() for t in store),'source_tokens':sum(len(c) for c in chunks),'peak_allocated_bytes':alloc,'peak_reserved_bytes':reserved,'selected':ix,'generated_ids':generated,'prediction':prediction,'score':score(row,prediction),'stats':stats,'status':'ok'}
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            for rep in range(-1,1 if a.smoke else 3):
                arms=[(method,on) for method in ('replay','comem','chunkkv') for on in (False,True)];rng.shuffle(arms)
                for method,on in arms:
                    gc.collect();torch.cuda.empty_cache()
                    try:
                        with contextlib.nullcontext() if on else wrapper.disable_adapter():r=operation(row,method,on)
                    except torch.cuda.OutOfMemoryError as e:r={'id':row['id'],'cell':row['cell'],'method':method,'adapter_on':on,'process':a.process,'status':'OOM','error':str(e)}
                    r.update({'rep':rep,'warmup':rep<0});f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();n+=1
                dump(out/'progress.json',{'records':n,'total_records':len(rows)*6*(2 if a.smoke else 4),'id':row['id'],'rep':rep,'elapsed_s':time.time()-began});print(json.dumps({'records':n,'id':row['id'],'rep':rep}),flush=True)
    dump(out/'complete.json',{'complete':True,'records':n,'elapsed_s':time.time()-began})
if __name__=='__main__':main()
