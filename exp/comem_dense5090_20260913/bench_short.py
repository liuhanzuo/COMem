"""Paired 8k/16k full-source Dense and CoMem measurements under 28 GB."""
from pathlib import Path
import argparse, gc, json, os, platform, random, sys, time
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'exp'),str(ROOT/'exp/comem_e2e_20260913')]
from bench_dense import CAP, dump, generate, attach_adapter
from bench_e2e import continue_tokens
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.integrations.sdpa_attention as sdpa
from torch.nn.attention import sdpa_kernel, SDPBackend
from comem import CoMem
from comem.selectors import iter_bm25_indices

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--process',type=int,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    outdir=HERE/('short_smoke' if a.smoke else 'short_results')/f'process_{a.process:02d}'
    outdir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.set_num_interop_threads(16); torch.manual_seed(42)
    from gpu_gate import acquire_gpu
    admission=acquire_gpu(need_gb=26.5,cap_gb=28,idle_slack_gb=5,poll=10,max_wait=7200,
                          tag='comem_dense5090_short_20260913')
    sdpa.use_gqa_in_sdpa=lambda *args,**kwargs:False
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    model,adapter=attach_adapter(model,a.adapter)
    assert all(v.device.type=='cuda' for v in model.parameters())
    assert all(v.dtype==torch.bfloat16 for n,v in model.named_parameters() if 'lora_' not in n)
    cm=CoMem(model,12,tokenizer=tok); bos=model.config.bos_token_id
    assert not cm.block_diagonal and not cm.write_sink
    meta={'process':a.process,'smoke':a.smoke,'gpu':torch.cuda.get_device_name(0),
        'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,
        'python':platform.python_version(),'model':str(a.model),'adapter_path':str(a.adapter),
        'adapter':adapter,'adapter_config':json.loads((a.adapter/'adapter_config.json').read_text()),
        'admission':admission,'allocator_budget_bytes':CAP,'cpu_threads':2,'interop_threads':16,
        'budget_scope':'total per-process CUDA allocator including weights, KV, inputs and temporaries; decimal GB',
        'memory':'allocated and reserved peaks include weights; driver desktop/context overhead separate',
        'offload':False,'quantization':False,'truncation':False,'rope_extension':False,
        'comem_store':'CPU-pinned full-source residual store, as specified by CoMem; weights fully GPU resident',
        'attention':'fused SDPA only; explicit repeat_kv; math disabled',
        'source_lengths':[8192,16384],'query_tokens':512,'topk':12,'hop_topk':2,
        'output_tokens':128,'eos_policy':'fixed length; ignore EOS','warmups':1,'reps':1 if a.smoke else 3,
        'boundaries':{
            'dense_prefill':'GPU-ready BOS/full-source/query IDs -> native full-model cached prefill -> first logits',
            'dense_ttft':'pretokenized CPU source/query -> full ID assembly/pin/H2D -> native cached prefill -> first logits',
            'dense_e2e':'same CPU start as TTFT -> full prefill -> 128 output token IDs',
            'comem_prefill':'GPU-ready sink/selected residuals/query residual -> cached suffix prefill -> first logits',
            'comem_ttft':'pretokenized CPU query and prewritten CPU store -> BM25 rebuild/select -> fetch -> sink/query Write -> cached prefill -> first logits',
            'comem_e2e':'pretokenized CPU source/query -> fresh CPU chunk/raw store and full document Write -> BM25 -> fetch -> sink/query Write -> cached prefill -> 128 output token IDs'},
        'excluded':'model load, tokenization/detokenization, network/queue/external I/O; store/cache teardown after output',
        'checkpoint_scope':'principal unmerged FP32 rank-32 suffix adapter, shared across all compared arms',
        'oom_policy':'record actual CUDA OOM; stop repetitions of that cell, attempt remaining cells; never truncate/quantize/offload'}
    dump(outdir/'metadata.json',meta)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        small=torch.tensor([[bos]+tok.encode('Alice keeps number 42. What number does Alice keep? Answer:',add_special_tokens=False)],device='cuda')
        result,cached=generate(model,small,8); del result
        current=small.clone(); recomputed=[]
        for _ in range(8):
            result=model(input_ids=current,use_cache=False,logits_to_keep=1)
            token=int(result.logits[0,-1].float().argmax().item()); recomputed.append(token)
            current=torch.cat([current,torch.tensor([[token]],device='cuda')],dim=1); del result
        check={'dense':{'equal':cached==recomputed,'cached_ids':cached,'recomputed_ids':recomputed}}
        assert cached==recomputed,check
        del small,current
        short=tok.encode('Alice keeps number 42 in a blue notebook. '*12,add_special_tokens=False)
        question=tok.encode('What number does Alice keep? Answer:',add_special_tokens=False)
        sink=cm.write_chunk([bos]); states=[cm.write_chunk(short[:64]),cm.write_chunk(short[64:128])]
        query,bottom,qpos=cm.write_prefill(question)
        logits,top,ppos=cm.read_prefill(sink,states,query)
        actual=continue_tokens(cm,logits,bottom,top,qpos,ppos,8)
        expected=cm._decode_from_pack(sink,states,question,None,8,False)
        check['comem']={'equal':actual==expected,'cached_ids':actual,'recomputed_ids':expected}
        assert actual==expected,check
        dump(outdir/'correctness.json',check)
        del sink,states,query,bottom,logits,top
        gc.collect(); torch.cuda.empty_cache()
        rng=random.Random(9600+a.process); started=time.time(); records=0; cells=0; failed_cells=0
        for length in (8192,16384):
            workloads=json.loads((HERE/'workloads'/f'{length}.json').read_text(encoding='utf-8'))
            assert len(workloads)==3
            for wi,w in enumerate(workloads[:1] if a.smoke else workloads):
                assert len(w['source_tokens'])==length and len(w['query_tokens'])==512
                selected=w['selected']['12']; assert len(selected)==12
                methods=['dense','comem']; rng.shuffle(methods)
                for method in methods:
                    chunks=raw=residuals=None
                    if method=='comem':
                        chunks=list(torch.tensor(w['source_tokens'],dtype=torch.long).split(512))
                        raw=[c.pin_memory() for c in chunks]
                        residuals=[cm.write_chunk(c).to('cpu').pin_memory() for c in chunks]
                    phases=['prefill','ttft','e2e']; rng.shuffle(phases)
                    for phase in phases:
                        cells+=1
                        gc.collect(); torch.cuda.empty_cache()
                        ready=None
                        if phase=='prefill':
                            if method=='dense':
                                ready=torch.tensor([[bos]+w['source_tokens']+w['query_tokens']],device='cuda')
                            else:
                                ready=(cm.write_chunk([bos]),[residuals[i].to('cuda',non_blocking=True) for i in selected],cm.write_chunk(w['query_tokens']))
                        torch.cuda.synchronize()
                        def operation():
                            if method=='dense':
                                ids=ready if ready is not None else torch.tensor([[bos]+w['source_tokens']+w['query_tokens']],dtype=torch.long).pin_memory().to('cuda',non_blocking=True)
                                assert ids.shape==(1,length+513)
                                if phase=='e2e': result,generated=generate(model,ids)
                                else: result=model(input_ids=ids,use_cache=True,logits_to_keep=1); generated=[]
                            elif phase=='prefill':
                                result=cm.read_prefill(*ready); assert result[2]==6657; generated=[]
                            else:
                                if phase=='e2e':
                                    request_chunks=list(torch.tensor(w['source_tokens'],dtype=torch.long).split(512))
                                    request_raw=[c.pin_memory() for c in request_chunks]
                                    request_residuals=[cm.write_chunk(c).to('cpu').pin_memory() for c in request_chunks]
                                else: request_chunks=chunks; request_residuals=residuals
                                ix=iter_bm25_indices(request_chunks,w['selector_query_tokens'],12,iter_hop_topk=2)
                                assert ix==selected
                                hh=[request_residuals[i].to('cuda',non_blocking=True) for i in ix]
                                sink=cm.write_chunk([bos]); query,bottom,qpos=cm.write_prefill(w['query_tokens'])
                                logits,top,ppos=cm.read_prefill(sink,hh,query); assert ppos==6657
                                generated=continue_tokens(cm,logits,bottom,top,qpos,ppos,128) if phase=='e2e' else []
                                result=(logits,top,bottom)
                            torch.cuda.synchronize()
                            # Stop at output readiness before releasing CPU stores/caches.
                            return result,generated,time.perf_counter()
                        def attempt(rep,warmup):
                            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                            t0=time.perf_counter()
                            row={'process':a.process,'source_tokens':length,'workload':wi,'book_index':w['book_index'],
                                'method':method,'phase':phase,'rep':rep,'warmup':warmup,
                                'model_tokens':length+513 if method=='dense' else 6657,
                                'selected_chunks':None if method=='dense' else selected,'allocator_budget_bytes':CAP}
                            try:
                                result,generated,ended=operation()
                                row.update(status='ok',latency_ms=(ended-t0)*1000,
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                    generated_ids=generated,output_tokens=len(generated))
                                assert row['output_tokens']==(128 if phase=='e2e' else 0)
                                assert row['peak_allocated_bytes']<=row['peak_reserved_bytes']<=CAP
                                del result
                            except torch.OutOfMemoryError as error:
                                row.update(status='OOM',latency_ms=None,error=str(error),
                                    allocated_at_oom=torch.cuda.memory_allocated(),reserved_at_oom=torch.cuda.memory_reserved(),
                                    observed_peak_before_oom=torch.cuda.max_memory_allocated())
                            return row
                        failed=False; successes=0
                        for rep,warmup in [(-1,True),*[(i,False) for i in range(meta['reps'])]]:
                            row=attempt(rep,warmup)
                            with (outdir/'records.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
                            records+=1
                            successes+=int(not warmup and row['status']=='ok')
                            dump(outdir/'progress.json',{'cells_started':cells,'records':records,'source_tokens':length,
                                'workload':wi,'method':method,'phase':phase,'status':row['status'],'elapsed_s':time.time()-started})
                            print(json.dumps({k:v for k,v in row.items() if k not in ('generated_ids','error','selected_chunks')}),flush=True)
                            if row['status']=='OOM':
                                failed=True; failed_cells+=1; gc.collect(); torch.cuda.empty_cache(); break
                        dump(outdir/f'cell_{length}_{wi}_{method}_{phase}.json',{'status':'OOM' if failed else 'ok',
                            'source_tokens':length,'workload':wi,'method':method,'phase':phase,'successful_formal_reps':successes})
                        del ready
                        gc.collect(); torch.cuda.empty_cache()
                    del chunks,raw,residuals
                    gc.collect(); torch.cuda.empty_cache()
        dump(outdir/'complete.json',{'complete':True,'cells':cells,'records':records,'failed_cells':failed_cells,'elapsed_s':time.time()-started})

if __name__=='__main__': main()
