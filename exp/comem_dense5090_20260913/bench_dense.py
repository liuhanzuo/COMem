"""Full-source GPU-resident Dense baseline under a 28 GB allocator budget."""
from pathlib import Path
import argparse, gc, json, os, platform, random, sys, time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'exp'),str(ROOT/'exp/comem_infra_20260912')]
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
import transformers
import transformers.integrations.sdpa_attention as sdpa
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.attention import sdpa_kernel, SDPBackend
from bench_local import attach_adapter

CAP=28_000_000_000
def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)

def generate(model,ids,count=128):
    out=model(input_ids=ids,use_cache=True,logits_to_keep=1)
    generated=[int(out.logits[0,-1].float().argmax().item())]
    for _ in range(1,count):
        token=torch.tensor([[generated[-1]]],device='cuda')
        out=model(input_ids=token,past_key_values=out.past_key_values,use_cache=True,logits_to_keep=1)
        generated.append(int(out.logits[0,-1].float().argmax().item()))
    return out,generated

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--process',type=int,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    outdir=HERE/('smoke' if a.smoke else 'results')/f'process_{a.process:02d}'
    outdir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.set_num_interop_threads(16); torch.manual_seed(42)
    from gpu_gate import acquire_gpu
    # Admission uses GiB; the allocator budget below uses decimal GB.
    admission=acquire_gpu(need_gb=26.5,cap_gb=28,idle_slack_gb=5,poll=10,max_wait=7200,
                          tag='comem_dense5090_20260913')
    sdpa.use_gqa_in_sdpa=lambda *args,**kwargs: False
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    model,adapter=attach_adapter(model,a.adapter)
    assert all(p.device.type=='cuda' for p in model.parameters())
    assert all(p.dtype==torch.bfloat16 for n,p in model.named_parameters() if 'lora_' not in n)
    meta={'process':a.process,'smoke':a.smoke,'gpu':torch.cuda.get_device_name(0),
          'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,
          'python':platform.python_version(),'model':str(a.model),'adapter_path':str(a.adapter),
          'adapter':adapter,'adapter_config':json.loads((a.adapter/'adapter_config.json').read_text()),
          'admission':admission,'allocator_budget_bytes':CAP,
          'budget_scope':'total per-process PyTorch CUDA allocator, including weights, KV and temporaries; decimal GB, not an incremental activation allowance',
          'memory':'allocated and reserved peaks include weights; driver desktop/context overhead is separate',
          'offload':False,'quantization':False,'truncation':False,'rope_extension':False,
          'attention':'fused SDPA only; explicit repeat_kv; math disabled',
          'cpu_threads':2,'interop_threads':16,'output_tokens':128,'eos_policy':'fixed length; ignore EOS',
          'warmups':0 if a.smoke else 1,'reps':1 if a.smoke else 3,
          'boundaries':{'prefill':'GPU-ready full BOS/source/query IDs -> cached full-model prefill -> first logits',
             'ttft':'pretokenized CPU source/query -> assemble and pin full IDs -> H2D -> full cached prefill -> first logits',
             'e2e':'same CPU start as TTFT -> full prefill -> 128 output token IDs (127 cached decode steps)'},
          'excluded':'model load, tokenization/detokenization, network/queue/external I/O',
          'quality':'unscored original local PG19 workloads; no accuracy claim beyond native positions',
          'oom_policy':'record CUDA allocator OOM, never shorten/quantize/offload/fill a latency; other exceptions fail the worker'}
    dump(outdir/'metadata.json',meta)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        small=torch.tensor([[model.config.bos_token_id]+tok.encode('Alice keeps number 42. What number does Alice keep? Answer:',add_special_tokens=False)],device='cuda')
        result,cached=generate(model,small,8)
        del result
        current=small.clone(); recomputed=[]
        for _ in range(8):
            result=model(input_ids=current,use_cache=False,logits_to_keep=1)
            next_id=int(result.logits[0,-1].float().argmax().item())
            recomputed.append(next_id)
            current=torch.cat([current,torch.tensor([[next_id]],device='cuda')],dim=1)
            del result
        check={'cached_recompute_equal':cached==recomputed,'cached_ids':cached,'recomputed_ids':recomputed}
        dump(outdir/'correctness.json',check)
        assert cached==recomputed,check
        del small,current
        gc.collect(); torch.cuda.empty_cache()
        rng=random.Random(8300+a.process)
        records=0; cells=0; failed_cells=0; started=time.time()
        for length in (32768,131072):
            workloads=json.loads((HERE/'workloads'/f'{length}.json').read_text(encoding='utf-8'))
            assert len(workloads)==3
            for wi,w in enumerate(workloads[:1] if a.smoke else workloads):
                assert len(w['source_tokens'])==length and len(w['query_tokens'])==512
                full=[model.config.bos_token_id]+w['source_tokens']+w['query_tokens']
                assert len(full)==length+513
                phases=['prefill','ttft','e2e']; rng.shuffle(phases)
                for phase in phases:
                    cells+=1
                    gc.collect(); torch.cuda.empty_cache()
                    # All full-source IDs fit trivially; prefill excludes the transfer.
                    ready=torch.tensor([full],device='cuda') if phase=='prefill' else None
                    torch.cuda.synchronize()
                    def operation():
                        ids=ready if ready is not None else torch.tensor([full],dtype=torch.long).pin_memory().to('cuda',non_blocking=True)
                        assert ids.shape==(1,length+513)
                        if phase=='e2e': result,generated=generate(model,ids)
                        else: result=model(input_ids=ids,use_cache=True,logits_to_keep=1); generated=[]
                        torch.cuda.synchronize()
                        return result,generated,time.perf_counter()
                    def attempt(rep,warmup):
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        t0=time.perf_counter()
                        row={'process':a.process,'source_tokens':length,'model_tokens':length+513,
                             'workload':wi,'book_index':w['book_index'],'phase':phase,'rep':rep,'warmup':warmup,
                             'allocator_budget_bytes':CAP,'selected_chunks':None}
                        try:
                            result,generated,ended=operation()
                            assert not generated or len(generated)==128
                            row.update(status='ok',latency_ms=(ended-t0)*1000,
                                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),generated_ids=generated,
                                       output_tokens=len(generated))
                            assert row['peak_reserved_bytes']<=CAP and row['peak_allocated_bytes']<=CAP
                            del result
                        except torch.OutOfMemoryError as error:
                            row.update(status='OOM',latency_ms=None,error=str(error),
                                allocated_at_oom=torch.cuda.memory_allocated(),
                                reserved_at_oom=torch.cuda.memory_reserved(),
                                observed_peak_before_oom=torch.cuda.max_memory_allocated())
                        return row
                    reps=1 if a.smoke else 3
                    sequence=[(0,False)] if a.smoke else [(-1,True),*[(i,False) for i in range(reps)]]
                    failed=False
                    for rep,warmup in sequence:
                        row=attempt(rep,warmup)
                        with (outdir/'records.jsonl').open('a',encoding='utf-8') as f:
                            f.write(json.dumps(row)+'\n')
                        records+=1
                        dump(outdir/'progress.json',{'cells_started':cells,'records':records,'source_tokens':length,
                             'workload':wi,'phase':phase,'status':row['status'],'elapsed_s':time.time()-started})
                        print(json.dumps({k:v for k,v in row.items() if k not in ('generated_ids','error')}),flush=True)
                        if row['status']=='OOM':
                            failed=True; failed_cells+=1
                            gc.collect(); torch.cuda.empty_cache()
                            break
                    dump(outdir/f'cell_{length}_{wi}_{phase}.json',{'status':'OOM' if failed else 'ok',
                         'source_tokens':length,'workload':wi,'phase':phase,'successful_formal_reps':0 if failed else reps})
                    del ready
                    gc.collect(); torch.cuda.empty_cache()
        dump(outdir/'complete.json',{'complete':True,'cells':cells,'records':records,
             'failed_cells':failed_cells,'elapsed_s':time.time()-started})

if __name__=='__main__': main()
