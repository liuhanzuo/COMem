"""Matched B300 full-context / selected-pack infrastructure recheck.

Run only in a one-GPU Slurm allocation. No quality results are inferred.
Native fused SDPA is required so the dense baseline cannot silently time the
quadratic-memory math backend. All principal arms share the exact local LoRA.
"""
from pathlib import Path
import argparse, contextlib, gc, json, os, platform, random, statistics, subprocess, sys, time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.attention import sdpa_kernel, SDPBackend
from comem import CoMem
from comem.selectors import iter_bm25_indices

def dump(p,v): p.write_text(json.dumps(v,indent=2),encoding='utf-8')

def attach(model,path):
    from peft import PeftModel
    from unittest.mock import patch
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):
        wrapper=PeftModel.from_pretrained(model,str(path),autocast_adapter_dtype=True)
    return wrapper,wrapper.base_model.model.eval()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--adapter',type=Path,default=ROOT/'adapter')
    p.add_argument('--workloads',type=Path,default=ROOT/'workloads.json')
    p.add_argument('--process',type=int,required=True)
    p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--reps',type=int,default=3)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Must use Slurm GPU allocation'
    assert torch.cuda.device_count()==1, 'Exactly one visible GPU required'
    torch.set_num_threads(2); torch.set_num_interop_threads(4); torch.manual_seed(42)
    out=ROOT/('smoke' if a.smoke else 'results')/f'process_{a.process:02d}'
    out.mkdir(parents=True,exist_ok=False)
    device=torch.cuda.get_device_properties(0)
    assert (device.major,device.minor)==(10,3), 'Allocated device is not the requested B300 architecture'
    assert device.total_memory>200_000_000_000, 'Unexpected device memory for the requested B300 allocation'
    metadata={'gpu_display_name':device.name,'hardware_label':'B300 (CUDA sm_103; driver alias L20D)','total_memory_bytes':device.total_memory,'compute_capability':[device.major,device.minor],'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),'slurm_job':os.environ['SLURM_JOB_ID'],'node':platform.node(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'memory':'maximum PyTorch allocated bytes including weights, decimal GB','sdpa':'native GQA; flash/efficient SDPA only, math backend disabled','source_tokens':131072,'query_tokens':512,'full_prompt_tokens':131585,'selected_pack_tokens':6657,'quality':'Unscored PG19 cost workloads; no long-context accuracy claim or positional extrapolation change','model_offload':False,'quantization':False,'adapter_scope':'principal rank-32 suffix adapter, shared with the RTX5090 measurements and paper accuracy configuration','boundaries':{'dense_stock':'GPU-ready full IDs to first logits, use_cache=True, adapter disabled','dense_shared':'GPU-ready full IDs to first logits, use_cache=True, same unmerged FP32 adapter as CoMem','replay_read':'GPU-ready embedding pack through all layers to first logits with KV','comem_read':'GPU-ready residual pack through suffix to first logits with KV','replay_ttft':'pretokenized query -> online BM25 -> raw fetch -> prefill to first logits','comem_ttft':'pretokenized query -> online BM25 -> pinned residual fetch -> sink/query Write -> suffix prefill to first logits'},'excluded':'document Write, model load, tokenization, decode, external I/O; selector rebuild included in TTFT'}
    dump(out/'metadata.json',metadata)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    wrapper,model=attach(base,a.adapter)
    assert all(p.device.type=='cuda' for p in model.parameters())
    adapter_params=[p for n,p in model.named_parameters() if 'lora_' in n]
    assert adapter_params and all(p.dtype==torch.float32 for p in adapter_params)
    metadata['adapter_config']=json.loads((a.adapter/'adapter_config.json').read_text())
    metadata['adapter_parameter_count']=sum(p.numel() for p in adapter_params)
    dump(out/'metadata.json',metadata)
    cm,rp=CoMem(model,12,tokenizer=tok),CoMem(model,0,tokenizer=tok)
    bos=model.config.bos_token_id
    workloads=json.loads(a.workloads.read_text())
    assert len(workloads)==3 and all(len(w['source_tokens'])==131072 for w in workloads)
    # Check the actual j=0 adapter-matched reader against stock last logits.
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        check_ids=torch.tensor([[bos]+workloads[0]['source_tokens'][:255]],device='cuda')
        stock=model(input_ids=check_ids,use_cache=True,logits_to_keep=1).logits
        split=rp.read_prefill(None,[],rp.write_chunk(check_ids))[0]
        check={'max_abs_logit_difference':float((stock-split).abs().max()),'same_top1':bool(stock.argmax(-1).eq(split.argmax(-1)).all())}
        dump(out/'correctness.json',check)
        assert check['same_top1'] and check['max_abs_logit_difference']<=.125,check
        del stock,split,check_ids
        rng=random.Random(42+a.process)
        records=[]
        for wi,w in enumerate(workloads[:1] if a.smoke else workloads):
            source=w['source_tokens'][:8192] if a.smoke else w['source_tokens']
            chunks=list(torch.tensor(source,dtype=torch.long).split(512))
            raw=[c.pin_memory() for c in chunks]
            selected=iter_bm25_indices(chunks,w['selector_query_tokens'],12,iter_hop_topk=2)
            assert len(selected)==12
            if not a.smoke: assert selected==w['selected']['12']
            residual=[cm.write_chunk(c).to('cpu').pin_memory() for c in chunks]
            full_ids=torch.tensor([[bos]+source+w['query_tokens']],device='cuda')
            arms=['dense_stock','dense_shared','replay_read','comem_read','replay_ttft','comem_ttft']; rng.shuffle(arms)
            for name in arms:
                gc.collect(); torch.cuda.empty_cache()
                ready=None
                if name.endswith('_read'):
                    reader=cm if name.startswith('comem') else rp
                    hs=[residual[i].to('cuda',non_blocking=True) if reader.resume_j else reader.write_chunk(raw[i]) for i in selected]
                    ready=(reader.write_chunk([bos]),hs,reader.write_chunk(w['query_tokens']))
                    del hs
                def operation():
                    if name.startswith('dense'):
                        return model(input_ids=full_ids,use_cache=True,logits_to_keep=1)
                    if name.endswith('_read'): return reader.read_prefill(*ready)
                    r=cm if name.startswith('comem') else rp
                    ix=iter_bm25_indices(chunks,w['selector_query_tokens'],12,iter_hop_topk=2)
                    assert ix==selected
                    hh=[residual[i].to('cuda',non_blocking=True) if r.resume_j else r.write_chunk(raw[i]) for i in ix]
                    sink=r.write_chunk([bos]); query,bottom,_=r.write_prefill(w['query_tokens'])
                    return r.read_prefill(sink,hh,query),bottom
                context=wrapper.disable_adapter() if name=='dense_stock' else contextlib.nullcontext()
                with context:
                    for _ in range(0 if a.smoke else a.warmups):
                        result=operation(); torch.cuda.synchronize(); del result
                    for rep in range(1 if a.smoke else a.reps):
                        torch.cuda.synchronize(); baseline=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
                        start=time.perf_counter(); result=operation(); torch.cuda.synchronize()
                        row={'process':a.process,'workload':wi,'book_index':w['book_index'],'arm':name,'rep':rep,'latency_ms':(time.perf_counter()-start)*1000,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'baseline_allocated_bytes':baseline,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'source_tokens':len(source),'pack_tokens':len(source)+513 if name.startswith('dense') else 6657,'selected_chunks':selected if not name.startswith('dense') else None}
                        with (out/'records.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                        records.append(row); del result
                print(json.dumps({'workload':wi,'arm':name,'median_ms':statistics.median(r['latency_ms'] for r in records if r['workload']==wi and r['arm']==name),'peak_GB':max(r['peak_allocated_bytes'] for r in records if r['workload']==wi and r['arm']==name)/1e9}),flush=True)
                del ready; gc.collect(); torch.cuda.empty_cache()
            del residual,raw,chunks,full_ids; gc.collect(); torch.cuda.empty_cache()
    dump(out/'complete.json',{'complete':True,'records':len(records)})
if __name__=='__main__': main()
