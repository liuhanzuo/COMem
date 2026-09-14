"""Measure source preparation through 128 output tokens on Table 2(a) inputs."""
from pathlib import Path
import argparse, gc, json, os, platform, random, sys, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'exp'), str(ROOT/'exp/comem_infra_20260912')]
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.integrations.sdpa_attention as sdpa
from comem import CoMem
from comem.selectors import iter_bm25_indices
from bench_local import attach_adapter

def dump(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)

def continue_tokens(reader, logits, bottom, top, query_pos, pack_pos, count):
    token = int(logits[0, -1].float().argmax().item())
    generated = [token]
    for _ in range(1, count):
        logits = reader.decode_step(token, bottom, top, query_pos, pack_pos)
        query_pos += 1
        pack_pos += 1
        token = int(logits[0, -1].float().argmax().item())
        generated.append(token)
    return generated

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--process', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--reps', type=int, default=3)
    a = p.parse_args()
    assert a.tokens > 0 and a.reps > 0
    out = HERE/('smoke' if a.smoke else 'results')/f'process_{a.process:02d}'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    torch.manual_seed(42)
    from gpu_gate import acquire_gpu
    admission = acquire_gpu(need_gb=23, cap_gb=26, idle_slack_gb=5, poll=10,
                            max_wait=3600, tag='comem_e2e_20260913')
    sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
              attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    model, adapter_meta = attach_adapter(model, a.adapter)
    assert all(v.device.type == 'cuda' for v in model.parameters())
    cm, rp = CoMem(model, 12, tokenizer=tok), CoMem(model, 0, tokenizer=tok)
    bos = model.config.bos_token_id
    assert not cm.block_diagonal and not cm.write_sink
    metadata = {'process':a.process, 'smoke':a.smoke, 'gpu':torch.cuda.get_device_name(0),
        'torch':torch.__version__, 'transformers':transformers.__version__,
        'cuda':torch.version.cuda, 'python':platform.python_version(),
        'model':str(a.model), 'adapter_path':str(a.adapter), 'adapter':adapter_meta,
        'cpu_threads':2, 'interop_threads':16, 'admission':admission,
        'generation_tokens':a.tokens, 'decode_steps':a.tokens-1,
        'eos_policy':'fixed output length; EOS does not stop generation',
        'boundary':'pretokenized source/query -> CPU chunk/store preparation (full document Write for CoMem) -> CPU iterative BM25 construction/selection -> pinned fetch -> sink/query Write -> cached prefill -> greedy output token IDs',
        'excluded':'model loading, input tokenization, output detokenization, network/queue/external I/O',
        'document_write_included':True, 'store_rebuilt_each_request':True,
        'source_lengths':[32768,131072], 'topk':12, 'hop_topk':2,
        'warmups':a.warmups, 'reps':a.reps, 'attention':'SDPA, explicit repeat_kv',
        'checkpoint_scope':'principal unmerged FP32 rank-32 suffix adapter, shared with Table 2(a) and the accuracy configuration'}
    dump(out/'metadata.json', metadata)
    # Check the split path, then its autoregressive continuation against a full
    # recomputation on a small multi-chunk pack before any formal observation.
    check_ids = [bos] + tok.encode('Alice keeps number 42. Bob keeps number 73.', add_special_tokens=False)
    stock = model(input_ids=torch.tensor([check_ids], device='cuda'), use_cache=False, logits_to_keep=1).logits
    split = rp.read_core(None, [], rp.write_chunk(check_ids), logits_tail=1)
    check = {'j0_max_abs_difference':float((stock-split).abs().max()),
             'j0_top1_equal':bool(stock.argmax(-1).eq(split.argmax(-1)).all()), 'decode':{}}
    assert check['j0_top1_equal'] and check['j0_max_abs_difference'] <= .125
    del stock, split
    short = tok.encode('Alice keeps number 42 in a blue notebook. '*12, add_special_tokens=False)
    question = tok.encode('What number does Alice keep? Answer:', add_special_tokens=False)
    for name, reader in [('replay_k12',rp),('comem_k12',cm)]:
        sink = reader.write_chunk([bos])
        states = [reader.write_chunk(short[:64]), reader.write_chunk(short[64:128])]
        query, bottom, qpos = reader.write_prefill(question)
        logits, top, ppos = reader.read_prefill(sink, states, query)
        actual = continue_tokens(reader, logits, bottom, top, qpos, ppos, 8)
        expected = reader._decode_from_pack(sink, states, question, None, 8, False)
        check['decode'][name] = {'equal':actual == expected, 'cached_ids':actual, 'recomputed_ids':expected}
        assert actual == expected, check['decode'][name]
        del sink, states, query, bottom, logits, top
    dump(out/'correctness.json', check)
    gc.collect(); torch.cuda.empty_cache()
    rng = random.Random(4200+a.process)
    records = 0
    started = time.time()
    for length in (32768,131072):
        workload_path = HERE/'workloads'/f'{length}.json'
        workloads = json.loads(workload_path.read_text(encoding='utf-8'))
        assert len(workloads) == 3
        for wi, w in enumerate(workloads[:1] if a.smoke else workloads):
            assert len(w['source_tokens']) == length and len(w['query_tokens']) == 512
            selected = w['selected']['12']
            assert len(selected) == 12
            torch.cuda.synchronize()
            gc.collect(); torch.cuda.empty_cache()
            arms = [('replay_k12',rp),('comem_k12',cm)]
            rng.shuffle(arms)
            for name, reader in arms:
                def operation():
                    chunks = list(torch.tensor(w['source_tokens'], dtype=torch.long).split(512))
                    raw = [x.pin_memory() for x in chunks]
                    residuals = [cm.write_chunk(x).to('cpu').pin_memory() for x in chunks] if reader.resume_j else None
                    ix = iter_bm25_indices(chunks, w['selector_query_tokens'], 12, iter_hop_topk=2)
                    assert ix == selected
                    states = [residuals[i].to('cuda', non_blocking=True) if reader.resume_j else reader.write_chunk(raw[i]) for i in ix]
                    sink = reader.write_chunk([bos])
                    query, bottom, qpos = reader.write_prefill(w['query_tokens'])
                    logits, top, ppos = reader.read_prefill(sink, states, query)
                    assert ppos == 6657
                    generated = continue_tokens(reader, logits, bottom, top, qpos, ppos, a.tokens)
                    torch.cuda.synchronize()
                    # Stop at the completed output, before releasing reusable
                    # stores/caches as this benchmark call returns.
                    ended = time.perf_counter()
                    peak = torch.cuda.max_memory_allocated()
                    return generated, ended, peak
                warmups = 1 if a.smoke else a.warmups
                reps = 1 if a.smoke else a.reps
                for _ in range(warmups):
                    output, _, _ = operation(); del output
                gc.collect()
                for rep in range(reps):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    output, ended, peak = operation()
                    elapsed_ms = (ended-t0)*1000
                    assert len(output) == a.tokens
                    row = {'process':a.process, 'source_tokens':length, 'workload':wi,
                        'book_index':w['book_index'], 'arm':name, 'rep':rep,
                        'latency_ms':elapsed_ms, 'peak_allocated_bytes':peak,
                        'selected_chunks':selected, 'pack_tokens':6657,
                        'output_tokens':len(output), 'decode_steps':len(output)-1, 'generated_ids':output}
                    with (out/'records.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(row)+'\n')
                    records += 1
                    dump(out/'progress.json', {'records':records,'source_tokens':length,'workload':wi,
                         'arm':name,'last_ms':elapsed_ms,'elapsed_s':time.time()-started})
                    print(json.dumps({k:v for k,v in row.items() if k not in ('generated_ids','selected_chunks')}),flush=True)
                gc.collect(); torch.cuda.empty_cache()
            gc.collect(); torch.cuda.empty_cache()
    dump(out/'complete.json', {'complete':True,'records':records,'elapsed_s':time.time()-started})

if __name__ == '__main__':
    main()
