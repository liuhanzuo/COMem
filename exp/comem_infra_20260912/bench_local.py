"""Local CoMem infrastructure supplement; fixed workloads, no accuracy scoring.

Fresh independent processes use the same unmerged adapter in every arm. Results
belong to the RTX 5090 environment; H20 timings are a separate experiment.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'exp'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.integrations.sdpa_attention as sdpa
from comem import CoMem
from comem.selectors import iter_bm25_indices


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def attach_adapter(model, path):
    from peft import PeftModel
    from unittest.mock import patch
    # This is an unquantized nn.Linear backbone; the installed optional TorchAO
    # dispatcher is incompatible with PEFT and is irrelevant to these weights.
    with patch('peft.tuners.lora.model.dispatch_torchao', return_value=None):
        wrapper = PeftModel.from_pretrained(model, str(path), autocast_adapter_dtype=True)
    model = wrapper.base_model.model.eval()
    params = [(n, p) for n, p in model.named_parameters() if 'lora_' in n]
    assert params and all(p.dtype == torch.float32 for _, p in params)
    return model, {'tensor_count': len(params), 'parameter_count': sum(p.numel() for _, p in params),
                   'dtype': 'float32', 'merged': False}


def prepare_workloads(tok, source, out, source_length=32768):
    workloads = []
    with source.open(encoding='utf-8') as f:
        for book_i, line in enumerate(f):
            obj = json.loads(line)
            ids = tok.encode(obj['text'], add_special_tokens=False)
            if len(ids) < source_length + 512:
                continue
            doc, query = ids[:source_length], ids[source_length:source_length + 512]
            chunks = list(torch.tensor(doc, dtype=torch.long).split(512))
            selected = {str(k): iter_bm25_indices(chunks, query[:32], k, iter_hop_topk=2) for k in (10, 12)}
            assert all(len(selected[str(k)]) == k for k in (10, 12))
            record = {'book_index': book_i, 'source_tokens': doc, 'query_tokens': query,
                      'selector_query_tokens': query[:32], 'selected': selected,
                      'description': 'PG19 prefix followed by 512-token continuation; unscored fixed-workload timing'}
            workloads.append(record)
            if len(workloads) == 3:
                break
    assert len(workloads) == 3
    dump(out, workloads)
    return workloads


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--process', type=int, required=True)
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--reps', type=int, default=20)
    p.add_argument('--source-length', type=int, default=32768)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    torch.manual_seed(42)
    from gpu_gate import acquire_gpu
    admission = acquire_gpu(need_gb=23, cap_gb=26, idle_slack_gb=5, poll=10,
                            max_wait=600, tag='comem_infra_20260912')
    # Existing Windows policy: explicit repeat_kv preserves the efficient SDPA
    # kernel on Torch 2.7, whose native GQA path otherwise selects math attention.
    sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    workload_path = a.out.parent / 'workloads.json'
    if workload_path.exists():
        workloads = json.loads(workload_path.read_text(encoding='utf-8'))
    else:
        workloads = prepare_workloads(tok, a.source, workload_path, a.source_length)
    assert all(len(w['source_tokens']) == a.source_length for w in workloads), 'Existing workload has a different source length; use a new output root'
    print('Loading local BF16 backbone and shared FP32 LoRA', flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
             attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval()
    model, adapter_meta = attach_adapter(model, a.adapter)
    assert all(p.device.type == 'cuda' for p in model.parameters())
    cm, rp = CoMem(model, 12, tokenizer=tok), CoMem(model, 0, tokenizer=tok)
    assert not cm.write_sink and not cm.block_diagonal
    bos = model.config.bos_token_id
    meta = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'python': platform.python_version(), 'platform': platform.platform(),
            'adapter': adapter_meta, 'adapter_config': json.loads((a.adapter/'adapter_config.json').read_text()),
            'admission': admission, 'cpu_threads': 2, 'interop_threads': 16,
            'attention': 'SDPA with explicit repeat_kv, all arms', 'batch_size': 1,
            'memory': 'PyTorch allocated peak including weights, GB=10^9 bytes; reserved peak also recorded',
            'checkpoint_scope': 'principal rank-32 suffix adapter, 4000-step checkpoint',
            'read_boundary': 'GPU-ready embeddings/residuals through cached prefill to first logits; no fetch/query Write',
            'ttft_boundary': 'pretokenized query -> online iterative BM25 -> CPU-pinned fetch -> sink/query Write -> cached prefill -> first logits',
            'excluded': 'tokenization, document Write, model load, decode, external I/O; CPU selector rebuild is included in TTFT',
            'ttft_budget': 'fixed k=10/12; no equal-latency calibration on this GPU'}
    dump(a.out / 'metadata.json', meta)
    # One numerical check exercises the actual shared-adapter j=0 path.
    check_ids = torch.tensor([[bos] + workloads[0]['source_tokens'][:31]], device='cuda')
    stock = model(input_ids=check_ids, use_cache=False, logits_to_keep=1).logits
    split = rp.read_core(None, [], rp.write_chunk(check_ids), logits_tail=1)
    check = {'max_abs_logit_difference': float((stock-split).abs().max()),
             'same_top1': bool(stock.argmax(-1).eq(split.argmax(-1)).all())}
    assert check['same_top1'] and check['max_abs_logit_difference'] <= 0.125, check
    dump(a.out/'correctness.json', check)
    del stock, split, check_ids
    records = []
    rng = random.Random(42 + a.process)
    arms = [('replay_k12', rp, 12), ('replay_k10', rp, 10), ('comem_k12', cm, 12)]
    for wi, w in enumerate(workloads):
        chunks = list(torch.tensor(w['source_tokens'], dtype=torch.long).split(512))
        raw = [c.pin_memory() for c in chunks]
        residual = []
        # Fully write the source offline, retaining only a CPU-pinned store.
        for c in chunks:
            h = cm.write_chunk(c).to('cpu').pin_memory()
            residual.append(h)
        torch.cuda.synchronize()
        gc.collect(); torch.cuda.empty_cache()
        for phase in ('read', 'ttft'):
            arm_order = arms.copy()
            rng.shuffle(arm_order)
            for name, reader, k in arm_order:
                selected = w['selected'][str(k)]
                ready = None
                if phase == 'read':
                    hs = [residual[i].to('cuda', non_blocking=True) if reader.resume_j else reader.write_chunk(raw[i]) for i in selected]
                    ready = (reader.write_chunk([bos]), hs, reader.write_chunk(w['query_tokens']))
                    torch.cuda.synchronize()

                def operation():
                    if phase == 'read':
                        return reader.read_prefill(*ready)
                    ix = iter_bm25_indices(chunks, w['selector_query_tokens'], k, iter_hop_topk=2)
                    assert ix == selected
                    hh = [residual[i].to('cuda', non_blocking=True) if reader.resume_j else reader.write_chunk(raw[i]) for i in ix]
                    sink = reader.write_chunk([bos])
                    query, bottom, _ = reader.write_prefill(w['query_tokens'])
                    result = reader.read_prefill(sink, hh, query)
                    return result, bottom

                for _ in range(a.warmups):
                    result = operation(); torch.cuda.synchronize(); del result
                gc.collect()
                for rep in range(a.reps):
                    torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    result = operation()
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - start) * 1000
                    peak = torch.cuda.max_memory_allocated()
                    row = {'process': a.process, 'workload': wi, 'book_index': w['book_index'],
                           'phase': phase, 'arm': name, 'rep': rep, 'latency_ms': ms,
                           'peak_allocated_bytes': peak, 'baseline_allocated_bytes': baseline,
                           'incremental_peak_bytes': peak-baseline,
                           'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                           'pack_tokens': 1+k*512+512, 'selected_chunks': selected}
                    records.append(row)
                    with (a.out/'records.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(row)+'\n')
                    del result
                subset = records[-a.reps:]
                print(json.dumps({'workload': wi, 'phase': phase, 'arm': name,
                      'median_ms': statistics.median(r['latency_ms'] for r in subset),
                      'max_peak_GB': max(r['peak_allocated_bytes'] for r in subset)/1e9}), flush=True)
                del ready
                if phase == 'read':
                    del hs
                gc.collect(); torch.cuda.empty_cache()
        del residual, raw, chunks
        gc.collect(); torch.cuda.empty_cache()
    dump(a.out/'complete.json', {'complete': True, 'records': len(records), 'process': a.process})


if __name__ == '__main__':
    main()
