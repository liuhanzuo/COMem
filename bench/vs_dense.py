#!/usr/bin/env python
"""CoMem vs Dense — head-to-head speed + accuracy benchmark, and a decode
correctness gate.

Modes (``--mode``):

* ``profile``     — CoMem per-phase timing: WRITE serial (per-chunk) vs BATCHED
                    (``write_chunks``), READ prefill, and per-step KV-cache DECODE.
* ``speed``       — Dense (stock ``model.generate`` over the FULL context) vs
                    CoMem prefill time / decode tok-s / peak GPU memory at each
                    ``--context_lengths`` bucket. Dense OOMs / exceeds the RoPE
                    window past ~32-64k (recorded as ``OOM``); CoMem stays constant.
* ``accuracy``    — RULER ``niah_single`` recall, Dense vs CoMem, per length.
* ``correctness`` — KV-cache decode logits == recompute decode logits (per-step
                    argmax identical, max|diff| < tol). Runs on a tiny random
                    Qwen3 on CPU (fp32) or the real model.

Examples
--------
    python -m bench.vs_dense --mode correctness --tiny --device cpu
    python -m bench.vs_dense --mode all --model_path /path/to/Qwen3-8B \\
        --resume_j 12 --topk 12 --context_lengths 8k 16k 32k 64k 128k --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comem import CoMem                                    # noqa: E402
from comem.selftest import build_tiny_qwen3, _TinyTok      # noqa: E402

_LEN_TOKENS = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192,
               "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gb(device):
    return (torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else 0.0)


def _is_oom(err):
    return isinstance(err, torch.cuda.OutOfMemoryError) or (
        isinstance(err, RuntimeError) and "out of memory" in str(err).lower())


def _cleanup(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# decode correctness: KV-cache decode logits == recompute decode logits
# --------------------------------------------------------------------------- #
def run_correctness(cm, tokenizer, input_ids, *, chunk_size, topk, selector,
                    max_new_tokens, tol, sink_tokens="bos"):
    bare_q_ids = input_ids[0].tolist()[-8:]

    def gen(use_kv):
        stats = {"capture_step_logits": True}
        out = cm.generate_from_ids(
            input_ids, chunk_size=chunk_size, max_new_tokens=max_new_tokens,
            selector=selector, topk=topk, sink_tokens=sink_tokens,
            bare_question_ids=bare_q_ids, stats=stats, use_kv_cache=use_kv,
            tokenizer=tokenizer)
        return out, stats

    out_base, st_base = gen(False)
    out_kv, st_kv = gen(True)
    lb, lk = st_base["step_logits"], st_kv["step_logits"]
    gb, gk = st_base["generated_ids"], st_kv["generated_ids"]
    n = min(len(lb), len(lk))
    max_diff = max((lb[s] - lk[s]).abs().max().item() for s in range(n)) if n else 0.0
    argmax_ok = all(int(lb[s].argmax()) == int(lk[s].argmax()) for s in range(n))
    ok = (len(lb) == len(lk)) and (gb == gk) and argmax_ok and (max_diff < tol)
    print("=" * 72)
    print("CoMem decode correctness: KV-cache vs recompute (per-step logits)")
    print(f"  steps recompute={len(lb)} kv={len(lk)}  tokens identical={gb == gk}")
    print(f"  top-1 argmax identical every step: {argmax_ok}")
    print(f"  max|logit diff| over {n} steps: {max_diff:.3e}  (tol {tol:.1e})")
    print(f"CORRECTNESS: {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    return ok


# --------------------------------------------------------------------------- #
# dense timing / answering
# --------------------------------------------------------------------------- #
def _dense_time(model, tokenizer, input_ids, *, max_new_tokens, device):
    gk = dict(do_sample=False, num_beams=1,
              pad_token_id=(tokenizer.pad_token_id
                            if tokenizer.pad_token_id is not None else 0))

    def once(mnt):
        _sync(device)
        t0 = time.perf_counter()
        model.generate(input_ids, max_new_tokens=mnt, min_new_tokens=mnt, **gk)
        _sync(device)
        return time.perf_counter() - t0

    try:
        once(1)
        _reset_peak(device)
        t1, tN = once(1), once(max_new_tokens)
    except Exception as e:
        if _is_oom(e):
            _cleanup(device)
            return {"oom": True}
        raise
    dps = (tN - t1) / max(1, max_new_tokens - 1)
    return {"oom": False, "prefill_s": max(0.0, t1 - dps),
            "decode_tok_s": (1.0 / dps if dps > 0 else 0.0), "peak_gb": _peak_gb(device)}


@torch.no_grad()
def _dense_answer(model, tokenizer, input_ids, *, max_new_tokens, device):
    try:
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, pad_token_id=(tokenizer.pad_token_id
                             if tokenizer.pad_token_id is not None else 0))
    except Exception as e:
        if _is_oom(e):
            _cleanup(device)
            return None
        raise
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def _comem_time(cm, tokenizer, input_ids, *, chunk_size, topk, selector,
                sink_tokens, max_new_tokens, device, bare_q_ids):
    def once(mnt, stats=None):
        _sync(device)
        t0 = time.perf_counter()
        cm.generate_from_ids(input_ids, chunk_size=chunk_size, max_new_tokens=mnt,
                             selector=selector, topk=topk, sink_tokens=sink_tokens,
                             bare_question_ids=bare_q_ids, use_kv_cache=True,
                             stats=stats, tokenizer=tokenizer)
        _sync(device)
        return time.perf_counter() - t0

    st = {}
    try:
        once(1, stats=st)
        _reset_peak(device)
        t1, tN = once(1), once(max_new_tokens)
    except Exception as e:
        if _is_oom(e):
            _cleanup(device)
            return {"oom": True}
        raise
    dps = (tN - t1) / max(1, max_new_tokens - 1)
    return {"oom": False, "prefill_s": max(0.0, t1 - dps),
            "decode_tok_s": (1.0 / dps if dps > 0 else 0.0),
            "peak_gb": _peak_gb(device), "read_len": st.get("read_len")}


def run_profile(cm, tokenizer, *, lengths, chunk_size, topk, device, vocab,
                decode_steps=16):
    print("=" * 78)
    print(f"CoMem per-phase profile (resume_j={cm.resume_j}, topk={topk}, chunk={chunk_size})")
    print(f"{'ctx':>6} | {'#ctx':>5} | {'w_serial':>9} | {'w_batch':>8} | "
          f"{'speedup':>7} | {'read':>8} | {'dec/step':>8}")
    g = torch.Generator(device="cpu").manual_seed(0)
    bos = tokenizer.bos_token_id or 1
    for L in lengths:
        n_tok = _LEN_TOKENS.get(L) or int(L)
        ids = torch.randint(2, vocab, (1, n_tok), generator=g).to(device)
        chunks = list(ids[0].split(chunk_size))
        ctx, query_chunk = chunks[:-1], chunks[-1]
        sel = ctx[-topk:] if topk < len(ctx) else ctx
        cm.write_chunk(sel[0]); _sync(device)
        _sync(device); t0 = time.perf_counter()
        _ = [cm.write_chunk(c) for c in sel]
        _sync(device); t_serial = time.perf_counter() - t0
        _sync(device); t0 = time.perf_counter()
        sel_hj = cm.write_chunks(sel)
        _sync(device); t_batch = time.perf_counter() - t0
        sink_hj = cm.write_chunk([bos]); q_hj = cm.write_chunk(query_chunk)
        _sync(device); t0 = time.perf_counter()
        logits1, top_cache, pack_pos = cm.read_prefill(sink_hj, sel_hj, q_hj)
        _sync(device); t_read = time.perf_counter() - t0
        _, bottom_cache, q_local = cm.write_prefill(query_chunk.tolist())
        tok = int(logits1[0, -1].argmax())
        _sync(device); t0 = time.perf_counter()
        for _ in range(decode_steps):
            lg = cm.decode_step(tok, bottom_cache, top_cache, q_local, pack_pos)
            q_local += 1; pack_pos += 1; tok = int(lg[0, -1].argmax())
        _sync(device); t_dec = (time.perf_counter() - t0) / decode_steps
        sp = t_serial / t_batch if t_batch > 0 else float("nan")
        print(f"{L:>6} | {len(ctx):>5} | {t_serial*1e3:>7.1f}ms | {t_batch*1e3:>6.1f}ms | "
              f"{sp:>6.1f}x | {t_read*1e3:>6.1f}ms | {t_dec*1e3:>6.1f}ms")
        _cleanup(device)


def run_speed(model, cm, tokenizer, *, lengths, chunk_size, topk, selector,
              sink_tokens, max_new_tokens, device, vocab):
    print("=" * 78)
    print(f"Dense vs CoMem SPEED (resume_j={cm.resume_j}, topk={topk}, mnt={max_new_tokens})")
    print(f"{'ctx':>6} | {'D_pref':>8} {'D_tok/s':>8} {'D_mem':>6} | "
          f"{'Q_pref':>8} {'Q_tok/s':>8} {'Q_mem':>6} {'read':>6}")
    g = torch.Generator(device="cpu").manual_seed(0)
    for L in lengths:
        n_tok = _LEN_TOKENS.get(L) or int(L)
        ids = torch.randint(2, vocab, (1, n_tok), generator=g).to(device)
        bare_q = ids[0].tolist()[-8:]
        d = _dense_time(model, tokenizer, ids, max_new_tokens=max_new_tokens, device=device)
        _cleanup(device)
        q = _comem_time(cm, tokenizer, ids, chunk_size=chunk_size, topk=topk,
                        selector=selector, sink_tokens=sink_tokens,
                        max_new_tokens=max_new_tokens, device=device, bare_q_ids=bare_q)
        _cleanup(device)
        dstr = "OOM" if d.get("oom") else f"{d['prefill_s']:.2f}s {d['decode_tok_s']:.1f} {d['peak_gb']:.1f}G"
        qstr = "OOM" if q.get("oom") else f"{q['prefill_s']:.2f}s {q['decode_tok_s']:.1f} {q['peak_gb']:.1f}G {q['read_len']}"
        print(f"{L:>6} | {dstr:>26} | {qstr}")


def run_accuracy(model, cm, tokenizer, *, lengths, chunk_size, topk, selector,
                 sink_tokens, n_acc, max_new_tokens, device, base_seed, task):
    from eval import ruler
    print("=" * 78)
    print(f"Dense vs CoMem ACCURACY RULER {task} string_match_all (n={n_acc}, {selector})")
    print(f"{'ctx':>6} | {'Dense':>8} | {'CoMem':>8} | {'n':>4}")
    rt = ruler._resolve_task(task)
    for L in lengths:
        target = ruler._LENGTH_TOKENS.get(L, _LEN_TOKENS.get(L, int(L)))
        d_sum = q_sum = 0.0
        n_done = 0
        d_oom = False
        for k in range(n_acc):
            rng = random.Random(base_seed + k)
            prompt, answers, gold = ruler._build_sample(rt, target, tokenizer, rng, None)
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = enc.input_ids.to(device)
            bare_q = tokenizer(ruler._bare_question(prompt),
                               add_special_tokens=False).input_ids
            if not d_oom:
                d_out = _dense_answer(model, tokenizer, input_ids,
                                      max_new_tokens=max_new_tokens, device=device)
                if d_out is None:
                    d_oom = True
                else:
                    d_sum += ruler._string_match_all_one(d_out, answers)
            _cleanup(device)
            q_out = cm.generate_from_ids(
                input_ids, chunk_size=chunk_size, max_new_tokens=max_new_tokens,
                selector=selector, topk=topk, sink_tokens=sink_tokens,
                bare_question_ids=bare_q, use_kv_cache=True)
            q_sum += ruler._string_match_all_one(q_out, answers)
            n_done += 1
            _cleanup(device)
        d_acc = "OOM" if d_oom else f"{100.0 * d_sum / n_done:.1f}"
        print(f"{L:>6} | {d_acc:>8} | {100.0 * q_sum / n_done:>7.1f} | {n_done:>4}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["profile", "speed", "accuracy", "correctness",
                                       "both", "all"], default="all")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--tiny_layers", type=int, default=6)
    ap.add_argument("--tiny_hidden", type=int, default=64)
    ap.add_argument("--tiny_vocab", type=int, default=256)
    ap.add_argument("--resume_j", type=int, default=12)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--selector", default="bm25")
    ap.add_argument("--sink_tokens", default="bos", choices=["bos", "none"])
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--context_lengths", nargs="+", default=["8k", "16k", "32k", "64k", "128k"])
    ap.add_argument("--acc_lengths", nargs="+", default=["8k", "16k", "32k"])
    ap.add_argument("--acc_task", default="niah_single")
    ap.add_argument("--n_acc", type=int, default=30)
    ap.add_argument("--acc_seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--tol", type=float, default=-1.0)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    if args.tiny or not args.model_path:
        print("[bench] tiny random Qwen3 (fp32) — plumbing smoke only")
        model, cfg = build_tiny_qwen3(args.tiny_layers, args.tiny_hidden, args.tiny_vocab)
        model = model.to(device=device, dtype=torch.float32).eval()
        vocab, tokenizer = args.tiny_vocab, _TinyTok(args.tiny_vocab)
        L = args.tiny_layers
        resume_j = max(0, min(args.resume_j if args.resume_j <= L else L // 2, L))
        chunk_size = min(args.chunk_size, 8)
        is_tiny = True
        tol = args.tol if args.tol > 0 else 1e-4
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                                  local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
            trust_remote_code=True, local_files_only=True).to(device).eval()
        vocab = int(model.config.vocab_size)
        L = int(model.config.num_hidden_layers)
        resume_j = max(0, min(args.resume_j, L))
        chunk_size = args.chunk_size
        is_tiny = False
        tol = args.tol if args.tol > 0 else (1e-4 if dtype == torch.float32 else 1e-2)

    cm = CoMem(model, resume_j=resume_j, tokenizer=tokenizer)
    print(f"[bench] backbone L={L} resume_j={resume_j} chunk_size={chunk_size} device={device}")

    speed_lengths = ["1k", "2k"] if is_tiny else args.context_lengths
    ok = True
    if args.mode in ("correctness", "all"):
        cs = chunk_size
        g = torch.Generator(device="cpu").manual_seed(1234)
        ids = torch.randint(2, vocab, (1, 5 * cs + 6), generator=g).to(device)
        ok = run_correctness(cm, tokenizer, ids, chunk_size=cs, topk=args.topk,
                             selector="recency", max_new_tokens=min(args.max_new_tokens, 24),
                             tol=tol, sink_tokens=args.sink_tokens)
    if args.mode in ("profile", "all"):
        run_profile(cm, tokenizer, lengths=speed_lengths, chunk_size=chunk_size,
                    topk=args.topk, device=device, vocab=vocab,
                    decode_steps=min(16, args.max_new_tokens))
    if args.mode in ("speed", "both", "all"):
        run_speed(model, cm, tokenizer, lengths=speed_lengths, chunk_size=chunk_size,
                  topk=args.topk, selector=args.selector, sink_tokens=args.sink_tokens,
                  max_new_tokens=args.max_new_tokens, device=device, vocab=vocab)
    if args.mode in ("accuracy", "both", "all"):
        if is_tiny:
            print("[bench] accuracy skipped for tiny model (needs a real tokenizer)")
        else:
            run_accuracy(model, cm, tokenizer, lengths=args.acc_lengths,
                         chunk_size=chunk_size, topk=args.topk, selector=args.selector,
                         sink_tokens=args.sink_tokens, n_acc=args.n_acc,
                         max_new_tokens=args.max_new_tokens, device=device,
                         base_seed=args.acc_seed, task=args.acc_task)
    if args.mode in ("correctness", "all") and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
