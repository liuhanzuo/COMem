#!/usr/bin/env python
"""CoMem — RULER (NIAH needle-in-a-haystack + variable_tracking) eval driver.

Thin: build a :class:`comem.CoMem`, chunk each RULER prompt (query == trailing
chunk), ``generate_from_ids`` (write chunks -> select topk -> resume -> decode),
score with RULER ``string_match_all`` recall. Self-contained: the RULER sample
synthesis + scoring are embedded here (ported verbatim from the research repo,
which reproduces NVIDIA/RULER's constants). No dependency outside ``comem``.

The NIAH prose haystack ("niah_single_2" / "niah_multikey_1") uses a natural-text
corpus at ``--essay_path`` (one big text/JSONL file); if absent it falls back to
the noise haystack. Scoring = fraction of reference strings that appear as a
case-insensitive substring of the output (RULER ``string_match_all``).

Usage:
    python -m eval.ruler --model_path /path/to/Qwen3-8B \\
        --resume_j 12 --selector bm25 --topk 12 \\
        --ruler_tasks niah_single niah_multi vt --lengths 4k 8k 16k 32k \\
        --limit 50 --output_dir ruler_results/comem_j12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comem import CoMem                          # noqa: E402
from comem import selectors as _sel              # noqa: E402
from eval import _cli                            # noqa: E402
from eval._common import (load_backbone, resolve_baseline, write_results_csv,  # noqa: E402
                          dense_generate, DENSE_MODES)

# --------------------------------------------------------------------------- #
# RULER constants (verbatim from NVIDIA/RULER data/synthetic/constants.py)
# --------------------------------------------------------------------------- #
NIAH_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} "
    "afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for "
    "{query} mentioned in the provided text?"
)
NIAH_ANSWER_PREFIX = (
    " The special magic {type_needle_v} for {query} mentioned in the provided "
    "text are"
)
NIAH_NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
VT_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the "
    "following text.\n\n{context}\nQuestion: Find all variables that are "
    "assigned the value {query} in the text above."
)
VT_ANSWER_PREFIX = (
    " Answer: According to the chain(s) of variable assignment in the text "
    "above, {num_v} variables are assigned the value {query}, they are: "
)
NOISE_HAYSTACK = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again."
)
_LENGTH_TOKENS = {
    "1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384,
    "32k": 32768, "64k": 65536, "128k": 131072, "256k": 262144,
}
DEPTHS = [int(round(x)) for x in [i * 100 / 39 for i in range(40)]]

_ESSAY_WORDS_CACHE = None
_ESSAY_PATH = None
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _load_essay_words():
    """Load a chunk of natural prose (whitespace-split word list) from
    ``--essay_path``. Returns None if no corpus is available (caller uses noise)."""
    global _ESSAY_WORDS_CACHE
    if _ESSAY_WORDS_CACHE is not None:
        return _ESSAY_WORDS_CACHE
    if not _ESSAY_PATH or not os.path.exists(_ESSAY_PATH):
        return None
    with open(_ESSAY_PATH, "r", errors="ignore") as f:
        text = f.read(8_000_000)
    text = re.sub(r"\s+", " ", text).strip()
    _ESSAY_WORDS_CACHE = text.split(" ")
    return _ESSAY_WORDS_CACHE


def _sent_tokenize(text):
    return [s for s in _SENT_RE.split(text.strip()) if s]


def _rand_word(rng):
    def w():
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 8)))
    return f"{w()}-{w()}"


def _rand_number(rng, num_digits=7):
    return str(rng.randint(10 ** (num_digits - 1), 10 ** num_digits - 1))


def _make_niah(num_haystack, type_haystack, num_needle_k, rng):
    keys, values, needles = [], [], []
    for _ in range(num_needle_k):
        k = _rand_word(rng)
        v = _rand_number(rng)
        keys.append(k)
        values.append([v])
        needles.append(NIAH_NEEDLE.format(type_needle_v="numbers", key=k, value=v))
    random.Random(rng.randint(0, 10 ** 9)).shuffle(needles)

    essay_words = _load_essay_words() if type_haystack == "essay" else None
    if type_haystack == "noise" or essay_words is None:
        sentences = [NOISE_HAYSTACK] * num_haystack
        idxs = sorted(rng.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(idxs, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)
    else:
        words = essay_words
        if num_haystack > len(words):
            reps = (num_haystack + len(words) - 1) // len(words)
            text = " ".join((words * reps)[:num_haystack])
        else:
            text = " ".join(words[:num_haystack])
        sents = _sent_tokenize(text) or [text]
        chosen = rng.sample(DEPTHS, min(len(needles), len(DEPTHS)))
        ins = [0] + sorted(int(len(sents) * (d / 100)) for d in chosen) + [len(sents)]
        parts = []
        for i in range(1, len(ins)):
            parts.append(" ".join(sents[ins[i - 1]:ins[i]]))
            if i - 1 < len(needles):
                parts.append(needles[i - 1])
        context = " ".join(parts)

    query = keys[0]
    answers = values[0]
    gold_needle = NIAH_NEEDLE.format(type_needle_v="numbers", key=keys[0],
                                     value=values[0][0])
    return context, query, answers, gold_needle


def _render_niah(context, query):
    full = NIAH_TEMPLATE + NIAH_ANSWER_PREFIX
    full = full.replace("Some", "A").replace("are all", "is").replace("are", "is")
    full = full.replace("answers", "answer")
    return full.format(type_needle_v="number", context=context, query=query)


def _gen_chain(num_hops, rng, icl=False):
    k = 3 if icl else 5
    nvars = num_hops + 1
    vars_all = []
    while len(set(vars_all)) < nvars:
        vars_all.append("".join(rng.choices(string.ascii_uppercase, k=k)))
    vars_all = list(dict.fromkeys(vars_all))[:nvars]
    first_val = "12345" if icl else str(rng.randint(10000, 99999))
    chain = [f"VAR {vars_all[0]} = {first_val}"]
    for j in range(num_hops):
        chain.append(f"VAR {vars_all[j + 1]} = VAR {vars_all[j]} ")
    return vars_all, chain, first_val


def _make_vt(num_noises, num_hops, rng):
    vars_all, chain, value = _gen_chain(num_hops, rng)
    sentences = [NOISE_HAYSTACK] * num_noises
    positions = sorted(rng.sample(range(len(sentences)), len(chain)))
    for offset, (pos, c) in enumerate(zip(positions, chain)):
        sentences.insert(pos + offset, c)
    context = "\n".join(sentences).replace(". \n", ".\n")
    return context, value, vars_all, num_hops + 1


def _make_vt_icl(rng, num_hops):
    nh = min(num_hops, 10)
    vars_all, chain, value = _gen_chain(nh, rng, icl=True)
    sentences = [NOISE_HAYSTACK] * 5
    positions = sorted(rng.sample(range(len(sentences)), len(chain)))
    for offset, (pos, c) in enumerate(zip(positions, chain)):
        sentences.insert(pos + offset, c)
    ctx = "\n".join(sentences).replace(". \n", ".\n")
    body = VT_TEMPLATE.format(context=ctx, query=value)
    prefix = VT_ANSWER_PREFIX.format(num_v=nh + 1, query=value)
    return body + prefix + " ".join(vars_all) + "\n"


def _render_vt(context, query, num_v, icl_block):
    body = VT_TEMPLATE.format(context=context, query=query)
    prefix = VT_ANSWER_PREFIX.format(num_v=num_v, query=query)
    return (icl_block + body + prefix) if icl_block else (body + prefix)


def _build_sample(task, target_tokens, tokenizer, rng, vt_icl):
    if task == "variable_tracking":
        num_hops, incremental = 4, 5
        def render(n):
            ctx, val, vars_all, num_v = _make_vt(n, num_hops, rng)
            return _render_vt(ctx, val, num_v, vt_icl or ""), vars_all, None
    else:
        if task == "niah_single_1":
            type_haystack, num_k = "noise", 1
        elif task == "niah_single_2":
            type_haystack, num_k = "essay", 1
        elif task == "niah_multikey_1":
            type_haystack, num_k = "essay", 4
        else:
            raise ValueError(f"unknown task {task}")
        incremental = 25 if type_haystack == "noise" else 500
        def render(n):
            ctx, query, answers, gold = _make_niah(n, type_haystack, num_k, rng)
            return _render_niah(ctx, query), answers, gold

    n = incremental
    last_ok = None
    text, answers, gold = render(n)
    while True:
        text, answers, gold = render(n)
        ntok = len(tokenizer.encode(text, add_special_tokens=True))
        if ntok >= target_tokens:
            break
        last_ok = (text, answers, gold)
        n += incremental if n < incremental * 8 else incremental * 4
        if n > 400_000:
            break
    lo, hi = max(incremental, n // 2), n
    best = last_ok or (text, answers, gold)
    while lo <= hi:
        mid = (lo + hi) // 2
        text, answers, gold = render(mid)
        ntok = len(tokenizer.encode(text, add_special_tokens=True))
        if ntok <= target_tokens:
            best = (text, answers, gold)
            lo = mid + incremental
        else:
            hi = mid - incremental
    return best


def _string_match_all_one(pred, refs):
    pl = pred.lower()
    return sum(1.0 for r in refs if r.lower() in pl) / len(refs)


def _bare_question(prompt):
    return prompt[prompt.rfind("\n") + 1:].strip()


_TASK_ALIAS = {
    "niah_single": "niah_single_2", "niah_single_noise": "niah_single_1",
    "niah_single_essay": "niah_single_2", "niah_multi": "niah_multikey_1",
    "niah_multikey": "niah_multikey_1", "vt": "variable_tracking",
}
_CANONICAL = {"niah_single_1", "niah_single_2", "niah_multikey_1", "variable_tracking"}


def _resolve_task(name):
    if name in _CANONICAL:
        return name
    if name in _TASK_ALIAS:
        return _TASK_ALIAS[name]
    raise ValueError(f"unknown ruler task {name!r}")


def main():
    global _ESSAY_PATH
    p = argparse.ArgumentParser(description="CoMem RULER eval")
    p.add_argument("--model", "--model_path", dest="model_path", required=True)
    p.add_argument("--j", "--resume_j", dest="resume_j", type=_cli.j_type, default=12,
                   help="split depth (int) or 'auto' (per-model, see model_registry)")
    p.add_argument("--top_prepay_b", type=int, default=0)
    p.add_argument("--reuse_kv_blockdiag", action="store_true", default=False)
    p.add_argument("--adapter", "--lora_adapter", dest="lora_adapter", default="")
    p.add_argument("--baseline", default="none", choices=_cli.BASELINE_CHOICES)
    p.add_argument("--selector", default="iter_bm25_adaptive",
                   choices=["bm25", "recency", "oracle", "reader_attn",
                            "iter_reader_attn", "iter_bm25", "iter_bm25_adaptive"],
                   help="Chunk selector. Default 'iter_bm25_adaptive' is the "
                        "single universal selector for ALL tasks: it self-degrades "
                        "via a confidence stop (--iter_conf_ratio) so chain-free "
                        "tasks (niah_*) stop after round 1 (== single-shot bm25) "
                        "while chain tasks (variable_tracking) keep following the "
                        "VAR reference chain. Pass an explicit --selector "
                        "(bm25 / iter_bm25 / reader_attn / oracle / ...) to "
                        "override it on ALL tasks for controls.")
    p.add_argument("--iter_rounds", type=int, default=0)
    p.add_argument("--iter_hop_topk", type=int, default=4)
    p.add_argument("--iter_score", default="meanpool", choices=["meanpool", "maxsim"])
    p.add_argument("--iter_conf_ratio", type=float, default=0.3,
                   help="iter_bm25_adaptive: stop a hop when its best BM25 score "
                        "falls below this ratio times the round-1 best score.")
    p.add_argument("--iter_max_chunks", type=int, default=64,
                   help="iter_bm25_adaptive: hard cap on accumulated chunks.")
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--sink_tokens", default="bos", choices=["bos", "none"])
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--limit", "--n", dest="limit", type=int, default=500)  # paper default n=500/cell (aligns with official RULER)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--ruler_tasks", nargs="+", default=["niah_single", "niah_multi", "vt"])
    p.add_argument("--lengths", nargs="+", default=["4k", "8k", "16k", "32k"])
    p.add_argument("--essay_path", default="data/pg19_train.jsonl",
                   help="Natural-prose corpus for the NIAH essay haystack "
                        "(falls back to noise if absent).")
    p.add_argument("--output_dir", "--out", dest="output_dir", default="ruler_results/comem")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    _cli.normalize_args(args, task_attrs=("ruler_tasks",))
    _ESSAY_PATH = args.essay_path
    resume_j, no_retrieval, mode, lora = resolve_baseline(
        args.baseline, args.resume_j, args.lora_adapter)
    dense_mode = mode if mode in DENSE_MODES else None
    tasks = [_resolve_task(t) for t in args.ruler_tasks]

    model, tok = load_backbone(args.model_path, args.dtype, args.attn_impl,
                               args.device, lora)
    L = int(model.config.num_hidden_layers)
    cm = CoMem(model, resume_j=resume_j, top_prepay_b=args.top_prepay_b,
               block_diagonal=args.reuse_kv_blockdiag, tokenizer=tok)
    device = torch.device(args.device)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""
    summary = {}
    for task in tqdm(tasks, desc="tasks"):
        summary[task] = {}
        sel = args.selector
        for length in tqdm(args.lengths, desc="lengths", leave=False):
            if length not in _LENGTH_TOKENS:
                continue
            target = _LENGTH_TOKENS[length]
            print(f"[CoMem-RULER] {task}/{length}: selector={sel}")
            base_seed = args.seed + (hash((task, length)) % 100000)
            vt_icl = _make_vt_icl(random.Random(base_seed + 777), 4) \
                if task == "variable_tracking" else None
            sample_indices = set(range(args.limit))
            sample_indices = {i for i in range(args.limit)
                              if i % args.num_shards == args.shard_index}
            df = pd.DataFrame({"target": [], "output": [], "question": [], "recall": []})
            recall_sum, total = 0.0, 0
            mnt = args.max_new_tokens if task != "variable_tracking" \
                else max(args.max_new_tokens, 60)
            for i in tqdm(range(args.limit), desc=f"{task}/{length}", leave=False):
                rng = random.Random(base_seed * 1000 + i)
                prompt, answers, gold = _build_sample(task, target, tok, rng, vt_icl)
                if i not in sample_indices:
                    continue
                ids = tok.encode(prompt, add_special_tokens=True, return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                input_ids = ids.to(device)
                bare_q_ids = tok.encode(_bare_question(prompt), add_special_tokens=False)
                needle_set = None
                if sel == "oracle" and gold:
                    needle_set = _sel.locate_needle_chunks(
                        input_ids, gold, tok, args.chunk_size)
                try:
                    if dense_mode:
                        out = dense_generate(cm.model, tok, input_ids, dense_mode,
                                             max_new_tokens=mnt)
                    else:
                        out = cm.generate_from_ids(
                            input_ids, chunk_size=args.chunk_size, max_new_tokens=mnt,
                            selector=sel, topk=args.topk,
                            sink_tokens=args.sink_tokens, needle_chunk_set=needle_set,
                            bare_question_ids=bare_q_ids, no_retrieval=no_retrieval,
                            iter_rounds=args.iter_rounds, iter_hop_topk=args.iter_hop_topk,
                            iter_score=args.iter_score,
                            iter_conf_ratio=args.iter_conf_ratio,
                            iter_max_chunks=args.iter_max_chunks)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    out = "[OOM]"
                    torch.cuda.empty_cache()
                rec = _string_match_all_one(out, answers)
                recall_sum += rec
                total += 1
                df.loc[len(df)] = [" | ".join(answers), out, _bare_question(prompt), rec]
                if len(df) % 10 == 0:
                    write_results_csv(df, outdir / f"{task}_{length}{shard_tag}.csv")
            score = (recall_sum / total * 100.0) if total else 0.0
            summary[task][length] = {"score": round(score, 2), "n": total}
            write_results_csv(df, outdir / f"{task}_{length}{shard_tag}.csv")
            print(f"[CoMem-RULER] {task}/{length}: recall={score:.2f} ({total})")
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[CoMem-RULER] done.")


if __name__ == "__main__":
    main()
