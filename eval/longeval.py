#!/usr/bin/env python
"""CoMem — LongEval (LongChat lines-retrieval) eval driver.

The cleanest single-hop exact-retrieval benchmark: a record of N lines

    line <random-label>: REGISTER_CONTENT is <6-digit number>

after which the model returns the REGISTER_CONTENT of one queried line. bm25 has a
rare, discriminative needle (the queried line label) to lock onto. Thin: build
CoMem, ``generate_from_ids``, judge by exact-value match. Self-contained (prompt
synthesis + judging embedded, ported verbatim).

Usage:
    python -m eval.longeval --model_path /path/to/Qwen3-8B --resume_j 12 \\
        --selector bm25 --topk 12 --lengths 4k 8k 16k 32k --num_samples 50 \\
        --output_dir longeval_results/comem_j12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import zlib
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comem import CoMem                          # noqa: E402
from comem import selectors as _sel              # noqa: E402
from eval._common import load_backbone, resolve_baseline  # noqa: E402

_LENGTH_TOKENS = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192,
                  "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}
_PROMPT_HEADER = (
    "Below is a record of lines I want you to remember. Each line begins with "
    "'line <line index>' and contains a '<REGISTER_CONTENT>' at the end of the "
    "line as a numerical value. For each line index, memorize its corresponding "
    "<REGISTER_CONTENT>. At the end of the record, I will ask you to retrieve the "
    "corresponding <REGISTER_CONTENT> of a certain line index. Now the record "
    "start:\n\n"
)
_NUM_RE = re.compile(r"\d{4,}")


def _random_label(rng):
    def word():
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(5, 9)))
    return f"{word()}-{word()}"


def build_lines_prompt(target_tokens, tokenizer, rng):
    """Build one lines-retrieval sample sized to ~target_tokens. Returns
    (prompt, expected_value, target_label, n_lines)."""
    labels, lines, values = [], [], []

    def render(query_label):
        query = (f"\nNow the record is over. Tell me what is the <REGISTER_CONTENT> in "
                 f"line {query_label}? I need the number.\nThe <REGISTER_CONTENT> in line "
                 f"{query_label} is")
        return _PROMPT_HEADER + "".join(lines) + query

    while True:
        for _ in range(64):
            label = _random_label(rng)
            value = str(rng.randint(100000, 999999))
            labels.append(label)
            values.append(value)
            lines.append(f"line {label}: REGISTER_CONTENT is <{value}>\n")
        approx = render(labels[len(labels) // 2])
        if len(tokenizer.encode(approx, add_special_tokens=True)) >= target_tokens:
            break
    ti = rng.randrange(len(labels))
    return render(labels[ti]), values[ti], labels[ti], len(labels)


def extract_prediction(output):
    m = _NUM_RE.search(output)
    return m.group(0) if m else ""


def _oracle_needle_chunks(input_ids, expected_value, target_label, tokenizer, chunk_size):
    probes = []
    if expected_value:
        probes.extend([expected_value, f"<{expected_value}>"])
    if target_label:
        probes.append(f"line {target_label}")
    for probe in probes:
        if not probe:
            continue
        chunks = _sel.locate_needle_chunks(input_ids, probe, tokenizer, chunk_size)
        if chunks:
            return chunks
    return None


def main():
    p = argparse.ArgumentParser(description="CoMem LongEval eval")
    p.add_argument("--model_path", required=True)
    p.add_argument("--resume_j", type=int, default=12)
    p.add_argument("--top_prepay_b", type=int, default=0)
    p.add_argument("--reuse_kv_blockdiag", action="store_true", default=False)
    p.add_argument("--lora_adapter", default="")
    p.add_argument("--baseline", default="none", choices=["none", "kvdirect", "hcache"])
    p.add_argument("--selector", default="bm25",
                   choices=["bm25", "recency", "oracle", "reader_attn"])
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--sink_tokens", default="bos", choices=["bos", "none"])
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--lengths", nargs="+", default=["4k", "8k", "16k", "32k"])
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--output_dir", default="longeval_results/comem")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", default="sdpa")
    args = p.parse_args()

    resume_j, no_retrieval, mode, lora = resolve_baseline(
        args.baseline, args.resume_j, args.lora_adapter)
    model, tok = load_backbone(args.model_path, args.dtype, args.attn_impl,
                               args.device, lora)
    cm = CoMem(model, resume_j=resume_j, top_prepay_b=args.top_prepay_b,
               block_diagonal=args.reuse_kv_blockdiag, tokenizer=tok)
    device = torch.device(args.device)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    summary = {}
    for length in args.lengths:
        if length not in _LENGTH_TOKENS:
            continue
        target = _LENGTH_TOKENS[length]
        length_seed = args.seed + (zlib.crc32(length.encode()) % 100000)
        sample_indices = list(range(args.num_samples))[args.shard_index::args.num_shards]
        records, correct, total = [], 0, 0
        for i in tqdm(sample_indices, desc=length, leave=False):
            rng = random.Random(length_seed * 1000 + i)
            prompt, expected, target_label, n_lines = build_lines_prompt(target, tok, rng)
            ids = tok.encode(prompt, add_special_tokens=True, return_tensors="pt")
            if isinstance(ids, list):
                ids = torch.tensor([ids], dtype=torch.long)
            input_ids = ids.to(device)
            bare_q_ids = tok.encode(f"line {target_label}", add_special_tokens=False)
            needle_set = None
            if args.selector == "oracle":
                needle_set = _oracle_needle_chunks(
                    input_ids, expected, target_label, tok, args.chunk_size)
            try:
                out = cm.generate_from_ids(
                    input_ids, chunk_size=args.chunk_size,
                    max_new_tokens=args.max_new_tokens, selector=args.selector,
                    topk=args.topk, sink_tokens=args.sink_tokens,
                    needle_chunk_set=needle_set, bare_question_ids=bare_q_ids,
                    no_retrieval=no_retrieval)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                out = "[OOM]"
                torch.cuda.empty_cache()
            pred = extract_prediction(out)
            ok = (pred == expected)
            correct += int(ok)
            total += 1
            records.append({"sample_index": i, "label": target_label,
                            "expected": expected, "output": out, "pred": pred,
                            "correct": ok, "n_lines": n_lines})
        acc = correct / total if total else 0.0
        summary[length] = {"accuracy": round(acc, 4), "correct": correct, "total": total}
        with open(outdir / f"longeval_{length}{shard_tag}.json", "w") as f:
            json.dump({"length": length, "summary": summary[length],
                       "records": records}, f, indent=2)
        print(f"[CoMem-LongEval] {length}: acc={acc:.3f} ({correct}/{total})")
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[CoMem-LongEval] done.")


if __name__ == "__main__":
    main()
