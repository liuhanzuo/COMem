#!/usr/bin/env python
"""CoMem — LongBench (real long-document QA) eval driver.

Runs CoMem on LongBench's genuine long-context QA tasks (narrativeqa / qasper /
hotpotqa / 2wikimqa / musique / multifieldqa_en). Thin: build CoMem,
``generate_from_ids`` per sample, score with the official SQuAD-style token-F1 /
EM (``qa_f1_score``). Self-contained: prompt templates + F1/EM + JSONL loader +
shard merge scorer are embedded here (ported verbatim from the research repo).

Data: local JSONL at ``--data_dir/{ds}.jsonl`` (each line an object with
context/input/answers), falling back to HuggingFace ``THUDM/LongBench``.

Usage:
    python -m eval.longbench --model_path /path/to/Qwen3-8B --resume_j 12 \\
        --selector bm25 --topk 12 --tasks narrativeqa qasper hotpotqa 2wikimqa \\
        --output_dir longbench_results/comem_j12
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comem import CoMem                          # noqa: E402
from eval._common import load_backbone, resolve_baseline  # noqa: E402

DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question asconcisely as you can, using a single "
        "phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\n"
        "Now, answer the question based on the story asconcisely as you can, using "
        "a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as "
        "concisely as you can, using a single phrase or sentence if possible. If the "
        'question cannot be answered based on the information in the article, write '
        '"unanswerable". If the question is a yes/no question, answer "yes", "no", '
        'or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n'
        " Answer the question based on the above article as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be "
        'answered based on the information in the article, write "unanswerable". '
        'If the question is a yes/no question, answer "yes", "no", or "unanswerable". '
        "Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer "
        "the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
}
DATASET2MAXGEN = {"hotpotqa": 32, "narrativeqa": 128, "qasper": 128,
                  "multifieldqa_en": 64, "2wikimqa": 32, "musique": 32}
DEFAULT_DATASETS = list(DATASET2MAXGEN.keys())


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction, ground_truth):
    pt = normalize_answer(prediction).split()
    gt = normalize_answer(ground_truth).split()
    if len(pt) == 0 and len(gt) == 0:
        return 1.0
    if len(pt) == 0 or len(gt) == 0:
        return 0.0
    common = collections.Counter(pt) & collections.Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return (2 * precision * recall) / (precision + recall)


def compute_f1_multi(prediction, answers):
    return max((compute_f1(prediction, a) for a in answers), default=0.0)


def compute_em_multi(prediction, answers):
    np_ = normalize_answer(prediction)
    return max((1.0 if np_ == normalize_answer(a) else 0.0 for a in answers),
               default=0.0)


def load_longbench_dataset(dataset_name, datasets_list, data_dir):
    all_data = {}
    for ds_name in datasets_list:
        local_path = os.path.join(data_dir, f"{ds_name}.jsonl")
        samples = []
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    ans = item.get("answers", [])
                    if isinstance(ans, str):
                        try:
                            ans = json.loads(ans)
                        except json.JSONDecodeError:
                            ans = [ans]
                    if not isinstance(ans, list):
                        ans = [ans]
                    samples.append({"context": item.get("context", ""),
                                    "input": item.get("input", ""),
                                    "answers": ans, "dataset": ds_name})
            print(f"[LongBench] loaded {len(samples)} from {local_path}")
        else:
            try:
                import datasets as hf
                data = hf.load_dataset(dataset_name, ds_name, split="test",
                                       trust_remote_code=True)
                for item in data:
                    ans = item.get("answers", [])
                    if not isinstance(ans, list):
                        ans = [ans]
                    samples.append({"context": item.get("context", ""),
                                    "input": item.get("input", ""),
                                    "answers": ans, "dataset": ds_name})
                print(f"[LongBench] loaded {len(samples)} from HF for {ds_name}")
            except Exception as e:
                print(f"[LongBench] ERROR loading {ds_name}: {e}")
        all_data[ds_name] = samples
    return all_data


def format_prompt(sample, dataset_name):
    template = DATASET2PROMPT.get(dataset_name, DATASET2PROMPT["hotpotqa"])
    return template.format(context=sample["context"], input=sample["input"])


def run_scoring(output_dir, datasets_list):
    output_path = Path(output_dir)
    results = {}
    for ds_name in datasets_list:
        shard_files = sorted(output_path.glob(f"{ds_name}_*.jsonl"))
        if not shard_files:
            continue
        preds, seen = [], set()
        for sf in shard_files:
            with open(sf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    idx = item.get("index", len(preds))
                    if idx not in seen:
                        seen.add(idx)
                        preds.append(item)
        if not preds:
            continue
        f1s = [compute_f1_multi(x.get("pred", ""), x.get("answers", [])) for x in preds]
        ems = [compute_em_multi(x.get("pred", ""), x.get("answers", [])) for x in preds]
        results[ds_name] = {"f1": sum(f1s) / len(f1s) * 100,
                            "em": sum(ems) / len(ems) * 100, "n_samples": len(preds)}
        print(f"[LongBench] {ds_name:20s}: F1={results[ds_name]['f1']:.2f}  "
              f"EM={results[ds_name]['em']:.2f}  (n={len(preds)})")
    if results:
        allf1 = [v["f1"] for v in results.values()]
        results["AVERAGE"] = {"f1": sum(allf1) / len(allf1),
                              "em": sum(v["em"] for v in results.values()) / len(results)}
        print(f"[LongBench] {'AVERAGE':20s}: F1={results['AVERAGE']['f1']:.2f}")
    with open(output_path / "scores.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    p = argparse.ArgumentParser(description="CoMem LongBench eval")
    p.add_argument("--model_path", default="")
    p.add_argument("--resume_j", type=int, default=12)
    p.add_argument("--top_prepay_b", type=int, default=0)
    p.add_argument("--reuse_kv_blockdiag", action="store_true", default=False)
    p.add_argument("--lora_adapter", default="")
    p.add_argument("--baseline", default="none", choices=["none", "kvdirect", "hcache"])
    p.add_argument("--selector", default="bm25", choices=["bm25", "recency", "reader_attn"])
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--sink_tokens", default="bos", choices=["bos", "none"])
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--data_dir", default="data/longbench_raw/data")
    p.add_argument("--hf_dataset", default="THUDM/LongBench")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--output_dir", default="longbench_results/comem")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", default="sdpa")
    p.add_argument("--score_only", action="store_true")
    args = p.parse_args()

    datasets_list = args.tasks if args.tasks else DEFAULT_DATASETS
    if args.score_only:
        run_scoring(args.output_dir, datasets_list)
        return

    resume_j, no_retrieval, mode, lora = resolve_baseline(
        args.baseline, args.resume_j, args.lora_adapter)
    model, tok = load_backbone(args.model_path, args.dtype, args.attn_impl,
                               args.device, lora)
    cm = CoMem(model, resume_j=resume_j, top_prepay_b=args.top_prepay_b,
               block_diagonal=args.reuse_kv_blockdiag, tokenizer=tok)
    device = torch.device(args.device)
    all_data = load_longbench_dataset(args.hf_dataset, datasets_list, args.data_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"shard{args.shard_index}of{args.num_shards}" if sharded else "0"

    for ds_name in datasets_list:
        samples = all_data.get(ds_name, [])
        if not samples:
            continue
        if args.max_samples > 0:
            samples = samples[:args.max_samples]
        sample_indices = list(range(len(samples)))[args.shard_index::args.num_shards]
        max_gen = DATASET2MAXGEN.get(ds_name, 64)
        outfile = output_path / f"{ds_name}_{shard_tag}.jsonl"
        buf, t0 = [], time.time()
        for pos, idx in enumerate(tqdm(sample_indices, desc=ds_name, leave=True)):
            sample = samples[idx]
            prompt = format_prompt(sample, ds_name)
            ids = tok.encode(prompt, add_special_tokens=True, return_tensors="pt")
            if isinstance(ids, list):
                ids = torch.tensor([ids], dtype=torch.long)
            input_ids = ids.to(device)
            bare_q_ids = tok.encode((sample.get("input") or "").strip(),
                                    add_special_tokens=False)
            try:
                pred = cm.generate_from_ids(
                    input_ids, chunk_size=args.chunk_size, max_new_tokens=max_gen,
                    selector=args.selector, topk=args.topk,
                    sink_tokens=args.sink_tokens, bare_question_ids=bare_q_ids,
                    no_retrieval=no_retrieval)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                pred = "[OOM]"
                torch.cuda.empty_cache()
            buf.append({"index": idx, "pred": pred, "answers": sample["answers"],
                        "dataset": ds_name})
            if (pos + 1) % 10 == 0 or pos == len(sample_indices) - 1:
                with open(outfile, "w") as f:
                    for r in buf:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f1 = sum(compute_f1_multi(r["pred"], r["answers"]) for r in buf) / max(1, len(buf)) * 100
        print(f"[CoMem-LongBench] {ds_name}: F1={f1:.2f}% ({len(buf)}, {time.time()-t0:.1f}s)")
    if args.num_shards == 1:
        run_scoring(args.output_dir, datasets_list)


if __name__ == "__main__":
    main()
