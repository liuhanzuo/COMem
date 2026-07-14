#!/usr/bin/env python
"""CoMem — BABILong (synthetic long-context recall) eval driver.

Runs CoMem on BABILong (qa1..qa10 x lengths). Thin: build CoMem,
``generate_from_ids`` per sample, write the nested CSV layout the official
``babilong.metrics`` scorer consumes. Self-contained apart from the ``babilong``
package (prompts + metric) which is the benchmark's own code — install it (pip
install babilong, or add the babilong repo to PYTHONPATH).

The offline Arrow-cache dataset loader is embedded here (ported verbatim) so no
network is needed once the dataset is cached under ``~/.cache/huggingface`` or
``$HF_DATASETS_CACHE``.

Usage:
    python -m eval.babilong --model_path /path/to/Llama-3-8B --resume_j 6 \\
        --selector bm25 --topk 4 --tasks qa1 qa2 qa5 --lengths 0k 1k 2k 4k 8k 16k \\
        --output_name comem_j6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comem import CoMem                          # noqa: E402
from comem import selectors as _sel              # noqa: E402
from eval._common import load_backbone, resolve_baseline, write_results_csv  # noqa: E402

from babilong.prompts import (  # noqa: E402
    DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input)


# --------------------------------------------------------------------------- #
# offline Arrow-cache dataset loader (ported verbatim)
# --------------------------------------------------------------------------- #
def _candidate_cache_dirs(user_cache_dir):
    roots = []
    if user_cache_dir:
        roots.append(Path(user_cache_dir).expanduser())
    for env in ("HF_DATASETS_CACHE", "HF_HOME"):
        if os.environ.get(env):
            root = Path(os.environ[env]).expanduser()
            roots.append(root if env == "HF_DATASETS_CACHE" else root / "datasets")
    roots += [Path.cwd() / ".cache/huggingface/datasets",
              Path.home() / ".cache/huggingface/datasets"]
    seen, out = set(), []
    for root in roots:
        key = str(root.absolute())
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def _load_from_arrow_cache(dataset_name, split_name, cache_dir):
    import datasets
    root = cache_dir / dataset_name.replace("/", "___") / split_name
    arrow_roots = [p for p in root.glob("*/*")
                   if p.is_dir() and any(p.glob("babilong-*.arrow"))]
    if not arrow_roots:
        return None
    arrow_root = max(arrow_roots, key=lambda p: p.stat().st_mtime)
    data = {p.stem.removeprefix("babilong-"): datasets.Dataset.from_file(str(p))
            for p in sorted(arrow_root.glob("babilong-*.arrow"))}
    return data or None


def load_babilong_dataset(dataset_name, split_name, cache_dir=None):
    import datasets
    last_error = None
    for candidate in _candidate_cache_dirs(cache_dir):
        try:
            data = datasets.load_dataset(dataset_name, split_name,
                                         cache_dir=str(candidate),
                                         download_mode="reuse_dataset_if_exists")
            return data
        except Exception as e:
            last_error = e
            data = _load_from_arrow_cache(dataset_name, split_name, candidate)
            if data is not None:
                return data
    try:
        return datasets.load_dataset(dataset_name, split_name,
                                     download_mode="reuse_dataset_if_exists")
    except Exception:
        raise last_error


def main():
    p = argparse.ArgumentParser(description="CoMem BABILong eval")
    p.add_argument("--model_path", required=True)
    p.add_argument("--resume_j", type=int, default=6)
    p.add_argument("--top_prepay_b", type=int, default=0)
    p.add_argument("--reuse_kv_blockdiag", action="store_true", default=False)
    p.add_argument("--lora_adapter", default="")
    p.add_argument("--baseline", default="none", choices=["none", "kvdirect", "hcache"])
    p.add_argument("--selector", default="bm25",
                   choices=["bm25", "recency", "oracle", "reader_attn",
                            "iter_reader_attn", "iter_bm25"])
    p.add_argument("--iter_rounds", type=int, default=0)
    p.add_argument("--iter_hop_topk", type=int, default=2)
    p.add_argument("--iter_score", default="meanpool", choices=["meanpool", "maxsim"])
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--sink_tokens", default="bos", choices=["bos", "none"])
    p.add_argument("--results_folder", default="./babilong_results")
    p.add_argument("--output_name", required=True)
    p.add_argument("--dataset_name", default="RMT-team/babilong")
    p.add_argument("--tasks", nargs="+", default=["qa1", "qa2", "qa5"])
    p.add_argument("--lengths", nargs="+", default=["0k", "1k", "2k", "4k", "8k", "16k"])
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
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
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    for task in tqdm(args.tasks, desc="tasks"):
        if task not in DEFAULT_PROMPTS:
            continue
        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"],
            "examples": DEFAULT_PROMPTS[task]["examples"],
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"],
            "template": DEFAULT_TEMPLATE, "chat_template": False, "system_prompt": "",
        }
        prompt_name = "_".join(f"{k}_yes" if prompt_cfg[k] else f"{k}_no"
                               for k in prompt_cfg if k != "template")
        for split_name in tqdm(args.lengths, desc="lengths", leave=False):
            try:
                data = load_babilong_dataset(args.dataset_name, split_name)
                task_data = data[task]
            except Exception as e:
                print(f"[ERROR] load {args.dataset_name}/{split_name}/{task}: {e}")
                continue
            outdir = Path(args.results_folder) / args.output_name
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{task}_{split_name}_{prompt_name}{shard_tag}.csv"
            df = pd.DataFrame({"target": [], "output": [], "question": []})
            num_samples = len(task_data)
            if args.limit > 0:
                num_samples = min(num_samples, args.limit)
            sample_indices = list(range(num_samples))[args.shard_index::args.num_shards]
            for idx in tqdm(sample_indices, desc=f"{task}/{split_name}", leave=False):
                sample = task_data[idx]
                target, question = sample["target"], sample["question"]
                input_text = get_formatted_input(
                    sample["input"], sample["question"], prompt_cfg["examples"],
                    prompt_cfg["instruction"], prompt_cfg["post_prompt"],
                    template=prompt_cfg["template"])
                ids = tok.encode(input_text, add_special_tokens=True, return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                input_ids = ids.to(device)
                needle_set = None
                if args.selector == "oracle":
                    needle_set = _sel.locate_needle_chunks(
                        input_ids, target, tok, args.chunk_size)
                bare_q_ids = tok.encode((question or "").strip(), add_special_tokens=False)
                try:
                    out = cm.generate_from_ids(
                        input_ids, chunk_size=args.chunk_size,
                        max_new_tokens=args.max_new_tokens, selector=args.selector,
                        topk=args.topk, sink_tokens=args.sink_tokens,
                        needle_chunk_set=needle_set, bare_question_ids=bare_q_ids,
                        no_retrieval=no_retrieval, iter_rounds=args.iter_rounds,
                        iter_hop_topk=args.iter_hop_topk, iter_score=args.iter_score)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    out = "[OOM]"
                    torch.cuda.empty_cache()
                df.loc[len(df)] = [target, out, question]
                if len(df) % 10 == 0:
                    write_results_csv(df, outfile)
            write_results_csv(df, outfile)
            print(f"[CoMem-BABILong] saved {len(df)} -> {outfile}")
    print("[CoMem-BABILong] done. Score with babilong.metrics.compare_answers.")


if __name__ == "__main__":
    main()
