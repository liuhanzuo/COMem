#!/usr/bin/env python
"""CoMem — LoCoMo (long-conversation memory) eval driver.

Runs CoMem on LoCoMo (ACL 2024): 10 extended two-speaker conversations with QA
across 5 categories (multi-hop / single-hop / temporal / open-domain /
adversarial). Thin: build CoMem, ``generate_from_ids`` per QA, score with
SQuAD-style F1 / EM / substring-acc (category-5 = abstention-correct).
Self-contained: LoCoMo parsing + scoring embedded (ported verbatim).

Usage:
    python -m eval.locomo --model_path /path/to/Qwen3-8B --resume_j 12 \\
        --selector bm25 --topk 12 --locomo_data data/locomo10.json \\
        --output_dir locomo_results/comem_j12
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
from comem import selectors as _sel              # noqa: E402
from eval._common import load_backbone, resolve_baseline  # noqa: E402

CATEGORY_NAMES = {1: "multi_hop", 2: "single_hop", 3: "temporal",
                  4: "open_domain", 5: "adversarial"}
_LOCOMO_INSTRUCTION = (
    "You are a helpful assistant with memory of a long conversation between "
    "{spa} and {spb}, organized into dated sessions. Read the conversation "
    "history, then answer the question using only the information in the "
    "history. Answer as concisely as possible with a short phrase, date, or "
    "number. Do not explain."
)
_REFUSAL_RE = re.compile(
    r"\b(i don'?t know|not (mentioned|sure|provided|available|specified)|"
    r"no (information|mention|record)|cannot (find|determine|answer)|"
    r"unanswerable|isn'?t (mentioned|provided)|wasn'?t mentioned)\b", re.IGNORECASE)


def render_locomo_history(conv):
    parts, i = [], 1
    while f"session_{i}" in conv:
        date = conv.get(f"session_{i}_date_time", "")
        parts.append(f"\n=== Session {i}{(' (' + date + ')') if date else ''} ===")
        for turn in conv[f"session_{i}"]:
            parts.append(f"{turn.get('speaker', '')}: {turn.get('text', '')}")
        i += 1
    return "\n".join(parts)


def _build_dia_id_map(conv):
    dia_map, i = {}, 1
    while f"session_{i}" in conv:
        for turn in conv[f"session_{i}"]:
            did = turn.get("dia_id", "")
            txt = (turn.get("text", "") or "").strip()
            if did and txt:
                dia_map[did] = txt
        i += 1
    return dia_map


def _resolve_evidence_texts(evidence, dia_map):
    texts = []
    if not isinstance(evidence, (list, tuple)):
        return texts
    for eid_raw in evidence:
        if not isinstance(eid_raw, str):
            continue
        for eid in re.split(r"[;,]\s*", eid_raw):
            eid = eid.strip()
            if eid in dia_map and dia_map[eid] not in texts:
                texts.append(dia_map[eid])
    return texts


def build_locomo_samples(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    samples = []
    for conv_idx, d in enumerate(data):
        conv = d.get("conversation", {})
        if not isinstance(conv, dict):
            continue
        spa = conv.get("speaker_a", "Speaker A")
        spb = conv.get("speaker_b", "Speaker B")
        instr = _LOCOMO_INSTRUCTION.format(spa=spa, spb=spb)
        history = render_locomo_history(conv)
        dia_map = _build_dia_id_map(conv)
        for qi, qa in enumerate(d.get("qa", [])):
            question = (qa.get("question", "") or "").strip()
            if not question:
                continue
            category = qa.get("category", -1)
            ans = qa.get("answer", None)
            if ans is None:
                ans = qa.get("adversarial_answer", "")
            ans = ans if isinstance(ans, str) else str(ans)
            prompt = (f"{instr}\n\n# Conversation history\n{history}\n\n"
                      f"# Question\n{question}\n\n# Answer\n")
            samples.append({
                "id": f"conv{conv_idx}_qa{qi}", "prompt": prompt,
                "question": question, "answers": [ans], "category": category,
                "is_abstention": (category == 5),
                "evidence_texts": _resolve_evidence_texts(qa.get("evidence", []), dia_map),
            })
    return samples


def _oracle_needle_chunks(input_ids, sample, tokenizer, chunk_size):
    probes = list(sample.get("evidence_texts") or [])
    ans = sample["answers"][0] if sample.get("answers") else ""
    if ans:
        probes.append(ans)
    chunks = set()
    for probe in probes:
        probe = (probe or "").strip()
        if not probe:
            continue
        got = _sel.locate_needle_chunks(input_ids, probe, tokenizer, chunk_size)
        if got:
            chunks |= got
    return chunks or None


def normalize_answer(s):
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(remove_articles("".join(
        ch for ch in (s or "").lower() if ch not in set(string.punctuation))).split())


def compute_f1(pred, gt):
    pt, gt_t = normalize_answer(pred).split(), normalize_answer(gt).split()
    if len(pt) == 0 and len(gt_t) == 0:
        return 1.0
    if len(pt) == 0 or len(gt_t) == 0:
        return 0.0
    num_same = sum((collections.Counter(pt) & collections.Counter(gt_t)).values())
    if num_same == 0:
        return 0.0
    p, r = num_same / len(pt), num_same / len(gt_t)
    return 2 * p * r / (p + r)


def compute_f1_multi(pred, answers):
    return max((compute_f1(pred, a) for a in answers), default=0.0)


def compute_em_multi(pred, answers):
    np_ = normalize_answer(pred)
    return max((1.0 if np_ == normalize_answer(a) else 0.0 for a in answers), default=0.0)


def substring_acc(pred, answers):
    np_ = normalize_answer(pred)
    for a in answers:
        na = normalize_answer(a)
        if na and (na in np_ or np_ in na):
            return 1.0
    return 0.0


def score_sample(item):
    pred = item.get("pred", "")
    answers = item.get("answers", [])
    refused = bool(_REFUSAL_RE.search(pred)) or pred.strip() == ""
    if item.get("is_abstention", False):
        acc = 1.0 if refused else 0.0
        return {"f1": acc, "em": acc, "acc": acc, "refused": refused}
    f1 = compute_f1_multi(pred, answers)
    em = compute_em_multi(pred, answers)
    acc = max(substring_acc(pred, answers), 1.0 if f1 >= 0.5 else 0.0)
    return {"f1": f1, "em": em, "acc": acc, "refused": refused}


def run_scoring(output_dir):
    output_path = Path(output_dir)
    shard_files = sorted(output_path.glob("preds*.jsonl"))
    if not shard_files:
        print(f"[CoMem-LoCoMo] no prediction files in {output_dir}")
        return None
    preds, seen = [], set()
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item["id"] not in seen:
                    seen.add(item["id"])
                    preds.append(item)
    if not preds:
        return None
    overall = collections.defaultdict(list)
    by_cat = collections.defaultdict(lambda: collections.defaultdict(list))
    for item in preds:
        sc = score_sample(item)
        cat = str(item.get("category", "?"))
        for k in ("f1", "em", "acc"):
            overall[k].append(sc[k])
            by_cat[cat][k].append(sc[k])
    n = len(preds)

    def _avg(xs):
        return (sum(xs) / len(xs) * 100) if xs else 0.0

    results = {"benchmark": "locomo", "n_samples": n,
               "overall_f1": _avg(overall["f1"]), "overall_em": _avg(overall["em"]),
               "overall_acc": _avg(overall["acc"]), "by_category": {}}
    print(f"\n[CoMem-LoCoMo] locomo  n={n}")
    print(f"  OVERALL   F1={results['overall_f1']:6.2f}  "
          f"EM={results['overall_em']:6.2f}  acc={results['overall_acc']:6.2f}")
    for cat in sorted(by_cat, key=lambda c: (c == "?", c)):
        v = by_cat[cat]
        name = CATEGORY_NAMES.get(int(cat), cat) if cat.lstrip("-").isdigit() else cat
        entry = {"n": len(v["f1"]), "f1": _avg(v["f1"]), "em": _avg(v["em"]),
                 "acc": _avg(v["acc"])}
        results["by_category"][cat] = entry
        print(f"  cat{cat:>2} {name:12s} F1={entry['f1']:6.2f}  "
              f"EM={entry['em']:6.2f}  acc={entry['acc']:6.2f}  (n={entry['n']})")
    with open(output_path / "scores.json", "w") as fh:
        json.dump(results, fh, indent=2)
    return results


def main():
    p = argparse.ArgumentParser(description="CoMem LoCoMo eval")
    p.add_argument("--model_path", default="")
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
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--locomo_data", default="data/locomo10.json")
    p.add_argument("--categories", default=None)
    p.add_argument("--output_dir", default="locomo_results/comem")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", default="sdpa")
    p.add_argument("--score_only", action="store_true")
    args = p.parse_args()

    if args.score_only:
        run_scoring(args.output_dir)
        return

    resume_j, no_retrieval, mode, lora = resolve_baseline(
        args.baseline, args.resume_j, args.lora_adapter)
    data_path = args.locomo_data
    model, tok = load_backbone(args.model_path, args.dtype, args.attn_impl,
                               args.device, lora)
    cm = CoMem(model, resume_j=resume_j, top_prepay_b=args.top_prepay_b,
               block_diagonal=args.reuse_kv_blockdiag, tokenizer=tok)
    device = torch.device(args.device)

    categories = None
    if args.categories:
        categories = {int(c.strip()) for c in args.categories.split(",") if c.strip()}
    samples = build_locomo_samples(data_path)
    if categories is not None:
        samples = [s for s in samples if s["category"] in categories]
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    shard = samples[args.shard_index::args.num_shards]
    print(f"[CoMem-LoCoMo] shard {args.shard_index}/{args.num_shards}: {len(shard)} samples")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""
    outfile = outdir / f"preds{shard_tag}.jsonl"
    buf, t0 = [], time.time()
    for pos, sample in enumerate(tqdm(shard, desc="locomo", leave=True)):
        ids = tok.encode(sample["prompt"], add_special_tokens=True, return_tensors="pt")
        if isinstance(ids, list):
            ids = torch.tensor([ids], dtype=torch.long)
        input_ids = ids.to(device)
        bare_q_ids = tok.encode(sample["question"], add_special_tokens=False)
        needle_set = None
        if not no_retrieval and args.selector == "oracle":
            needle_set = _oracle_needle_chunks(input_ids, sample, tok, args.chunk_size)
        try:
            pred = cm.generate_from_ids(
                input_ids, chunk_size=args.chunk_size,
                max_new_tokens=args.max_new_tokens, selector=args.selector,
                topk=args.topk, sink_tokens=args.sink_tokens,
                needle_chunk_set=needle_set, bare_question_ids=bare_q_ids,
                no_retrieval=no_retrieval)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            pred = "[OOM]"
            torch.cuda.empty_cache()
        buf.append({"id": sample["id"], "pred": pred, "answers": sample["answers"],
                    "category": sample["category"], "is_abstention": sample["is_abstention"],
                    "question": sample["question"]})
        if (pos + 1) % 10 == 0 or pos == len(shard) - 1:
            with open(outfile, "w") as f:
                for r in buf:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[CoMem-LoCoMo] shard done: {len(buf)} samples ({time.time()-t0:.1f}s)")
    if args.num_shards == 1:
        run_scoring(args.output_dir)


if __name__ == "__main__":
    main()
