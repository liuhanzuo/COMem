"""Shared helpers for the CoMem eval drivers (data loading / scoring live in the
per-benchmark modules; this holds only model loading + a CSV writer).

The eval drivers import ONLY ``comem`` + their benchmark's data/scoring code — no
dependency on the research repo. Each builds a :class:`comem.CoMem`, runs
``generate_from_ids`` (the fused encode+write+select+decode reference path over a
single prompt whose trailing chunk is the query), and applies the official metric.
"""
from __future__ import annotations

import csv
import os

import torch

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}


def load_backbone(model_path, dtype="bfloat16", attn_impl="sdpa", device="cuda:0",
                  lora_adapter=""):
    """Load a stock ``*ForCausalLM`` + tokenizer (offline), optionally applying a
    trained CoMem-distill LoRA adapter. Returns ``(model, tokenizer)``. The LoRA
    delta is applied by the peft-wrapped Linear submodules when CoMem calls the
    layers directly, so we hand CoMem the underlying ``base_model.model``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=DTYPES[dtype], attn_implementation=attn_impl,
        trust_remote_code=True, local_files_only=True,
    ).to(device).eval()
    if lora_adapter:
        from peft import PeftModel
        print(f"[comem-eval] loading LoRA adapter: {lora_adapter}")
        peft_model = PeftModel.from_pretrained(model, lora_adapter).eval()
        model = peft_model.base_model.model
    return model, tok


def resolve_baseline(baseline, resume_j, lora_adapter):
    """Resolve a mechanism-level baseline to (resume_j, no_retrieval, mode, lora).

    * ``none``     -> CoMem: retrieval + given resume_j + LoRA.
    * ``kvdirect`` -> full-depth recompute (resume_j=0) + no retrieval + no LoRA.
    * ``hcache``   -> mid-layer recompute (given resume_j) + no retrieval + no LoRA.
    """
    if baseline == "kvdirect":
        return 0, True, "kvdirect", ""
    if baseline == "hcache":
        return resume_j, True, "hcache", ""
    return resume_j, False, "comem", lora_adapter


def sanitize_output(text):
    """Flatten embedded newlines so one physical CSV line == one record."""
    if not isinstance(text, str):
        return text
    return text.replace("\r", " ").replace("\n", " ")


def write_results_csv(df, outfile):
    """Write a (target, output, question, ...) frame with newlines flattened and
    every field quoted (QUOTE_ALL) — matches the BABILong nested-CSV layout."""
    safe = df.copy()
    if "output" in safe.columns:
        safe["output"] = safe["output"].map(sanitize_output)
    safe.to_csv(outfile, index=False, quoting=csv.QUOTE_ALL)
