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


# Baselines that bypass the CoMem read-pack entirely and run the stock HF model
# over the raw ids (so they must prefill the WHOLE prompt). Dispatched via
# ``dense_generate`` in the drivers.
DENSE_MODES = ("dense", "streamingllm")
# Prefill-then-compress KV-cache baselines (see comem.kvcompress): still a stock
# full prefill, but only a bounded slice of the KV is retained for decode.
KVCOMPRESS_MODES = ("snapkv", "pyramidkv")
FULL_MODEL_MODES = DENSE_MODES + KVCOMPRESS_MODES


def resolve_baseline(baseline, resume_j, lora_adapter):
    """Resolve a mechanism-level baseline to (resume_j, no_retrieval, mode, lora).

    * ``none``         -> CoMem: retrieval + given resume_j + LoRA.
    * ``kvdirect``     -> full-depth recompute (resume_j=0) + no retrieval + no LoRA.
    * ``hcache``       -> mid-layer recompute (given resume_j) + no retrieval + no LoRA.
    * ``dense``        -> stock full-context generation (no CoMem, no LoRA).
    * ``streamingllm`` -> attention-sink + sliding-window truncation, then dense.
    * ``snapkv`` / ``pyramidkv`` -> stock FULL prefill, then compress the stored KV
      to a bounded per-layer budget and decode from it (no CoMem, no LoRA).
    * ``cacheblend``   -> same retrieval / pack / sink as CoMem but the cached
      object is the FULL per-layer chunk KV instead of one depth-``j`` residual
      (no LoRA; ``resume_j`` is irrelevant, the path runs all layers).
    """
    if baseline == "kvdirect":
        return 0, True, "kvdirect", ""
    if baseline == "hcache":
        return resume_j, True, "hcache", ""
    if baseline == "cacheblend":
        return resume_j, False, "cacheblend", ""
    if baseline in FULL_MODEL_MODES:
        return resume_j, False, baseline, ""
    return resume_j, False, "comem", lora_adapter


def resolve_reader(cm, mode, recompute_ratio=0.15):
    """Return the object whose ``generate_from_ids`` the driver should call.

    ``cacheblend`` -> a :class:`comem.cacheblend.CacheBlend` wrapping ``cm``
    (signature-compatible with :meth:`comem.CoMem.generate_from_ids`, so the
    driver's call site does not change); anything else -> ``cm`` itself."""
    if mode != "cacheblend":
        return cm
    from comem.cacheblend import CacheBlend
    reader = CacheBlend(cm, recompute_ratio=recompute_ratio)
    print(f"[comem-eval] baseline cacheblend: full-depth chunk KV "
          f"({reader.kv_bytes_per_tok()} B/token) + selective recompute "
          f"r={recompute_ratio}")
    return reader


def install_baseline(model, mode, kv_budget=6657, kv_window=32):
    """Install a baseline that has to patch the backbone (``snapkv`` /
    ``pyramidkv``); a no-op for every other mode. Call ONCE after loading the
    model. Returns the resolved config dict (or ``None``)."""
    if mode not in KVCOMPRESS_MODES:
        return None
    from comem.kvcompress import install_kv_compression
    cfg = install_kv_compression(model, mode, max_capacity_prompt=kv_budget,
                                 window_size=kv_window)
    print(f"[comem-eval] baseline {mode}: retained KV budget {kv_budget}/layer "
          f"(window {kv_window}) — full prompt is still prefilled")
    return cfg


def resolve_dense_retriever(selector, retriever_path, device="cuda:0",
                            dtype="bfloat16"):
    """Build the frozen BGE retriever the ``dense_bge`` selector needs (``None``
    for every other selector). Fails closed when the path is missing, since
    silently falling back to another ranking would break the selector control."""
    if selector != "dense_bge":
        return None
    if not retriever_path:
        raise SystemExit(
            "--selector dense_bge needs --retriever_path <local BGE dir> "
            "(e.g. a BAAI/bge-large-en-v1.5 snapshot)")
    from comem.selectors import DenseBGERetriever
    print(f"[comem-eval] selector dense_bge: frozen retriever {retriever_path}")
    return DenseBGERetriever(retriever_path, device=device, dtype=DTYPES[dtype])


@torch.no_grad()
def dense_generate(model, tokenizer, input_ids, mode="dense", max_new_tokens=32,
                   sink_size=4, window_size=4096, stats=None):
    """Full-model generation baselines (no CoMem read-pack).

    * ``dense``        -> stock greedy generation over the entire prompt.
    * ``streamingllm`` -> keep the first ``sink_size`` tokens + the last
      ``window_size`` tokens (attention-sink + sliding-window truncation), then
      greedy generate. A standard, simple fixed-budget long-context baseline.
    * ``snapkv`` / ``pyramidkv`` -> full (exact) prefill of the whole prompt, then
      greedy decode over the compressed KV cache with TRUE logical positions
      (see :mod:`comem.kvcompress`; needs :func:`install_baseline` first).
      ``stats`` receives the retained-KV audit.

    Returns the decoded continuation (or ``"[OOM]"`` on CUDA OOM).
    """
    ids = input_ids
    if mode == "streamingllm" and ids.shape[1] > sink_size + window_size:
        ids = torch.cat([ids[:, :sink_size], ids[:, -window_size:]], dim=1)
    try:
        if mode in KVCOMPRESS_MODES:
            from comem.kvcompress import generate_kvcompress
            gen = generate_kvcompress(model, ids, max_new_tokens=max_new_tokens,
                                      eos_token_id=tokenizer.eos_token_id,
                                      stats=stats)
            return tokenizer.decode(gen, skip_special_tokens=True).strip()
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=(tokenizer.pad_token_id
                          if tokenizer.pad_token_id is not None else 0),
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        return "[OOM]"
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


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
