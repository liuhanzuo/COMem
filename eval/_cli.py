"""Unified CLI surface shared by every CoMem eval driver.

Each ``eval/<bench>.py`` keeps its native flags but also exposes the
collaborator-facing aliases so one habit works everywhere::

    --model <hf_path>   (alias of --model_path)
    --j <int|auto>      (alias of --resume_j; ``auto`` -> comem.model_registry)
    --n <int>           (alias of the driver's sample-count flag)
    --adapter <path>    (alias of --lora_adapter; ``none`` -> disabled)
    --out <dir>         (alias of the driver's output-dir flag)
    --lengths 8k,16k    (comma OR space separated)
    --selector / --baseline / --topk / --chunk_size / ...

``--j auto`` picks the per-backbone split depth from
:mod:`comem.model_registry`.

Baseline / selector vocabulary
------------------------------
``BASELINE_CHOICES`` / ``SELECTOR_CHOICES`` are shared by every driver, and
:func:`add_baseline_args` adds the knobs the external baselines need
(``--recompute_ratio`` for ``cacheblend``, ``--kv_budget`` / ``--kv_window`` for
``snapkv`` / ``pyramidkv``, ``--retriever_path`` for the ``dense_bge`` selector),
so they cannot drift between drivers.
"""
from __future__ import annotations

from comem.model_registry import resolve_resume_j

# every driver advertises the same baseline vocabulary:
#   none/kvdirect/hcache -> CoMem re-parameterisations,
#   dense/streamingllm   -> stock full-context arms,
#   snapkv/pyramidkv     -> prefill-then-compress KV baselines (comem.kvcompress),
#   cacheblend           -> full-depth chunk-KV baseline (comem.cacheblend).
BASELINE_CHOICES = ["none", "dense", "kvdirect", "hcache", "streamingllm",
                    "snapkv", "pyramidkv", "cacheblend"]

# shared selector vocabulary (``dense_bge`` additionally needs --retriever_path);
# RULER's ``auto`` per-task routing stays local to that driver.
SELECTOR_CHOICES = ["bm25", "recency", "oracle", "reader_attn", "iter_reader_attn",
                    "iter_bm25", "iter_bm25_adaptive", "dense_bge"]


def add_baseline_args(p):
    """Add the shared external-baseline / dense-selector knobs to a driver's parser."""
    p.add_argument("--recompute_ratio", type=float, default=0.15,
                   help="cacheblend: fraction of CONTEXT tokens whose KV is "
                        "recomputed after the blend (r=0 pure reuse, r=1 == full "
                        "prefill). Sink and query are always recomputed.")
    p.add_argument("--kv_budget", type=int, default=6657,
                   help="snapkv/pyramidkv: retained KV tokens per layer including "
                        "the observation window (default 6657 == CoMem's read "
                        "pack: BOS 1 + top-12 x 512 + query <= 512).")
    p.add_argument("--kv_window", type=int, default=32,
                   help="snapkv/pyramidkv: observation-window size (the recent "
                        "tokens always kept, whose queries score the past).")
    p.add_argument("--retriever_path", default="",
                   help="--selector dense_bge: local frozen BGE dir (e.g. a "
                        "BAAI/bge-large-en-v1.5 snapshot).")
    return p


def j_type(value):
    """argparse type for ``--j`` / ``--resume_j``: an int, or the string 'auto'."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return int(value)


def split_lengths(items):
    """Flatten a mix of comma- and space-separated tokens into a clean list.

    ``["8k,16k", "32k"]`` and ``["8k", "16k", "32k"]`` both -> ``[8k, 16k, 32k]``.
    """
    out = []
    for item in items or []:
        for piece in str(item).split(","):
            piece = piece.strip()
            if piece:
                out.append(piece)
    return out


def resolve_j(resume_j, model_path):
    """Turn a parsed ``--j`` value into a concrete int (``auto`` -> registry)."""
    if resume_j == "auto":
        return resolve_resume_j(model_path)
    return int(resume_j)


def normalize_args(args, task_attrs=()):
    """In-place post-parse normalization common to every driver:

    * split comma/space length + task lists,
    * ``--adapter none`` (or empty) -> ``""`` (disabled),
    * ``--j auto`` -> concrete int via the model registry.
    """
    if getattr(args, "lengths", None):
        args.lengths = split_lengths(args.lengths)
    for attr in task_attrs:
        if getattr(args, attr, None):
            setattr(args, attr, split_lengths(getattr(args, attr)))
    if isinstance(getattr(args, "lora_adapter", None), str):
        if args.lora_adapter.strip().lower() in ("none", ""):
            args.lora_adapter = ""
    if hasattr(args, "resume_j"):
        args.resume_j = resolve_j(args.resume_j, args.model_path)
    return args
