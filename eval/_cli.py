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
"""
from __future__ import annotations

from comem.model_registry import resolve_resume_j

# every driver advertises the same baseline vocabulary
BASELINE_CHOICES = ["none", "dense", "kvdirect", "hcache", "streamingllm"]


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
