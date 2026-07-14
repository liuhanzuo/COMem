"""Per-model CoMem split depth (``resume_j`` ≈ 0.33 · num_hidden_layers).

The collaborator-facing ``--j auto`` flag resolves through here so a given
backbone always uses the same, reproducible split depth. Keys are matched as
case-insensitive substrings of the model path / name (longest key wins), so
``/path/to/Qwen3-8B`` and ``Qwen/Qwen3-8B`` both resolve to ``j = 12``.

Table (``resume_j`` ≈ round(0.33 · L)):

    Qwen3-0.6B  (L=28) -> 9      Qwen3-14B    (L=40) -> 13
    Qwen3-1.7B  (L=28) -> 9      Qwen3-32B    (L=64) -> 21
    Qwen3-4B    (L=36) -> 12     Qwen3-30B-A3B(L=48) -> 16
    Qwen3-8B    (L=36) -> 12

Unknown backbones fall back to ``round(0.33 · num_hidden_layers)`` (read from the
HF config) with a printed warning.
"""
from __future__ import annotations

DEFAULT_RATIO = 0.33

# key substring (lower-cased, '_' -> '-') -> (num_hidden_layers, resume_j)
MODEL_J_TABLE = {
    "qwen3-0.6b":    (28, 9),
    "qwen3-1.7b":    (28, 9),
    "qwen3-4b":      (36, 12),
    "qwen3-8b":      (36, 12),
    "qwen3-14b":     (40, 13),
    "qwen3-32b":     (64, 21),
    "qwen3-30b-a3b": (48, 16),
}


def _normalize(name: str) -> str:
    return str(name).lower().replace("_", "-")


def lookup(model_name: str):
    """Return ``(num_layers, resume_j)`` for the first (longest) registry key that
    is a substring of ``model_name``, else ``None``."""
    name = _normalize(model_name)
    for key in sorted(MODEL_J_TABLE, key=len, reverse=True):
        if key in name:
            return MODEL_J_TABLE[key]
    return None


def _num_hidden_layers(model_path: str) -> int:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True,
                                     local_files_only=True)
    return int(getattr(cfg, "num_hidden_layers"))


def resolve_resume_j(model_path: str, num_layers=None, verbose: bool = True) -> int:
    """Resolve the split depth for ``model_path``.

    1. registry substring match -> its ``resume_j``;
    2. otherwise ``round(DEFAULT_RATIO * num_layers)`` (num_layers from the HF
       config if not supplied), with a printed warning.
    """
    hit = lookup(model_path)
    if hit is not None:
        return hit[1]
    if num_layers is None:
        num_layers = _num_hidden_layers(model_path)
    j = int(round(DEFAULT_RATIO * num_layers))
    if verbose:
        print(f"[comem] --j auto: no registry match for {model_path!r}; "
              f"falling back to round({DEFAULT_RATIO} * {num_layers}) = {j}")
    return j
