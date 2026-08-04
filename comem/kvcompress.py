"""Prefill-then-compress KV-cache baselines for CoMem: **SnapKV** and **PyramidKV**.

These are *external comparison points*, not part of the CoMem mechanism: they
keep the stock backbone but evict most of the prefill KV cache, so the model
still has to **prefill the whole prompt** once (cost grows with the context) and
only the retained KV is bounded. CoMem's contrast is the opposite trade: a
bounded read over a persistent depth-``j`` store, with no full-prompt prefill.

Methods
-------
* **SnapKV** (Li et al., NeurIPS'24, arXiv:2404.14469) — score every past key by
  the attention it receives from the last ``window_size`` "observation window"
  queries, 1-D pool the scores (kernel ``kernel_size``, avg/max), keep the
  ``max_capacity_prompt - window_size`` highest-scoring past positions plus the
  whole observation window. Uniform budget on every layer.
* **PyramidKV** (Cai et al., arXiv:2406.02069) — the same observation-window
  scoring with a **pyramidal per-layer budget**: lower layers keep more, upper
  layers fewer, arithmetically interpolated between
  ``max_num = 2(C-w) - min_num`` and ``min_num = (C-w)/beta`` so the average
  per-layer budget is ``max_capacity_prompt``.

Implementation (clean-room, from the published specification)
-------------------------------------------------------------
The two upstream reference repos (FasterDecoding/SnapKV, Zefan-Cai/PyramidKV)
monkeypatch **transformers 4.37 + Llama/Mistral** attention: they target the
pre-``Cache``-refactor ``past_key_value.update`` contract and
``rotary_emb(..., seq_len=)``, which no longer exist. This module therefore
re-implements the *selection rule* from the papers' specification (formulae
above; GQA-aware, see below) against the modern
``ALL_ATTENTION_FUNCTIONS`` interface, and installs itself by registering a
custom attention implementation instead of patching a model class.

The selection math (observation-window softmax scores in fp32 -> 1-D pooling ->
top-k over the past -> keep the recent window, and PyramidKV's per-layer budget
schedule) was cross-checked against the upstream ``SnapKVCluster`` /
``PyramidKVCluster`` on random GQA tensors and reproduces their retained K/V
**bit-for-bit** (max|diff| = 0 on every layer, both methods), so the
re-implementation is functionally equivalent to the reference algorithm despite
being written against a different attention API.

Two contract details worth spelling out:

* **The full prefill is EXACT.** On the prefill call the attention output is
  computed over the FULL, uncompressed K/V — compression never perturbs the
  prefill logits, it only decides what is *stored* for the decode phase. This is
  the published behaviour and it makes the "no compression fires when the prompt
  is shorter than the budget => bit-identical to stock attention" self-test a
  meaningful faithfulness gate.
* **Decode must use TRUE logical positions.** Retained keys keep the RoPE of
  their true prefill positions, so the decode query has to be RoPE'd at the true
  next logical position, not at the (shorter) compressed cache length. HF's
  ``generate`` derives the position from the cache length, which is wrong here,
  so :func:`generate_kvcompress` runs an explicit greedy loop passing
  ``position_ids`` — the modern equivalent of the reference repos'
  ``prepare_inputs_for_generation`` override that tracked ``kv_seq_len``
  separately from the compressed cache length.

* **GQA.** Qwen3 stores unrepeated K/V (8 kv-heads vs 32 query heads). Scores are
  computed with transiently repeated keys at query-head granularity, then
  group-reduced (``mean`` by default) to kv-head granularity, and the top-k
  gather runs on the unrepeated tensors so the write-back matches the cache
  layout exactly.

Usage
-----
    from comem.kvcompress import install_kv_compression, generate_kvcompress
    cfg = install_kv_compression(model, "snapkv", max_capacity_prompt=6657)
    ids_out = generate_kvcompress(model, input_ids, max_new_tokens=32)
    uninstall_kv_compression(model)

``python -m comem.kvcompress`` runs the CPU faithfulness self-test.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

METHODS = ("snapkv", "pyramidkv")

# Retained-token budget used by the paper's equal-retained-budget diagnostic:
# CoMem's read pack == BOS 1 + top-12 x 512 + query <= 512 = 6657 tokens.
COMEM_READ_BUDGET = 6657

# Name of the attention implementation this module registers. Installing sets
# ``config._attn_implementation`` to it; uninstalling restores the previous value.
_ATTN_IMPL = "comem_kvcompress"


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[B, n_kv, T, d] -> [B, n_kv*n_rep, T, d]`` (transformers ``repeat_kv``)."""
    if n_rep == 1:
        return x
    B, n_kv, T, d = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, T, d).reshape(B, n_kv * n_rep, T, d)


# --------------------------------------------------------------------------- #
# the selection rule (SnapKV / PyramidKV observation-window scoring)
# --------------------------------------------------------------------------- #
def observation_window_scores(query_states, key_states, *, window_size: int,
                              kernel_size: int, pooling: str,
                              gqa_score_agg: str) -> torch.Tensor:
    """Per-past-position importance at KV-head granularity (SnapKV eq. 1-2).

    Attention from the last ``window_size`` queries to every key (causal inside
    the window), fp32 softmax, summed over the window's query axis, group-reduced
    to kv-head granularity, then 1-D pooled with ``kernel_size``. Returns
    ``[B, n_kv, T - window_size]`` scores over the PAST positions only.
    """
    B, n_q, T, head_dim = query_states.shape
    n_kv = key_states.shape[1]
    groups = n_q // n_kv
    key_rep = _repeat_kv(key_states, groups)
    attn = torch.matmul(query_states[..., -window_size:, :],
                        key_rep.transpose(2, 3)) / math.sqrt(head_dim)
    # causal inside the observation window (the window's own lower-right block)
    neg = torch.finfo(attn.dtype).min
    idx = torch.arange(window_size, device=attn.device)
    win_mask = torch.where(idx.view(1, -1) > idx.view(-1, 1),
                           torch.full((), neg, dtype=attn.dtype, device=attn.device),
                           torch.zeros((), dtype=attn.dtype, device=attn.device))
    attn[:, :, -window_size:, -window_size:] += win_mask[None, None]
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
    scores = attn[:, :, -window_size:, :-window_size].sum(dim=-2)   # [B, n_q, T-w]
    if groups > 1:
        grouped = scores.reshape(B, n_kv, groups, scores.shape[-1])
        if gqa_score_agg == "mean":
            scores = grouped.mean(dim=2)
        elif gqa_score_agg == "max":
            scores = grouped.amax(dim=2)
        elif gqa_score_agg == "sum":
            scores = grouped.sum(dim=2)
        else:
            raise ValueError(f"unknown gqa_score_agg {gqa_score_agg!r}; expected "
                             "'mean', 'max' or 'sum'")
    if pooling == "avgpool":
        return F.avg_pool1d(scores, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
    if pooling == "maxpool":
        return F.max_pool1d(scores, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
    raise ValueError(f"unknown pooling {pooling!r}; expected 'avgpool' or 'maxpool'")


def _gather_topk_kv(key_states, value_states, scores, capacity: int, window_size: int):
    """Keep the ``capacity`` top-scoring PAST positions + the recent window."""
    head_dim = key_states.shape[-1]
    idx = scores.topk(capacity, dim=-1).indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_past = key_states[:, :, :-window_size, :].gather(dim=2, index=idx)
    v_past = value_states[:, :, :-window_size, :].gather(dim=2, index=idx)
    return (torch.cat([k_past, key_states[:, :, -window_size:, :]], dim=2),
            torch.cat([v_past, value_states[:, :, -window_size:, :]], dim=2))


class KVCompressCluster:
    """SnapKV / PyramidKV prefill-KV selection for ONE attention layer.

    ``max_capacity_prompt`` is the TOTAL retained tokens per layer INCLUDING the
    ``window_size`` recent observation window (SnapKV: uniform on every layer;
    PyramidKV: the per-layer *average* under its pyramidal schedule). Prompts
    shorter than the budget are returned unchanged (no compression fires), which
    is what makes the stock-equivalence self-test meaningful.
    """

    def __init__(self, method: str, *, num_hidden_layers: int, layer_idx: int,
                 max_capacity_prompt: int = COMEM_READ_BUDGET, window_size: int = 32,
                 kernel_size: int = 5, pooling: str = "avgpool", beta: int = 20,
                 gqa_score_agg: str = "mean"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}; got {method!r}")
        if max_capacity_prompt - window_size <= 0:
            raise ValueError("max_capacity_prompt must exceed window_size")
        self.method = method
        self.num_hidden_layers = int(num_hidden_layers)
        self.layer_idx = int(layer_idx)
        self.max_capacity_prompt = int(max_capacity_prompt)
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.pooling = pooling
        self.beta = int(beta)
        self.gqa_score_agg = gqa_score_agg
        self.retained_len: Optional[int] = None
        self.compressed = False

    def layer_capacity(self, q_len: int) -> int:
        """Retained PAST positions for this layer (excludes the recent window).

        SnapKV: uniform ``C - w``. PyramidKV: arithmetic pyramid from
        ``max_num`` (layer 0) down by ``steps`` per layer, clipped so the whole
        past fits, and falling back to the uniform budget for prompts shorter
        than ``2(C - w)`` (exactly the reference schedule)."""
        C, w = self.max_capacity_prompt, self.window_size
        if self.method == "snapkv":
            return C - w
        min_num = (C - w) // self.beta
        max_num = (C - w) * 2 - min_num
        if max_num >= q_len - w:
            max_num = q_len - w
            min_num = (C - w) * 2 - max_num
        if q_len < (C - w) * 2:
            return C - w
        steps = (max_num - min_num) // max(1, self.num_hidden_layers - 1)
        return max_num - self.layer_idx * steps

    def compress(self, key_states, query_states, value_states):
        """Return the (K, V) to STORE for the decode phase."""
        q_len = query_states.shape[-2]
        if q_len < self.max_capacity_prompt:
            self.compressed = False
            return key_states, value_states          # no compression fires
        capacity = self.layer_capacity(q_len)
        capacity = max(1, min(capacity, q_len - self.window_size))
        scores = observation_window_scores(
            query_states, key_states, window_size=self.window_size,
            kernel_size=self.kernel_size, pooling=self.pooling,
            gqa_score_agg=self.gqa_score_agg)
        self.compressed = True
        return _gather_topk_kv(key_states, value_states, scores, capacity,
                               self.window_size)


# --------------------------------------------------------------------------- #
# install / uninstall (register an attention implementation; never patch a class)
# --------------------------------------------------------------------------- #
def _kvcompress_attention_forward(module, query, key, value, attention_mask,
                                  **kwargs):
    """Attention wrapper: exact full-prefill attention, compressed STORED cache.

    The stock forward has already RoPE'd Q/K and appended them to the cache, so
    ``key``/``value`` here are the full (uncompressed) prefill tensors. We run the
    UNDERLYING attention on them (exact — same kernel the model would have used),
    then — on the prefill call only — overwrite the layer's cache with the
    compressed K/V. Decode calls (``q_len == 1``) pass straight through: the
    appended token attends over the compressed cache.
    """
    base_impl = getattr(module.config, "_comem_kvc_base_attn_impl", "sdpa")
    base = ALL_ATTENTION_FUNCTIONS.get_interface(base_impl,
                                                 ALL_ATTENTION_FUNCTIONS["sdpa"])
    out, weights = base(module, query, key, value, attention_mask, **kwargs)
    cluster = getattr(module, "kv_cluster", None)
    cache = getattr(module, "_kvc_cache", None)
    if cluster is not None and cache is not None and query.shape[-2] > 1:
        k_comp, v_comp = cluster.compress(key, query, value)
        layer = cache.layers[module.layer_idx]
        layer.keys, layer.values = k_comp, v_comp
        cluster.retained_len = int(k_comp.shape[-2])
    return out, weights


ALL_ATTENTION_FUNCTIONS.register(_ATTN_IMPL, _kvcompress_attention_forward)
# Reuse the underlying impl's mask factory so mask shape / is-causal-skip
# semantics stay exactly stock (resolved per-install, see below).
ALL_MASK_ATTENTION_FUNCTIONS.register(_ATTN_IMPL,
                                      ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])


def install_kv_compression(model, method: str, *,
                           max_capacity_prompt: int = COMEM_READ_BUDGET,
                           window_size: int = 32, kernel_size: int = 5,
                           pooling: str = "avgpool", beta: int = 20,
                           gqa_score_agg: str = "mean") -> dict:
    """Attach a :class:`KVCompressCluster` to every attention layer and switch the
    model to the KV-compression attention implementation.

    ``method`` in ``{"snapkv", "pyramidkv"}``; ``max_capacity_prompt`` is the total
    retained tokens per layer (including ``window_size``). The model's existing
    attention implementation is remembered and used as the wrapper's underlying
    kernel, so the prefill attention is byte-for-byte what it would have been.
    Returns the resolved config dict. Reverse with
    :func:`uninstall_kv_compression`.
    """
    inner = getattr(model, "model", model)
    num_hidden_layers = int(inner.config.num_hidden_layers)
    for layer in inner.layers:
        attn = layer.self_attn
        attn.kv_cluster = KVCompressCluster(
            method, num_hidden_layers=num_hidden_layers, layer_idx=attn.layer_idx,
            max_capacity_prompt=max_capacity_prompt, window_size=window_size,
            kernel_size=kernel_size, pooling=pooling, beta=beta,
            gqa_score_agg=gqa_score_agg)
        attn._kvc_cache = None
    cfg = inner.config
    prev = getattr(cfg, "_attn_implementation", "sdpa")
    if prev == _ATTN_IMPL:
        raise RuntimeError("KV compression is already installed on this model; "
                           "call uninstall_kv_compression first")
    cfg._comem_kvc_prev_attn_impl = prev
    # The wrapper delegates to the mask factory registered for _ATTN_IMPL (SDPA's),
    # so only SDPA-family kernels keep stock mask semantics here.
    cfg._comem_kvc_base_attn_impl = prev
    cfg._attn_implementation = _ATTN_IMPL
    return {"method": method, "max_capacity_prompt": int(max_capacity_prompt),
            "window_size": int(window_size), "kernel_size": int(kernel_size),
            "pooling": pooling, "beta": int(beta), "gqa_score_agg": gqa_score_agg,
            "num_hidden_layers": num_hidden_layers, "base_attn_impl": prev}


def uninstall_kv_compression(model):
    """Restore the stock attention implementation and drop attached clusters."""
    inner = getattr(model, "model", model)
    cfg = inner.config
    cfg._attn_implementation = getattr(cfg, "_comem_kvc_prev_attn_impl", "sdpa")
    for attr in ("_comem_kvc_prev_attn_impl", "_comem_kvc_base_attn_impl"):
        if hasattr(cfg, attr):
            delattr(cfg, attr)
    for layer in inner.layers:
        attn = layer.self_attn
        for attr in ("kv_cluster", "_kvc_cache"):
            if hasattr(attn, attr):
                delattr(attn, attr)


def _bind_cache(model, cache):
    """Point every layer's cluster at ``cache`` so it can write back the
    compressed K/V (the attention interface only receives the module)."""
    inner = getattr(model, "model", model)
    for layer in inner.layers:
        layer.self_attn._kvc_cache = cache


def retained_kv_stats(cache) -> dict:
    """Per-layer retained KV length + total K+V bytes held in ``cache``.

    This is the on-device store the decode phase pays for — the number the
    equal-retained-budget diagnostic compares against CoMem's read pack."""
    total = 0
    per_layer = []
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or keys.numel() == 0:
            per_layer.append(0)
            continue
        total += keys.numel() * keys.element_size()
        total += values.numel() * values.element_size()
        per_layer.append(int(keys.shape[-2]))
    return {"kv_bytes": total, "per_layer_retained_len": per_layer,
            "min_retained_len": min(per_layer) if per_layer else None,
            "max_retained_len": max(per_layer) if per_layer else None,
            "mean_retained_len": (sum(per_layer) / len(per_layer)
                                  if per_layer else None)}


# --------------------------------------------------------------------------- #
# explicit greedy decode over the compressed cache (TRUE logical positions)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_kvcompress(model, input_ids, *, max_new_tokens: int, eos_token_id=None,
                        stats=None):
    """Greedy generation with the KV-compression hijack installed.

    Full-prefill attention is EXACT (over uncompressed K/V); only the stored cache
    is compressed. We do NOT use ``model.generate``: it derives the decode position
    from the cache length, which is wrong once the cache is compressed. ``stats``
    (if given) is filled with the retained-KV audit + the prompt length and
    ``full_prompt_seen=True`` (this family MUST prefill the whole prompt).

    Returns the generated token ids (1-D LongTensor, NEW tokens only).
    """
    from transformers.cache_utils import DynamicCache

    prompt_len = int(input_ids.shape[1])
    eos = set()
    if eos_token_id is not None:
        eos = ({int(e) for e in eos_token_id} if isinstance(eos_token_id, (list, tuple, set))
               else {int(eos_token_id)})

    cache = DynamicCache(config=model.config)
    _bind_cache(model, cache)
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    if stats is not None:
        stats.update(retained_kv_stats(cache))
        stats["prompt_tokens"] = prompt_len
        stats["full_prompt_seen"] = True

    next_id = int(out.logits[0, -1].argmax().item())
    generated = [next_id]
    for step in range(1, max_new_tokens):
        if next_id in eos:
            break
        # TRUE logical position of the token being fed, NOT the (compressed)
        # cache length — the retained keys carry their original prefill RoPE.
        pos = torch.tensor([[prompt_len + step - 1]], device=input_ids.device)
        tok = torch.tensor([[next_id]], dtype=torch.long, device=input_ids.device)
        out = model(input_ids=tok, past_key_values=cache, use_cache=True,
                    position_ids=pos)
        next_id = int(out.logits[0, -1].argmax().item())
        generated.append(next_id)
    return torch.tensor(generated, dtype=torch.long, device=input_ids.device)


# --------------------------------------------------------------------------- #
# faithfulness self-test (CPU, tiny random Qwen3, no weights)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_self_test(tol: float = 1e-4, verbose: bool = True) -> bool:
    """Gates (both methods):

      (A) **stock equivalence.** With a prompt SHORTER than the budget no
          compression fires, so the hijacked logits must equal stock attention
          (max|diff| < tol) — proves the wrapper never perturbs the model.
      (B) **budget honoured.** With a prompt LONGER than the budget, SnapKV keeps
          exactly the budget on every layer (uniform), and PyramidKV keeps a
          non-increasing per-layer schedule whose MEAN is the budget (its
          pyramid deliberately gives lower layers more than the budget and
          upper layers less — the average is what is held fixed).
    """
    from .selftest import build_tiny_qwen3
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    model, _ = build_tiny_qwen3(n_layers=6, hidden=64, vocab=256)
    model = model.to(torch.float32).eval()
    budget, window = 64, 8
    short_ids = torch.randint(2, 256, (1, budget - 8))
    long_ids = torch.randint(2, 256, (1, 4 * budget))
    ref_short = model(input_ids=short_ids, use_cache=False).logits.float()

    results = {}
    for method in METHODS:
        cfg = install_kv_compression(model, method, max_capacity_prompt=budget,
                                     window_size=window)
        try:
            # (A) short prompt -> no compression -> bit-comparable to stock
            cache = DynamicCache(config=model.config)
            _bind_cache(model, cache)
            got = model(input_ids=short_ids, past_key_values=cache,
                        use_cache=True).logits.float()
            diff = (got - ref_short).abs().max().item()
            fired = any(layer.self_attn.kv_cluster.compressed
                        for layer in model.model.layers)

            # (B) long prompt -> budget honoured (uniform / pyramidal-on-average)
            cache = DynamicCache(config=model.config)
            _bind_cache(model, cache)
            model(input_ids=long_ids, past_key_values=cache, use_cache=True)
            lens = retained_kv_stats(cache)["per_layer_retained_len"]
            mean_len = sum(lens) / len(lens)
            if method == "snapkv":
                # uniform == exactly the budget on every layer
                budget_ok = set(lens) == {budget}
            else:
                # pyramidal: non-increasing with depth, averaging the budget up to
                # the reference schedule's integer-floor residue in ``steps``
                # (<= (L-1)/2 tokens; 0.2% at the paper's 6657/36-layer setting).
                slack = max(2.0, 0.02 * budget)
                budget_ok = (all(a >= b for a, b in zip(lens, lens[1:]))
                             and lens[0] > lens[-1]
                             and abs(mean_len - budget) <= slack)
        finally:
            uninstall_kv_compression(model)
        results[method] = (diff, fired, lens, mean_len, budget_ok)

    ok = all((d < tol) and (not f) and b for d, f, _l, _m, b in results.values())
    if verbose:
        print("=" * 72)
        print(f"CoMem KV-compression self-test (tiny Qwen3, fp32, budget={budget}, "
              f"window={window}, tol={tol:.0e})")
        print("=" * 72)
        for method, (diff, fired, lens, mean_len, budget_ok) in results.items():
            passA = (diff < tol) and not fired
            print(f"  (A) {method:>10} short prompt == stock attention : {diff:.3e}  "
                  f"compression_fired={fired}  {'PASS' if passA else 'FAIL'}")
            print(f"  (B) {method:>10} retained/layer {lens} mean={mean_len:.1f} "
                  f"(budget {budget})  {'PASS' if budget_ok else 'FAIL'}")
        print("-" * 72)
        print(f"KVCOMPRESS SELF-TEST: {'ALL PASS' if ok else 'FAILURE'}")
        print("=" * 72)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_self_test() else 1)
