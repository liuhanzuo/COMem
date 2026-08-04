"""CacheBlend-style full-depth chunk-KV baseline for CoMem.

CacheBlend (Yao et al., **EuroSys'25**, arXiv:2405.16444) is the strong member of
the chunk-KV / position-independent-caching family (Prompt Cache, Block-Attention,
TurboRAG, EPIC): cache the **full per-layer K/V of every chunk**, then at query
time concatenate the retrieved chunks' caches, repair their positions, and
**selectively recompute** a small fraction of tokens to restore cross-chunk
attention. It is the natural single-variable control for CoMem, because the ONLY
thing that changes is the **cache object**:

============================  ==========================  ===================
arm                           stored per token             Qwen3-8B bf16
============================  ==========================  ===================
CacheBlend-style (this)       full ``L``-layer K/V         144 KiB/token
CoMem (``resume_j=12``)       one depth-``j`` residual     8 KiB/token
============================  ==========================  ===================

(``2 (K+V) · L · n_kv · head_dim · 2 B``; for Qwen3-8B ``2·36·8·128·2 = 147456``
= 144 KiB exactly. Note ``n_kv`` = KV heads under GQA, **not** the 32 query
heads — using those would overstate it 4x.) Everything else is held identical to
flagship CoMem: same selector, chunk size, top-k, sink, and pack order. So
CacheBlend does **not** compress storage; it wins only on prefill/TTFT, and any
prefill-latency claim must be reported next to its 144 KiB/token tier.

Pipeline (faithful CacheBlend, not naive KV concat)
---------------------------------------------------
1. **Per-chunk full-depth prefill** (:meth:`CacheBlend.prefill_chunk_full`) —
   each chunk is contextualised in ISOLATION over ``layers[0:L]`` with a
   chunk-local causal mask and chunk-local RoPE ``0:T``, keeping every layer's
   K/V.
2. **Concat + global RoPE reindex** (:meth:`CacheBlend.concat_kv_reindex`) — the
   classic TurboRAG/PIC failure point. HF stores POST-RoPE keys at chunk-local
   positions, and RoPE is a rotation (``R(a)·R(b) = R(a+b)``), so moving a chunk
   to global pack offset ``Δ`` needs exactly one extra uniform rotation
   ``k_global = k_local·cos(Δ) + rotate_half(k_local)·sin(Δ)`` applied to every
   token of the chunk. Values are NOT rotated. Getting this wrong silently
   corrupts positions; gate (A) of :func:`run_self_test` proves it exact.
3. **Selective recompute / HKVD** (:meth:`CacheBlend.read`) — the load-bearing
   step. Bootstrap layer 0 fresh over all packed tokens, rank context tokens by
   the deviation of their fresh layer-0 keys from the reused ones, and take
   ``R = sink ∪ query ∪ top-⌈r·n_ctx⌉``; for layers ``1..L-1`` forward ONLY those
   ``|R|`` tokens, overwriting the blended cache at the ``R`` positions and
   reusing the cached K/V elsewhere. Dropping this step would reduce the baseline
   to naive KV concat — a strawman that must not be shipped under the method's
   name.
4. **Decode** over the blended full-depth cache: plain single-token decode across
   all ``L`` layers (no resume split).

``recompute_ratio`` (``r``) is the only knob. ``r=1.0`` recomputes everything and
therefore reduces exactly to a vanilla full-context prefill — that identity is
the self-test gate. ``r=0.0`` is the pure-reuse (naive-concat) lower bound.

Clean-room note
---------------
There is no public CacheBlend implementation for this backbone/transformers
generation, so steps 1-4 are implemented from the paper's specification; the
recompute set is seeded once at the bootstrap layer and held fixed across layers
(the "faithful-minimal" variant), rather than re-ranked every few layers. Report
it as a *CacheBlend-style* baseline with that deviation stated.

Run ``python -m comem.cacheblend`` for the CPU correctness gate.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch

from transformers.cache_utils import DynamicCache

from . import selectors as _sel
from .model import CoMem


class CacheBlend:
    """Full-depth chunk-KV read over a stock backbone (see the module docstring).

    Built from a :class:`comem.CoMem` so it reuses CoMem's verified primitives
    (``_as_ids`` / ``_make_mask_and_rope`` / ``_run_layers`` / ``_layer_out_hidden``
    / ``_decode_attn_mask`` and the ``embed_tokens / layers / norm / rotary_emb /
    lm_head`` accessors) and therefore runs the SAME backbone, mask and RoPE
    conventions as the CoMem arm it is compared against. ``resume_j`` is
    irrelevant here — this path always executes all ``L`` layers — so the CoMem
    instance may be built at any ``j``.

    ``recompute_ratio`` is the default ``r`` for :meth:`generate_from_ids`, whose
    signature mirrors :meth:`CoMem.generate_from_ids` so an eval driver can swap
    the two arms by swapping one object.
    """

    def __init__(self, comem: CoMem, recompute_ratio: float = 0.15):
        if not isinstance(comem, CoMem):
            raise TypeError(f"CacheBlend needs a comem.CoMem instance; got {type(comem)!r}")
        if not (0.0 <= float(recompute_ratio) <= 1.0):
            raise ValueError(f"recompute_ratio must be in [0, 1]; got {recompute_ratio}")
        self.cm = comem
        self.recompute_ratio = float(recompute_ratio)
        self.model = comem.model
        self.config = comem.config
        self.num_layers = comem.num_layers
        self.device = comem.device
        self.dtype = comem.dtype
        self.tokenizer = comem.tokenizer

    # ------------------------------------------------------------------ #
    # storage accounting
    # ------------------------------------------------------------------ #
    def kv_bytes_per_tok(self, dtype_bytes: Optional[int] = None) -> int:
        """Full-depth per-token KV store size in bytes (GQA-correct):
        ``2 (K+V) · num_layers · num_key_value_heads · head_dim · dtype_bytes``.
        Qwen3-8B bf16: ``2*36*8*128*2 = 147456`` B = 144 KiB/token."""
        if dtype_bytes is None:
            dtype_bytes = 2 if self.dtype in (torch.bfloat16, torch.float16) else 4
        n_kv = int(getattr(self.config, "num_key_value_heads",
                           self.config.num_attention_heads))
        head_dim = int(getattr(self.config, "head_dim",
                               self.config.hidden_size // self.config.num_attention_heads))
        return 2 * self.num_layers * n_kv * head_dim * int(dtype_bytes)

    # ------------------------------------------------------------------ #
    # RoPE reindex primitives
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """RoPE ``rotate_half`` (matches transformers ``apply_rotary_pos_emb``)."""
        d = x.shape[-1] // 2
        return torch.cat((-x[..., d:], x[..., :d]), dim=-1)

    def _rope_delta_cos_sin(self, offset: int, like: torch.Tensor):
        """UN-scaled RoPE ``(cos, sin)`` for the single absolute position ``offset``.

        ``rotary_emb`` returns cos/sin ALREADY multiplied by ``attention_scaling``
        (1.0 for default rope, != 1 for e.g. YaRN). A reindex is a pure rotation
        composition, so the extra factor must be divided out — otherwise a
        non-default rope would scale the reindexed keys by ``s`` and silently
        corrupt them. Returns cos/sin broadcastable to ``[1, n_kv, T, head_dim]``.
        """
        pos = torch.tensor([[int(offset)]], device=like.device)
        cos, sin = self.cm.rotary_emb(like, position_ids=pos)      # [1, 1, head_dim]
        s = float(getattr(self.cm.rotary_emb, "attention_scaling", 1.0) or 1.0)
        if s != 1.0:
            cos, sin = cos / s, sin / s
        return cos.unsqueeze(1).to(like.dtype), sin.unsqueeze(1).to(like.dtype)

    def _rotate_k_by_offset(self, k: torch.Tensor, offset: int) -> torch.Tensor:
        """Reindex cached (post-RoPE) keys from chunk-local ``0:T`` to global
        ``offset:offset+T`` with one extra uniform rotation ``R(offset)``.

        Every token of the chunk gets the SAME delta: token ``i`` was rotated at
        local position ``i`` and belongs at global ``offset+i``, a uniform shift.
        Offset 0 (the sink at pack position 0) is the identity."""
        if int(offset) == 0:
            return k
        cos, sin = self._rope_delta_cos_sin(int(offset), k)
        return (k * cos) + (self._rotate_half(k) * sin)

    # ------------------------------------------------------------------ #
    # (1) per-chunk full-depth prefill
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def prefill_chunk_full(self, token_ids, rope_start: int = 0):
        """Full-depth chunk prefill — the CacheBlend precompute primitive.

        Mirrors :meth:`CoMem.write_prefill` but over the FULL band ``layers[0:L]``,
        keeping every layer's K/V. The chunk is contextualised in ISOLATION: a
        chunk-local causal mask over ``T`` tokens and RoPE positions
        ``rope_start:rope_start+T``. ``rope_start=0`` is the normal precompute;
        the self-test uses ``rope_start=Δ`` to build a "prefilled directly at the
        global offset" reference for the reindex gate. The causal mask stays
        chunk-local regardless of the RoPE offset (RoPE is relative, so
        intra-chunk attention is offset-invariant).

        Returns ``(kv_layers, T)`` with ``kv_layers[l] = (K, V)``, each
        ``[1, n_kv, T, head_dim]`` post q/k-norm and post-RoPE — exactly the space
        the backbone attention caches."""
        ids = self.cm._as_ids(token_ids)
        T = ids.shape[1]
        embeds = self.cm.embed_tokens(ids)
        mask_positions = torch.arange(T, device=self.device).unsqueeze(0)
        rope_positions = mask_positions + int(rope_start)
        causal_mask, _ = self.cm._make_mask_and_rope(embeds, mask_positions)
        position_embeddings = self.cm.rotary_emb(embeds, position_ids=rope_positions)
        cache = DynamicCache(config=self.config)
        self.cm._run_layers(embeds, slice(0, self.num_layers), causal_mask,
                            rope_positions, position_embeddings,
                            past_key_values=cache, use_cache=True)
        kv_layers = [(cache.layers[l].keys, cache.layers[l].values)
                     for l in range(self.num_layers)]
        return kv_layers, T

    # ------------------------------------------------------------------ #
    # (2) concat in pack order + global RoPE reindex
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def concat_kv_reindex(self, chunk_kv_list: Sequence, chunk_offsets: Sequence[int]):
        """Concatenate per-chunk full-depth K/V in pack order + reindex the keys.

        ``chunk_kv_list`` is the pack-ordered (``[sink ; ctx... ; query]``) list of
        :meth:`prefill_chunk_full` outputs; ``chunk_offsets[c]`` is chunk ``c``'s
        GLOBAL start position (running sum of chunk lengths). Per layer, each
        chunk's keys are re-rotated by its offset and concatenated on the sequence
        axis; values are concatenated as-is. Returns a length-``L`` list of
        ``(K, V)`` each ``[1, n_kv, H, head_dim]`` with ``H = sum(T_c)`` —
        positioned exactly as one contiguous full-context prefill (gate (A))."""
        merged: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for l in range(self.num_layers):
            k_parts, v_parts = [], []
            for c, kv_layers in enumerate(chunk_kv_list):
                K, V = kv_layers[l]
                k_parts.append(self._rotate_k_by_offset(K, int(chunk_offsets[c])))
                v_parts.append(V)
            merged.append((torch.cat(k_parts, dim=-2), torch.cat(v_parts, dim=-2)))
        return merged

    # ------------------------------------------------------------------ #
    # (3) selective recompute (HKVD)
    # ------------------------------------------------------------------ #
    def _sparse_attn_mask(self, keep_bool: torch.Tensor):
        """Coerce a ``[Sq, Skv]`` bool keep-mask to the attention impl's format.
        SDPA / FlashAttention take a bool mask (True = attend); eager wants an
        additive float mask (0 / -inf). Shape ``[1, 1, Sq, Skv]``."""
        keep = keep_bool.view(1, 1, *keep_bool.shape)
        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        if attn_impl in ("sdpa", "flash_attention_2", "flash_attention_3"):
            return keep
        mask = torch.zeros(keep.shape, dtype=self.dtype, device=keep.device)
        return mask.masked_fill(~keep, torch.finfo(self.dtype).min)

    @torch.no_grad()
    def read(self, pack_ids, merged_kv: Sequence, sink_len: int, query_len: int,
             recompute_ratio: float, stats=None):
        """CacheBlend read: blended full-depth KV + selective recompute.

        ``pack_ids`` is the ``[1, H]`` packed sequence ``[sink ; ctx ; query]`` whose
        per-chunk KV produced ``merged_kv``; ``sink_len`` / ``query_len`` mark the
        sink prefix and query tail, the middle being context.

        1. bootstrap layer 0 fresh over all ``H`` tokens (global RoPE ``0:H``,
           standard causal) -> ``h1`` for every token + fresh layer-0 K/V;
        2. deviation per context token ``= ||K0_fresh - K0_reused||_2`` over
           (kv-heads x head_dim); ``R = sink ∪ query ∪ top-⌈r·n_ctx⌉``;
        3. layers ``1..L-1``: forward ONLY the ``|R|`` tokens; each layer overwrites
           the reused cache at the ``R`` positions with fresh K/V and reuses cached
           K/V elsewhere, with ``R`` queries attending to the full ``H`` cache
           causally by GLOBAL position;
        4. norm + lm_head over ``R``.

        At ``r=1.0`` ``R`` is every token, so this reduces to a standard
        full-context prefill (the self-test identity). Returns
        ``(logits_R [1, |R|, V], R_idx [|R|], mixed)`` where ``mixed`` is the final
        length-``L`` list of full-``H`` ``(K, V)`` caches (for decode)."""
        ids = self.cm._as_ids(pack_ids)
        H = ids.shape[1]
        L = self.num_layers
        embeds = self.cm.embed_tokens(ids)
        positions = torch.arange(H, device=self.device).unsqueeze(0)   # global 0:H
        full_mask, full_pe = self.cm._make_mask_and_rope(embeds, positions)

        # (1) bootstrap: full layer 0 over all H tokens (fresh KV for all).
        boot_cache = DynamicCache(config=self.config)
        h1 = self.cm._run_layers(embeds, slice(0, 1), full_mask, positions, full_pe,
                                 past_key_values=boot_cache, use_cache=True)
        freshK0 = boot_cache.layers[0].keys            # [1, n_kv, H, head_dim]
        freshV0 = boot_cache.layers[0].values

        # (2) per-context-token deviation vs the reused (reindexed) layer-0 keys.
        ctx_start = int(sink_len)
        ctx_end = H - int(query_len)
        n_ctx = max(0, ctx_end - ctx_start)
        reusedK0 = merged_kv[0][0].to(freshK0.dtype)
        dev = (freshK0 - reusedK0).float().pow(2).sum(dim=(1, 3)).sqrt().squeeze(0)  # [H]

        r = float(recompute_ratio)
        n_recompute = max(0, min(int(math.ceil(r * n_ctx)) if n_ctx > 0 else 0, n_ctx))
        R = torch.zeros(H, dtype=torch.bool, device=self.device)
        if ctx_start > 0:
            R[:ctx_start] = True            # sink forced (identity reuse anyway)
        R[ctx_end:] = True                  # query forced (always recomputed)
        if n_recompute > 0:
            top = torch.topk(dev[ctx_start:ctx_end], n_recompute).indices + ctx_start
            R[top] = True
        R_idx = torch.nonzero(R, as_tuple=False).squeeze(1)   # sorted global positions

        # (3) selective recompute over layers 1..L-1. Layer 0 is fully fresh; the
        # rest start from the reused (reindexed) cache with R overwritten in place.
        mixed: List[Tuple[torch.Tensor, torch.Tensor]] = [(freshK0, freshV0)]
        for l in range(1, L):
            mixed.append((merged_kv[l][0].clone(), merged_kv[l][1].clone()))

        hidden_R = h1[:, R_idx, :]                      # [1, |R|, d]
        rope_R = positions[:, R_idx]                    # global positions of R
        pe_R = self.cm.rotary_emb(hidden_R, position_ids=rope_R)
        # R query at global pos p attends to a cache key at global pos k iff k <= p.
        col = torch.arange(H, device=self.device).view(1, H)
        attn_mask = self._sparse_attn_mask(col <= R_idx.view(-1, 1))

        for l in range(1, L):
            cache_l = _SparseWriteCache(mixed[l][0], mixed[l][1], R_idx)
            out = self.cm.layers[l](
                hidden_R, attention_mask=attn_mask, position_ids=rope_R,
                position_embeddings=pe_R, past_key_values=cache_l, use_cache=True)
            hidden_R = self.cm._layer_out_hidden(out)
            mixed[l] = (cache_l.keys, cache_l.values)

        logits_R = self.cm.lm_head(self.cm.norm(hidden_R))   # [1, |R|, V]

        if stats is not None:
            stats["cacheblend_kv_bytes_per_tok"] = self.kv_bytes_per_tok()
            stats["recompute_ratio"] = r
            stats["n_recompute_ctx"] = int(n_recompute)
            stats["n_context_tokens"] = int(n_ctx)
            stats["pack_len"] = int(H)
        return logits_R, R_idx, mixed

    # ------------------------------------------------------------------ #
    # (4) decode over the blended full-depth cache
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode_cache(self, mixed: Sequence) -> DynamicCache:
        """Seed a standard ``DynamicCache`` from a blended full-``H`` KV cache (it IS
        a valid past for every layer) so a plain single-token decode can extend it."""
        cache = DynamicCache(config=self.config)
        for l in range(self.num_layers):
            cache.update(mixed[l][0], mixed[l][1], l)
        return cache

    @torch.no_grad()
    def decode_step(self, token_id, decode_cache: DynamicCache, pack_pos: int):
        """One O(1) decode step over the blended cache (all ``L`` layers).
        Returns next-token logits ``[1, 1, V]``; the cache grows by one position."""
        ids = torch.tensor([[int(token_id)]], device=self.device, dtype=torch.long)
        emb = self.cm.embed_tokens(ids)
        t_pos = torch.tensor([[int(pack_pos)]], device=self.device)
        t_pe = self.cm.rotary_emb(emb, position_ids=t_pos)
        t_mask = self.cm._decode_attn_mask(int(pack_pos) + 1)
        hidden = self.cm._run_layers(emb, slice(0, self.num_layers), t_mask, t_pos,
                                     t_pe, past_key_values=decode_cache,
                                     use_cache=True)
        return self.cm.lm_head(self.cm.norm(hidden))

    # ------------------------------------------------------------------ #
    # end-to-end: same entry-point contract as CoMem.generate_from_ids
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_from_ids(self, input_ids: torch.Tensor, *, chunk_size: int = 512,
                          max_new_tokens: int = 20, selector: str = "iter_bm25",
                          topk: int = 12, sink_tokens: str = "bos",
                          needle_chunk_set=None, bare_question_ids=None,
                          no_retrieval: bool = False, stats=None,
                          iter_rounds: int = 0, iter_hop_topk: int = 4,
                          iter_score: str = "meanpool", iter_conf_ratio: float = 0.3,
                          iter_max_chunks: int = 64, dense_retriever=None,
                          recompute_ratio: Optional[float] = None,
                          use_kv_cache: bool = True, tokenizer=None) -> str:
        """Chunk a prompt (query == the LAST chunk), select ``topk`` context chunks
        with the SAME selector as CoMem, then run the CacheBlend read + decode.

        Signature-compatible with :meth:`CoMem.generate_from_ids` (plus
        ``recompute_ratio``, defaulting to ``self.recompute_ratio``) so a driver can
        swap the two arms by swapping the object, and the comparison stays
        single-variable: identical chunking, selector, pack order, sink and EOS
        contract; only the cache object differs. ``no_retrieval`` packs ALL context
        chunks (the no-retrieval floor); ``use_kv_cache`` is accepted for signature
        compatibility — this path always decodes over the blended KV cache."""
        tok = tokenizer if tokenizer is not None else self.tokenizer
        r = self.recompute_ratio if recompute_ratio is None else float(recompute_ratio)
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"recompute_ratio must be in [0, 1]; got {r}")
        tokens = input_ids[0]
        chunks = list(tokens.split(chunk_size))
        context_chunks = chunks[:-1]
        query_chunk = chunks[-1]

        sink_ids = None
        if sink_tokens == "bos":
            bos_id, _ = self.cm._bos_eos(tok, fallback_first=int(tokens[0].item()))
            sink_ids = [int(bos_id)]

        if no_retrieval:
            sel_idx = list(range(len(context_chunks)))
        else:
            # Selectors scoring cached h_j are undefined here (there is no depth-j
            # store), so this arm runs the lexical/dense selectors CoMem is
            # compared at.
            if selector in ("reader_attn", "iter_reader_attn"):
                raise ValueError(
                    f"selector {selector!r} scores cached depth-j hiddens, which the "
                    "full-depth CacheBlend path does not build; use a lexical/dense "
                    "selector (the arm is compared at CoMem's iter_bm25 anyway)")
            sel_idx = _sel.select_context_chunk_indices(
                selector, context_chunks, bare_question_ids or [], topk,
                needle_chunk_set, iter_rounds=iter_rounds,
                iter_hop_topk=iter_hop_topk, iter_score=iter_score,
                iter_conf_ratio=iter_conf_ratio, iter_max_chunks=iter_max_chunks,
                dense_retriever=dense_retriever, dense_tokenizer=tok)

        # (1) full-depth per-chunk prefill in pack order [sink ; ctx ; query]
        chunk_kv_list, chunk_offsets, seg_ids = [], [], []
        offset, sink_len = 0, 0
        if sink_ids is not None:
            kv, T = self.prefill_chunk_full(sink_ids)
            chunk_kv_list.append(kv); chunk_offsets.append(offset); offset += T
            seg_ids.append(torch.tensor(sink_ids, device=self.device, dtype=torch.long))
            sink_len = T
        for i in sel_idx:
            kv, T = self.prefill_chunk_full(context_chunks[i])
            chunk_kv_list.append(kv); chunk_offsets.append(offset); offset += T
            seg_ids.append(context_chunks[i].to(self.device).long().view(-1))
        q_ids = query_chunk.to(self.device).long().view(-1)
        kv, query_len = self.prefill_chunk_full(q_ids)
        chunk_kv_list.append(kv); chunk_offsets.append(offset); offset += query_len
        seg_ids.append(q_ids)
        H = offset
        pack_ids = torch.cat(seg_ids).view(1, -1)          # [1, H]

        # (2) + (3) blend + selective recompute
        merged = self.concat_kv_reindex(chunk_kv_list, chunk_offsets)
        logits_R, _R_idx, mixed = self.read(pack_ids, merged, sink_len, query_len,
                                            r, stats=stats)
        if stats is not None:
            stats["read_len"] = int(H)
            stats["n_selected_chunks"] = len(sel_idx)
            stats["n_context_chunks"] = len(context_chunks)

        # (4) greedy decode over the blended full-depth cache
        _, eos_id = self.cm._bos_eos(tok)
        next_logits = logits_R[0, -1].float()    # last query position -> first token
        if eos_id is not None:
            next_logits[eos_id] = float("-inf")  # step 0 never emits EOS
        next_tok = int(next_logits.argmax().item())
        generated = [next_tok]
        decode_cache = self.decode_cache(mixed)
        pack_pos = H
        for _step in range(1, max_new_tokens):
            logits = self.decode_step(next_tok, decode_cache, pack_pos)
            pack_pos += 1
            next_tok = int(logits[0, -1].float().argmax().item())
            if eos_id is not None and next_tok == eos_id:
                break
            generated.append(next_tok)
        if tok is not None:
            return tok.decode(generated, skip_special_tokens=True).strip()
        return generated


class _SparseWriteCache:
    """Minimal duck-typed per-layer cache for the selective-recompute pass.

    Pre-loaded with the reused (RoPE-reindexed) K/V for ALL ``H`` packed positions.
    The backbone attention calls ``past_key_values.update(k_fresh, v_fresh,
    layer_idx)`` with the freshly computed K/V of the ``|R|`` recompute tokens only;
    we overwrite the reused cache at the ``R`` positions and return the FULL
    ``[1, n_kv, H, head_dim]`` K/V (fresh at ``R``, reused elsewhere), so the ``|R|``
    queries attend over the whole blended context. ``.update(...)`` is the only
    Cache method a decoder layer's attention calls, so nothing else is needed."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor,
                 recompute_pos: torch.Tensor):
        self.keys = keys
        self.values = values
        self._pos = recompute_pos

    def update(self, key_states, value_states, *args, **kwargs):
        self.keys = self.keys.clone()
        self.values = self.values.clone()
        self.keys[:, :, self._pos, :] = key_states.to(self.keys.dtype)
        self.values[:, :, self._pos, :] = value_states.to(self.values.dtype)
        return self.keys, self.values


# --------------------------------------------------------------------------- #
# correctness gate (CPU, tiny random Qwen3, no weights)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_self_test(tol: float = 1e-4, verbose: bool = True) -> bool:
    """Gates:

      (A) **RoPE reindex is exact.** Prefilling a chunk chunk-locally then rotating
          by its global offset == prefilling it DIRECTLY at those global positions
          (max|dK|, max|dV| < tol). This is the TurboRAG/PIC failure point.
      (B) **r=1.0 == vanilla full prefill.** With every token recomputed the read
          reproduces ``model(pack_ids)`` (max|logit diff| < tol) and ``R`` covers
          every packed position — so blend + reindex + selective recompute
          degenerates correctly at the ceiling.
      (C) **r=0.0 is finite.** The pure-reuse (naive-concat) floor produces no
          NaN/Inf.
    """
    from .selftest import build_tiny_qwen3, _TinyTok

    torch.manual_seed(0)
    model, _ = build_tiny_qwen3(n_layers=4, hidden=64, vocab=256)
    model = model.to(torch.float32).eval()
    tok = _TinyTok(256)
    cb = CacheBlend(CoMem(model, resume_j=0, tokenizer=tok))

    def rid(n):
        return torch.randint(2, 256, (1, n))

    sink_ids = [tok.bos_token_id]
    c1, c2, c3, q = rid(13), rid(9), rid(11), rid(7)

    # (A) reindex exactness on c2 at its pack offset.
    offset = len(sink_ids) + c1.shape[1]
    kv_local, _ = cb.prefill_chunk_full(c2, rope_start=0)
    kv_ref, _ = cb.prefill_chunk_full(c2, rope_start=offset)
    reidx = cb.concat_kv_reindex([kv_local], [offset])
    maxK = max((reidx[l][0] - kv_ref[l][0]).abs().max().item()
               for l in range(cb.num_layers))
    maxV = max((reidx[l][1] - kv_ref[l][1]).abs().max().item()
               for l in range(cb.num_layers))

    # Build the full pack for the end-to-end gates.
    segs = [torch.tensor([sink_ids]), c1, c2, c3, q]
    pack_ids = torch.cat(segs, dim=1)
    kvs, offs, o = [], [], 0
    for seg in segs:
        kv, T = cb.prefill_chunk_full(seg)
        kvs.append(kv); offs.append(o); o += T
    merged = cb.concat_kv_reindex(kvs, offs)
    q_len = q.shape[1]
    ref = cb.cm.full_forward_logits(pack_ids).float()

    # (B) r=1.0 == vanilla full prefill.
    logits1, R1, _ = cb.read(pack_ids, merged, len(sink_ids), q_len, 1.0)
    diff_full = (logits1.float() - ref).abs().max().item()
    r1_all = int(R1.numel()) == int(pack_ids.shape[1])

    # (C) r=0.0 finite.
    logits0, _, _ = cb.read(pack_ids, merged, len(sink_ids), q_len, 0.0)
    finite0 = bool(torch.isfinite(logits0).all().item())

    okA = maxK < tol and maxV < tol
    okB = diff_full < tol and r1_all
    ok = okA and okB and finite0
    if verbose:
        print("=" * 72)
        print(f"CoMem CacheBlend self-test (tiny Qwen3, fp32, L={cb.num_layers}, "
              f"tol={tol:.0e})")
        print("=" * 72)
        print(f"  kv_bytes/tok (full-depth) = {cb.kv_bytes_per_tok()}")
        print(f"  (A) RoPE reindex exact      : max|dK|={maxK:.3e} max|dV|={maxV:.3e}"
              f"  {'PASS' if okA else 'FAIL'}")
        print(f"  (B) r=1.0 == full prefill   : max|logit diff|={diff_full:.3e} "
              f"R=all:{r1_all}  {'PASS' if okB else 'FAIL'}")
        print(f"  (C) r=0.0 finite (no NaN)   : {finite0}  "
              f"{'PASS' if finite0 else 'FAIL'}")
        print("-" * 72)
        print(f"CACHEBLEND SELF-TEST: {'ALL PASS' if ok else 'FAILURE'}")
        print("=" * 72)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_self_test() else 1)
